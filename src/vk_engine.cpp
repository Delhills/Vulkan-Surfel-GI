
#include "vk_engine.h"

#include "VkBootstrap.h"
#include "vk_initializers.h"
#include "vk_textures.h"
#include "window.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

VulkanEngine* VulkanEngine::engine = nullptr;

// vector with all possible paths
std::vector<std::string> searchPaths;

VulkanEngine::VulkanEngine()
{
	engine = this;
	_window = new Window();
}

void VulkanEngine::init()
{
	_mode =	HYBRID;

	_window->init("Vulkan Pinut", 1920, 1080);

	searchPaths = {
		"data/shaders/output",
		"data",
		"data/meshes",
		"data/textures"
	};

	// load core vulkan structures
	init_vulkan();

	// create the swapchain
	init_swapchain();

	init_upload_commands();

	_scene = new Scene();
	_scene->create_scene();

	// Add necessary features to the engine
	init_ray_tracing();

	renderer = new Renderer(_scene);

	init_imgui();

	mouse_locked = false;
	SDL_ShowCursor(!mouse_locked);

	debugTarget = 0;

	// Everything went fine
	_isInitialized = true;
}

void VulkanEngine::cleanup()
{
	if (_isInitialized) {

		for (auto& frames : renderer->_frames)
			vkWaitForFences(_device, 1, &frames._renderFence, VK_TRUE, 1000000000);

		//vkDeviceWaitIdle(_device);

		_mainDeletionQueue.flush();

		vkDestroySurfaceKHR(_instance, _surface, nullptr);
		vkDestroyDevice(_device, nullptr);
		vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
		vkDestroyInstance(_instance, nullptr);

		_window->clean();
	}
}

glm::vec3 uniformSamplerCone(float r1, float r2, float cosThetaMax)
{
	float cosTheta = (1.0 - r1) + r1 * cosThetaMax;
	float sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
	float phi = r2 * 2 * 3.1415;
	return glm::vec3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
}

void VulkanEngine::run()
{
	SDL_Event e;

	double lastFrame = 0.0f;
	while (!_bQuit)
	{
		double currentTime = SDL_GetTicks();
		double dt = (currentTime - lastFrame);
		lastFrame = currentTime;

		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_QUIT) _bQuit = true;
			_window->handleEvent(e, dt);
		}

		update(dt);

		renderer->render_gui();
		switch (_mode)
		{
		case DEFERRED:
			renderer->render();
			break;
		case RAYTRACING:
			renderer->raytrace();
			break;
		case HYBRID:
			renderer->rasterize_hybrid();
			break;
		default:
			break;
		}

		_frameNumber++;
	}
}

void VulkanEngine::update(const float dt)
{
	_window->input_update();
	updateFrame();
	updateCameraMatrices();

	// Skybox Matrix followin the camera
	if (_skyboxFollow) {
		glm::mat4 skyMatrix = glm::translate(glm::mat4(1), _scene->_camera->_position);
		void* skyMatData;
		vmaMapMemory(_allocator, renderer->_skyboxBuffer._allocation, &skyMatData);
		memcpy(skyMatData, &skyMatrix, sizeof(glm::mat4));
		vmaUnmapMemory(_allocator, renderer->_skyboxBuffer._allocation);
	}

	// TODO unify with the deferred update buffer
	void* rtLightData;
	vmaMapMemory(_allocator, renderer->_lightBuffer._allocation, &rtLightData);
	uboLight* rtLightUBO = (uboLight*)rtLightData;
	for (int i = 0; i < _scene->_lights.size(); i++)
	{
		_scene->_lights[i]->update();
		Light* l = _scene->_lights[i];
		if (l->type == DIRECTIONAL_LIGHT) {
			rtLightUBO[i].color		= glm::vec4(l->color.x, l->color.y, l->color.z, l->intensity);
			rtLightUBO[i].position	= glm::vec4(l->position.x, l->position.y, l->position.z, -1);
		}
		else {
			rtLightUBO[i].color		= glm::vec4(l->color.x, l->color.y, l->color.z, l->intensity);
			rtLightUBO[i].position	= glm::vec4(l->position.x, l->position.y, l->position.z, l->maxDistance);
			rtLightUBO[i].radius	= l->radius;
		}
	}
	memcpy(rtLightData, rtLightUBO, sizeof(rtLightUBO));
	vmaUnmapMemory(_allocator, renderer->_lightBuffer._allocation);

	// Shadow samples
	void* samplesData;
	vmaMapMemory(_allocator, renderer->_shadowSamplesBuffer._allocation, &samplesData);
	std::memcpy(samplesData, &_samples, sizeof(int));
	vmaUnmapMemory(_allocator, renderer->_shadowSamplesBuffer._allocation);

	// TODO: MEMORY LEAK and update only when instances changed
	// Rebuild instances matrix for TLAS
	int instanceIndex = 0;
	renderer->_tlas.clear();
	for (Object* obj : _scene->_entities)
	{
		if (!obj->prefab->_root.empty())
		{
			for (Node* root : obj->prefab->_root)
			{
				root->node_to_instance(renderer->_tlas, instanceIndex, obj->m_matrix);
			}
		}
	}
	
	renderer->buildTlas(renderer->_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR, true);	
}

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
	VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);

	VkCommandBuffer cmd;
	VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &cmd));

	VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	function(cmd);

	VK_CHECK(vkEndCommandBuffer(cmd));
	VkSubmitInfo submit = vkinit::submit_info(&cmd);

	VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._uploadFence));

	vkWaitForFences(_device, 1, &_uploadContext._uploadFence, VK_TRUE, 1000000000);
	vkResetFences(_device, 1, &_uploadContext._uploadFence);

	vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}

void VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, AllocatedBuffer &buffer, const bool destroy)
{
	VkBufferCreateInfo bufferInfo = vkinit::buffer_create_info(allocSize, usage);
	
	VmaAllocationCreateInfo vmaallocinfo = {};
	vmaallocinfo.usage = memoryUsage;

	VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocinfo,
		&buffer._buffer, 
		&buffer._allocation, 
		nullptr));

	if (destroy)
	{
		_mainDeletionQueue.push_function([=]() {
			vmaDestroyBuffer(_allocator, buffer._buffer, buffer._allocation);
			});
	}
}

// PRIVATE ----------------------------------------

void VulkanEngine::init_vulkan()
{
	vkb::InstanceBuilder builder;
	auto system_info_ret = vkb::SystemInfo::get_system_info();
	auto system_info = system_info_ret.value();
	
	auto inst_ret = builder.set_app_name("Vulkan Pinut")
		.request_validation_layers(true)
		.require_api_version(1, 2, 0)
		.use_default_debug_messenger()
		.enable_extension("VK_KHR_get_physical_device_properties2")
		.build();

	vkb::Instance vkb_inst = inst_ret.value();

	_instance = vkb_inst.instance;
	_debug_messenger = vkb_inst.debug_messenger;

	SDL_Vulkan_CreateSurface(_window->_handle, _instance, &_surface);

	std::vector<const char*> required_device_extensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
		VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,

		// VkRay
		VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
		VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,

		// Required by VK_KHR_acceleration_structure
		VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
		VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
		
		// Required by VK_KHR_raytracing_pipeline
		VK_KHR_SPIRV_1_4_EXTENSION_NAME,

		// Required by VK_KHR_spirv_1_4
		VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
	};

	vkb::PhysicalDeviceSelector selector{ vkb_inst };
	vkb::PhysicalDevice physicalDevice = selector
		.set_minimum_version(1, 1)
		.set_surface(_surface)
		.add_required_extensions(required_device_extensions)
		.select()
		.value();

	vkb::DeviceBuilder deviceBuilder{ physicalDevice };

	uint32_t count;
	vkEnumerateDeviceExtensionProperties(physicalDevice.physical_device, nullptr, &count, nullptr);
	std::vector<VkExtensionProperties> props(count);
	vkEnumerateDeviceExtensionProperties(physicalDevice.physical_device, nullptr, &count, props.data());

	get_enabled_features();

	vkb::Device vkbDevice = deviceBuilder.add_pNext(deviceCreatepNextChain).build().value();

	_device = vkbDevice.device;
	_gpu	= physicalDevice.physical_device;

	_graphicsQueue			= vkbDevice.get_queue(vkb::QueueType::graphics).value();
	_graphicsQueueFamily	= vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	vkGetPhysicalDeviceMemoryProperties(_gpu, &_memoryProperties);

	// Initialize the memory allocator
	VmaAllocatorCreateInfo allocatorInfo = {};
	allocatorInfo.physicalDevice	= _gpu;
	allocatorInfo.device			= _device;
	allocatorInfo.instance			= _instance;
	allocatorInfo.flags				= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator(&allocatorInfo, &_allocator);

	vkGetPhysicalDeviceProperties(_gpu, &_gpuProperties);
	std::cout << "The GPU has a minimum buffer alignment of " << _gpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;
	_mainDeletionQueue.push_function([=]() {
		vmaDestroyAllocator(_allocator);
		});
}
	
void VulkanEngine::init_swapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{ _gpu, _device, _surface };

	vkb::Swapchain vkbSwapchain = swapchainBuilder
		.use_default_format_selection()
		.set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
		.add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
		.set_desired_extent(_window->getWidth(), _window->getHeight())
		.build()
		.value();

	// Store swapchain and related images
	_swapchain				= vkbSwapchain.swapchain;
	_swapchainImages		= vkbSwapchain.get_images().value();
	_swapchainImageViews	= vkbSwapchain.get_image_views().value();

	_swapchainImageFormat = vkbSwapchain.image_format;

	_mainDeletionQueue.push_function([=]() {
		vkDestroySwapchainKHR(_device, _swapchain, nullptr);
		});

	VkExtent3D depthImageExtent = {
		_window->getWidth(),
		_window->getHeight(),
		1
	};

	_depthFormat = VK_FORMAT_D32_SFLOAT;
	VkImageCreateInfo depth_info = vkinit::image_create_info(
		_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);
	
	VmaAllocationCreateInfo dimg_allocinfo = {};
	dimg_allocinfo.usage			= VMA_MEMORY_USAGE_GPU_ONLY;
	dimg_allocinfo.requiredFlags	= VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(_allocator, &depth_info, &dimg_allocinfo, 
		&_depthImage._image, &_depthImage._allocation, nullptr);

	VkImageViewCreateInfo dview_info = vkinit::image_view_create_info(
		_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT);

	VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

	_mainDeletionQueue.push_function([=]() {
		vkDestroyImageView(_device, _depthImageView, nullptr);
		vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
	});
}

void VulkanEngine::recreate_swapchain()
{
	int width = 0, height = 0;
	SDL_Event e;
	//while (_window->isMinimized()) {
	//	SDL_PollEvent(&e);
	//}

	//vkDeviceWaitIdle(_device);

	clean_swapchain();

	init_swapchain();
	init_upload_commands();
	SDL_GetWindowSize(_window->_handle, &width, &height);
	_window->setWidth(width);
	_window->setHeight(height);
	renderer->recreate_renderer();
	//renderer = new Renderer();

}

void VulkanEngine::clean_swapchain()
{
	for(size_t i = 0; i < _swapchainImages.size(); i++) {
		vkDestroyFramebuffer(_device, renderer->_framebuffers[i], nullptr);
		vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
	}
	for (size_t i = 0; i < FRAME_OVERLAP; i++) {
		vkFreeCommandBuffers(_device, renderer->_frames[i]._commandPool, 1, &renderer->_frames[i]._mainCommandBuffer);
	}

	//vkFreeCommandBuffers(_device, renderer->_commandPool, 1, &renderer->_offscreenComandBuffer);

	vkDestroyPipeline(_device, renderer->_finalPipeline, nullptr);
	vkDestroyPipeline(_device, renderer->_forwardPipeline, nullptr);
	vkDestroyPipeline(_device, renderer->_offscreenPipeline, nullptr);
	vkDestroyPipeline(_device, renderer->_rtPipeline, nullptr);

	vkDestroyPipelineLayout(_device, renderer->_finalPipelineLayout, nullptr);
	vkDestroyPipelineLayout(_device, renderer->_forwardPipelineLayout, nullptr);
	vkDestroyPipelineLayout(_device, renderer->_offscreenPipelineLayout, nullptr);
	vkDestroyPipelineLayout(_device, renderer->_rtPipelineLayout, nullptr);

	vkDestroyRenderPass(_device, renderer->_renderPass, nullptr);
	vkDestroyRenderPass(_device, renderer->_forwardRenderPass, nullptr);
	vkDestroyRenderPass(_device, renderer->_offscreenRenderPass, nullptr);

	vkDestroySwapchainKHR(_device, _swapchain, nullptr);
}

void VulkanEngine::init_ray_tracing()
{
	// Requesting ray tracing properties
	_rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	VkPhysicalDeviceProperties2 deviceProperties2{};
	deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	deviceProperties2.pNext = &_rtProperties;
	vkGetPhysicalDeviceProperties2(_gpu, &deviceProperties2);

	// Requesting ray tracing features
	_asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	VkPhysicalDeviceFeatures2 deviceFeatures2{};
	deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	deviceFeatures2.pNext = &_asFeatures;
	vkGetPhysicalDeviceFeatures2(_gpu, &deviceFeatures2);

	vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(_device, "vkGetBufferDeviceAddressKHR"));
}

void VulkanEngine::init_upload_commands()
{
	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);

	VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));

	VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info();

	VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._uploadFence));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
		vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
	});
}

void VulkanEngine::init_imgui()
{
	// Create descriptor pool for IMGUI
	VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
		{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
		{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	VkDescriptorPoolCreateInfo pool_info = {};
	pool_info.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.pNext			= nullptr;
	pool_info.maxSets		= 1000;
	pool_info.flags			= VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	pool_info.poolSizeCount = std::size(pool_sizes);
	pool_info.pPoolSizes	= pool_sizes;

	VkDescriptorPool imguiPool;
	VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

	// Initialize imgui library
	ImGui::CreateContext();
	
	ImGui_ImplSDL2_InitForVulkan(_window->_handle);

	ImGui_ImplVulkan_InitInfo initInfo = {};
	initInfo.Instance		= _instance;
	initInfo.PhysicalDevice = _gpu;
	initInfo.Device			= _device;
	initInfo.Queue			= _graphicsQueue;
	initInfo.DescriptorPool = imguiPool;
	initInfo.MinImageCount	= 3;
	initInfo.ImageCount		= 3;

	ImGui_ImplVulkan_Init(&initInfo, renderer->_renderPass);

	immediate_submit([&](VkCommandBuffer cmd) {
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
	});

	ImGui_ImplVulkan_DestroyFontUploadObjects();

	_mainDeletionQueue.push_function([=]() {
		ImGui_ImplVulkan_Shutdown();
		vkDestroyDescriptorPool(_device, imguiPool, nullptr);
	});
}

void VulkanEngine::create_attachment(VkFormat format, VkImageUsageFlagBits usage, Texture* texture)
{
	VkImageAspectFlags aspectMask = 0;
	VkImageLayout imageLayout;

	if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
	{
		aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	}
	if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
	{
		aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; // | VK_IMAGE_ASPECT_STENCIL_BIT;
		imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	}

	assert(aspectMask > 0);

	VkExtent3D extent = { _window->getWidth(), _window->getHeight(), 1 };
	VkImageCreateInfo imageInfo = vkinit::image_create_info(format, usage | VK_IMAGE_USAGE_SAMPLED_BIT, extent);

	VmaAllocationCreateInfo memAlloc = {};
	memAlloc.usage			= VMA_MEMORY_USAGE_GPU_ONLY;
	memAlloc.requiredFlags	= VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(_allocator, &imageInfo, &memAlloc, 
		&texture->image._image, &texture->image._allocation, nullptr);

	VkImageViewCreateInfo imageView = vkinit::image_view_create_info(format, texture->image._image, aspectMask);

	VK_CHECK(vkCreateImageView(_device, &imageView, nullptr, &texture->imageView));
}

VkCommandBuffer VulkanEngine::create_command_buffer(VkCommandBufferLevel level, bool begin)
{
	VkCommandBuffer commandBuffer;
	if (!_commandPool)
	{
		VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamily);
		vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_commandPool);
		_mainDeletionQueue.push_function([=]() {
			vkDestroyCommandPool(_device, _commandPool, nullptr);
			});
	}

	VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(_commandPool, 1, level);
	VK_CHECK(vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer));

	if (begin) {
		VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(0);
		VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
	}

	return commandBuffer;
}

uint32_t VulkanEngine::getBufferDeviceAddress(VkBuffer buffer)
{
	VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
	bufferDeviceAddressInfo.sType	= VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAddressInfo.buffer	= buffer;
	return vkGetBufferDeviceAddressKHR(_device, &bufferDeviceAddressInfo);
}

void VulkanEngine::get_enabled_features()
{
	enabledIndexingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
	enabledIndexingFeatures.runtimeDescriptorArray = VK_TRUE;
	enabledIndexingFeatures.pNext = nullptr;

	enabledBufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
	enabledBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
	enabledBufferDeviceAddressFeatures.pNext = &enabledIndexingFeatures;

	enabledRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	enabledRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
	enabledRayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = VK_TRUE;
	enabledRayTracingPipelineFeatures.pNext = &enabledBufferDeviceAddressFeatures;

	enabledAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	enabledAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
	enabledAccelerationStructureFeatures.pNext = &enabledRayTracingPipelineFeatures;

	deviceCreatepNextChain = &enabledAccelerationStructureFeatures;
}

bool VulkanEngine::load_shader_module(const char* filePath, VkShaderModule* outShaderModule)
{
	// open file in binary mode with cursor at the end
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		return false;
	}

	// Find up the size by looking up the location of the cursor
	// Since it is at the end, it gives the number of bytes
	size_t fileSize = (size_t)file.tellg();

	// spirv expects the buffer to be on uint32, make sure to reserve a int vector big enough for the entire file
	std::vector<uint32_t> buffer((fileSize / sizeof(uint32_t)));

	// Put the cursor back to the beginning
	file.seekg(0);

	// Load the entire file to the buffer
	file.read((char*)buffer.data(), fileSize);

	// The file is loaded to the buffer, we can now close it
	file.close();

	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.pNext = nullptr;

	// CodeSize is in bytes, multiply the ints of the buffer by the size of int
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

	// Check if creation goes well
	VkShaderModule module;
	if (vkCreateShaderModule(_device, &createInfo, nullptr, &module) != VK_SUCCESS)
		return false;

	*outShaderModule = module;
	return true;

}

VkPipelineShaderStageCreateInfo VulkanEngine::load_shader_stage(const char* filePath, VkShaderModule* outShaderModule, VkShaderStageFlagBits stage)
{
	if (!load_shader_module(filePath, outShaderModule)) {
		std::cout << "Failed to create shader module!" << std::endl;
	}

	VkPipelineShaderStageCreateInfo shaderStage{};
	shaderStage.sType	= VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage	= stage;
	shaderStage.module	= *outShaderModule;
	shaderStage.pName	= "main";

	return shaderStage;
}

size_t VulkanEngine::pad_uniform_buffer_size(size_t originalSize)
{
	// Calculate required alignment based on minimun device offset alignment
	size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
	size_t alignedSize = originalSize;
	if (minUboAlignment > 0) {
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}
	return alignedSize;
}

void VulkanEngine::updateFrame()
{
	static glm::mat4 refMatrix;

	const auto& m = _scene->_camera->getView();

	if (memcmp(&refMatrix[0][0], &m[0][0], sizeof(glm::mat4)) != 0)
	{
		resetFrame();
		refMatrix = m;
	}
	//if(_denoise_frame < 100)
	_denoise_frame++;
}

void VulkanEngine::resetFrame()
{
	_denoise_frame = -1;
}

void VulkanEngine::updateCameraMatrices()
{
	static glm::mat4 prevView;
	static glm::mat4 prevProj;

	glm::mat4 view			= _scene->_camera->getView();
	glm::mat4 projection	= _scene->_camera->getProjection((float)_window->getWidth() / (float)_window->getHeight());
	projection[1][1] *= -1;

	if (memcmp(&prevView[0][0], &view[0][0], sizeof(glm::mat4)) != 0 || memcmp(&prevProj[0][0], &projection[0][0], sizeof(glm::mat4)) != 0)
	{
		void* camPosData;
		vmaMapMemory(_allocator, renderer->_cameraPositionBuffer._allocation, &camPosData);
		memcpy(camPosData, &_scene->_camera->_position, sizeof(glm::vec3));
		vmaUnmapMemory(_allocator, renderer->_cameraPositionBuffer._allocation);

		// Fill the GPU camera data struct
		GPUCameraData cameraData;
		cameraData.view			= view;
		cameraData.projection	= projection;
		cameraData.prevView		= prevView;
		cameraData.prevProj		= prevProj;

		prevView = view;
		prevProj = projection;

		void* data;
		vmaMapMemory(_allocator, renderer->_cameraBuffer._allocation, &data);
		memcpy(data, &cameraData, sizeof(GPUCameraData));
		vmaUnmapMemory(_allocator, renderer->_cameraBuffer._allocation);
	}

	void* frameData;
	vmaMapMemory(_allocator, renderer->_frameCountBuffer._allocation, &frameData);
	memcpy(frameData, &_denoise_frame, sizeof(int));
	vmaUnmapMemory(_allocator, renderer->_frameCountBuffer._allocation);

	// Copy RAY-TRACING camera, it need the inverse
	// --------------------------------------------
	// TODO: rethink how frame can be passed each frame and update the matrix only if changed

	// Copy ray tracing camera, it need the inverse
	RTCameraData rtCamera;
	rtCamera.invProj = glm::inverse(projection);
	rtCamera.invView = glm::inverse(view);
	rtCamera.frame = !_denoise ? glm::vec4(0) : glm::vec4(_denoise_frame);

	//std::cout << _denoise_frame << std::endl;

	void* rtCameraData;
	vmaMapMemory(_allocator, renderer->_rtCameraBuffer._allocation, &rtCameraData);
	memcpy(rtCameraData, &rtCamera, sizeof(RTCameraData));
	vmaUnmapMemory(_allocator, renderer->_rtCameraBuffer._allocation);
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device, VkRenderPass pass)
{
	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.pNext = nullptr;

	viewportState.viewportCount = 1;
	viewportState.pViewports	= &_viewport;
	viewportState.scissorCount	= 1;
	viewportState.pScissors		= &_scissor;

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = nullptr;

	pipelineInfo.stageCount				= _shaderStages.size();
	pipelineInfo.pStages				= _shaderStages.data();
	pipelineInfo.pVertexInputState		= &_vertexInputInfo;
	pipelineInfo.pInputAssemblyState	= &_inputAssembly;
	pipelineInfo.pViewportState			= &viewportState;
	pipelineInfo.pRasterizationState	= &_rasterizer;
	pipelineInfo.pMultisampleState		= &_multisampling;
	pipelineInfo.pColorBlendState		= &_colorBlendStateInfo;
	pipelineInfo.layout					= _pipelineLayout;
	pipelineInfo.renderPass				= pass;
	pipelineInfo.subpass				= 0;
	pipelineInfo.basePipelineHandle		= VK_NULL_HANDLE;
	pipelineInfo.pDepthStencilState =	&_depthStencil;

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		std::cout << "Failed to create pipeline!" << std::endl;
		return VK_NULL_HANDLE;
	}

	return newPipeline;

}

VkPipelineDepthStencilStateCreateInfo vkinit::depth_stencil_create_info(
	bool bDepthTest,
	bool bDepthWrite,
	VkCompareOp compareOp)
{
	VkPipelineDepthStencilStateCreateInfo info = {};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	info.pNext = nullptr;

	info.depthTestEnable		= bDepthTest ? VK_TRUE : VK_FALSE;
	info.depthWriteEnable		= bDepthWrite ? VK_TRUE : VK_FALSE;
	info.depthCompareOp			= bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS;
	info.depthBoundsTestEnable	= VK_FALSE;
	info.minDepthBounds			= 0.0f;
	info.maxDepthBounds			= 1.0f;
	info.stencilTestEnable		= VK_FALSE;

	return info;
}
