
#include <renderer.h>
#include <vk_engine.h>
#include <vk_initializers.h>
#include <ctime>
#include "window.h"
#include "vk_utils.h"

extern std::vector<std::string> searchPaths;

Renderer::Renderer(Scene* scene)
{
	device		= &VulkanEngine::engine->_device;
	swapchain	= &VulkanEngine::engine->_swapchain;
	frameNumber	= &VulkanEngine::engine->_frameNumber;
	gizmoEntity	= nullptr;
	_scene = scene;

	init_commands();
	init_render_pass();
	init_forward_render_pass();
	init_offscreen_render_pass();
	init_framebuffers();
	init_offscreen_framebuffers();
	init_sync_structures();

	load_data_to_gpu();
	
	create_storage_image();
	init_descriptors();
	init_deferred_descriptors();
	//init_forward_pipeline();
	init_deferred_pipelines();
	build_previous_command_buffer();

	// Ray tracing
	vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(*device, "vkCreateAccelerationStructureKHR"));
	vkBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(*device, "vkBuildAccelerationStructuresKHR"));
	vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(*device, "vkGetAccelerationStructureBuildSizesKHR"));
	vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(*device, "vkGetAccelerationStructureDeviceAddressKHR"));
	vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(*device, "vkCmdBuildAccelerationStructuresKHR"));
	vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(*device, "vkGetRayTracingShaderGroupHandlesKHR"));
	vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(*device, "vkCreateRayTracingPipelinesKHR"));
	vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(*device, "vkCmdTraceRaysKHR"));
	vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(*device, "vkDestroyAccelerationStructureKHR"));


	// post
	//create_post_renderPass();
	//create_post_framebuffers();
	//create_post_descriptor();
	//create_post_pipeline();
	
	// VKRay
	create_bottom_acceleration_structure();
	create_top_acceleration_structure();
	//create_rt_descriptors();
	//create_hybrid_descriptors();
	//init_raytracing_pipeline();
	//create_shader_binding_table();
	//build_shadow_command_buffer();
	//build_raytracing_command_buffers();
	//build_hybrid_command_buffers();
	//create_shadow_descriptors();
	//init_compute_pipeline();
	//build_compute_command_buffer();
	create_SurfelGi_resources();
	surfel_position();
	prepare_indirect();
	//grid_reset();
	update_surfels();
	//grid_offset();
	//surfel_binning();
	surfel_ray_tracing();
	surfel_shade();
	//todo_de_nuevo();
}

void Renderer::init_commands()
{
	// Create a command pool for commands to be submitted to the graphics queue
	VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(VulkanEngine::engine->_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(VulkanEngine::engine->_graphicsQueueFamily);

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VK_CHECK(vkCreateCommandPool(*device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

		// Allocate the default command buffer that will be used for rendering
		VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

		VK_CHECK(vkAllocateCommandBuffers(*device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

		VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
			vkDestroyCommandPool(*device, _frames[i]._commandPool, nullptr);
			});
	}

	VK_CHECK(vkCreateCommandPool(*device, &uploadCommandPoolInfo, nullptr, &_commandPool));
	VK_CHECK(vkCreateCommandPool(*device, &commandPoolInfo, nullptr, &_resetCommandPool));

	VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(_resetCommandPool, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	VK_CHECK(vkAllocateCommandBuffers(*device, &allocInfo, &_offscreenComandBuffer));

	VkCommandBufferAllocateInfo cmdDeferredAllocInfo = vkinit::command_buffer_allocate_info(_commandPool);
	VkCommandBufferAllocateInfo cmdPostAllocInfo = vkinit::command_buffer_allocate_info(_commandPool);
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_rtCommandBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_hybridCommandBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_shadowCommandBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_denoiseCommandBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_GridResetCmdBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_PrepareIndirectCmdBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_SurfelPositionCmd));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_UpdateSurfelsCmdBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_GridOffsetCmdBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_SurfelBinningCmdBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_SurfelRTXCommandBuffer));
	VK_CHECK(vkAllocateCommandBuffers(*device, &cmdPostAllocInfo, &_SurfelShadeCmdBuffer));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyCommandPool(*device, _commandPool, nullptr);
		vkDestroyCommandPool(*device, _resetCommandPool, nullptr);
		});
}

void Renderer::init_render_pass()
{
	VkAttachmentDescription color_attachment = {};
	color_attachment.format				= VulkanEngine::engine->_swapchainImageFormat;
	color_attachment.samples			= VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp				= VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
	// Do not care about stencil at the moment
	color_attachment.stencilLoadOp		= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp		= VK_ATTACHMENT_STORE_OP_DONT_CARE;
	// We do not know or care about the starting layout of the attachment
	color_attachment.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;
	// After the render pass ends, the image has to be on a layout ready for display
	color_attachment.finalLayout		= VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	// Attachment number will index into the pAttachments array in the parent renderpass itself
	color_attachment_ref.attachment		= 0;
	color_attachment_ref.layout			= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.format				= VulkanEngine::engine->_depthFormat;
	depth_attachment.samples			= VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.flags				= 0;
	depth_attachment.loadOp				= VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp		= VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp		= VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment		= 1;
	depth_attachment_ref.layout			= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	std::array<VkSubpassDependency, 2> dependencies{};
	dependencies[0].srcSubpass			= VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass			= 0;
	dependencies[0].srcStageMask		= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask		= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask		= VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask		= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags		= VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass			= 0;
	dependencies[1].dstSubpass			= VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask		= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask		= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask		= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask		= VK_ACCESS_MEMORY_READ_BIT;
	dependencies[1].dependencyFlags		= VK_DEPENDENCY_BY_REGION_BIT;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint			= VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount		= 1;
	subpass.pColorAttachments			= &color_attachment_ref;
	subpass.pDepthStencilAttachment		= &depth_attachment_ref;

	VkAttachmentDescription attachments[2] = { color_attachment, depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.pNext				= nullptr;
	render_pass_info.attachmentCount	= 2;
	render_pass_info.pAttachments		= &attachments[0];
	render_pass_info.subpassCount		= 1;
	render_pass_info.pSubpasses			= &subpass;
	render_pass_info.dependencyCount	= 1;
	render_pass_info.pDependencies		= dependencies.data();

	VkDevice device = VulkanEngine::engine->_device;
	VK_CHECK(vkCreateRenderPass(device, &render_pass_info, nullptr, &_renderPass));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(device, _renderPass, nullptr);
		});
}

void Renderer::init_forward_render_pass()
{
	VkAttachmentDescription color_attachment = {};
	color_attachment.format			= VulkanEngine::engine->_swapchainImageFormat;
	color_attachment.samples		= VK_SAMPLE_COUNT_1_BIT;
	color_attachment.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
	color_attachment.storeOp		= VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout	= VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0;
	color_attachment_ref.layout		= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depth_attachment = {};
	// Depth attachment
	depth_attachment.flags			= 0;
	depth_attachment.format			= VulkanEngine::engine->_depthFormat;
	depth_attachment.samples		= VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp		= VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout	= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount	= 1;
	subpass.pColorAttachments		= &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	//1 dependency, which is from "outside" into the subpass. And we can read or write color
	VkSubpassDependency dependency = {};
	dependency.srcSubpass		= VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass		= 0;
	dependency.srcStageMask		= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask	= 0;
	dependency.dstStageMask		= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask	= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	//array of 2 attachments, one for the color, and other for depth
	VkAttachmentDescription attachments[2] = { color_attachment, depth_attachment };

	VkRenderPassCreateInfo render_pass_info = {};
	render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_info.attachmentCount	= 2;
	render_pass_info.pAttachments		= &attachments[0];
	render_pass_info.subpassCount		= 1;
	render_pass_info.pSubpasses			= &subpass;
	render_pass_info.dependencyCount	= 1;
	render_pass_info.pDependencies		= &dependency;

	VK_CHECK(vkCreateRenderPass(*device, &render_pass_info, nullptr, &_forwardRenderPass));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(*device, _forwardRenderPass, nullptr);
		});
}

void Renderer::init_offscreen_render_pass()
{
	Texture position, normal, albedo, motion, material, emissive, depth;
	VulkanEngine::engine->create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &position);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &normal);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &albedo);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &motion);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &material);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &emissive);
	VulkanEngine::engine->create_attachment(VulkanEngine::engine->_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &depth);

	_deferredTextures.push_back(position);
	_deferredTextures.push_back(normal);
	_deferredTextures.push_back(albedo);
	_deferredTextures.push_back(motion);
	_deferredTextures.push_back(material);
	_deferredTextures.push_back(emissive);
	_deferredTextures.push_back(depth);

	const int nAttachments = _deferredTextures.size();

	std::array<VkAttachmentDescription, 7> attachmentDescs = {};

	// Init attachment properties
	for (uint32_t i = 0; i < nAttachments; i++)
	{
		attachmentDescs[i].samples			= VK_SAMPLE_COUNT_1_BIT;
		attachmentDescs[i].loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachmentDescs[i].storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescs[i].stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescs[i].stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE;
		if (i == nAttachments - 1)
		{
			attachmentDescs[i].initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
			attachmentDescs[i].finalLayout		= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; //ojo
		}
		else
		{
			attachmentDescs[i].initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
			attachmentDescs[i].finalLayout		= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
	}

	// Formats
	attachmentDescs[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachmentDescs[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
	attachmentDescs[2].format = VK_FORMAT_R8G8B8A8_UNORM;
	attachmentDescs[3].format = VK_FORMAT_R16G16_SFLOAT;
	attachmentDescs[4].format = VK_FORMAT_R8G8B8A8_UNORM;
	attachmentDescs[5].format = VK_FORMAT_R8G8B8A8_UNORM;
	attachmentDescs[6].format = VulkanEngine::engine->_depthFormat;

	std::vector<VkAttachmentReference> colorReferences;
	colorReferences.push_back({ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 4, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 5, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkAttachmentReference depthReference;
	depthReference.attachment	= nAttachments - 1;
	depthReference.layout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.pColorAttachments		= colorReferences.data();
	subpass.colorAttachmentCount	= static_cast<uint32_t>(colorReferences.size());
	subpass.pDepthStencilAttachment = &depthReference;

	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	// This dependency transitions the input attachment from color attachment to shader read
	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;


	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType			= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.pNext			= nullptr;
	renderPassInfo.pAttachments		= attachmentDescs.data();
	renderPassInfo.attachmentCount	= static_cast<uint32_t>(attachmentDescs.size());
	renderPassInfo.subpassCount		= 1;
	renderPassInfo.pSubpasses		= &subpass;
	renderPassInfo.dependencyCount	= 2;
	renderPassInfo.pDependencies	= dependencies.data();

	VK_CHECK(vkCreateRenderPass(*device, &renderPassInfo, nullptr, &_offscreenRenderPass));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(*device, _offscreenRenderPass, nullptr);
		for (int i = 0; i < _deferredTextures.size(); i++) {
			vkDestroyImageView(*device, _deferredTextures[i].imageView, nullptr);
			vmaDestroyImage(VulkanEngine::engine->_allocator, _deferredTextures[i].image._image, _deferredTextures[i].image._allocation);
		}
		});
}

FrameData& Renderer::get_current_frame()
{
	return _frames[*frameNumber % FRAME_OVERLAP];
}

void Renderer::rasterize()
{
	ImGui::Render();

	VK_CHECK(vkWaitForFences(*device, 1, &get_current_frame()._renderFence, VK_TRUE, 1000000000));
	VK_CHECK(vkResetFences(*device, 1, &get_current_frame()._renderFence));

	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	vkAcquireNextImageKHR(*device, *swapchain, UINT64_MAX, get_current_frame()._presentSemaphore, VK_NULL_HANDLE, &VulkanEngine::engine->_indexSwapchainImage);

	build_forward_command_buffer();

	VkSubmitInfo submit = {};
	submit.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext				= nullptr;
	submit.pWaitDstStageMask	= waitStages;
	submit.waitSemaphoreCount	= 1;
	submit.pWaitSemaphores		= &get_current_frame()._presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores	= &get_current_frame()._renderSemaphore;
	submit.pCommandBuffers		= &get_current_frame()._mainCommandBuffer; 

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext				= nullptr;
	presentInfo.swapchainCount		= 1;
	presentInfo.pSwapchains			= swapchain;
	presentInfo.waitSemaphoreCount	= 1;
	presentInfo.pWaitSemaphores		= &get_current_frame()._renderSemaphore;
	presentInfo.pImageIndices		= &VulkanEngine::engine->_indexSwapchainImage;

	VK_CHECK(vkQueuePresentKHR(VulkanEngine::engine->_graphicsQueue, &presentInfo));
}

void Renderer::render()
{
	//ImGui::Render();
	
	//std::cout << _scene->_camera->_position.x << ", " << _scene->_camera->_position.y << ", " << _scene->_camera->_position.z << std::endl;

	// Wait until the gpu has finished rendering the last frame. Timeout 1 second
	VK_CHECK(vkWaitForFences(*device, 1, &get_current_frame()._renderFence, VK_TRUE, 1000000000));
	VK_CHECK(vkResetFences(*device, 1, &get_current_frame()._renderFence));

	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkResult result = vkAcquireNextImageKHR(*device, *swapchain, UINT64_MAX, get_current_frame()._presentSemaphore, VK_NULL_HANDLE, &VulkanEngine::engine->_indexSwapchainImage);

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		VulkanEngine::engine->recreate_swapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("Failed to acquire swap chain image");
	}

	VK_CHECK(vkResetCommandBuffer(_offscreenComandBuffer, 0));
	build_previous_command_buffer();

	// First pass
	VkSubmitInfo submit = {};
	submit.sType					= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext					= nullptr;
	submit.pWaitDstStageMask		= waitStages;
	submit.waitSemaphoreCount		= 1;
	submit.pWaitSemaphores			= &get_current_frame()._presentSemaphore;
	submit.signalSemaphoreCount		= 1;
	submit.pSignalSemaphores		= &_shadowSemaphore;
	submit.commandBufferCount		= 1;
	submit.pCommandBuffers			= &_offscreenComandBuffer;

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	
	//build_surfel_position_command_buffer();

	//surfel coverage
	submit.pWaitSemaphores = &_shadowSemaphore;
	submit.pSignalSemaphores = &_SurfelPositionSemaphore;
	submit.pCommandBuffers = &_SurfelPositionCmd;

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	////prepare indirect
	//submit.pWaitSemaphores = &_SurfelPositionSemaphore;
	//submit.pSignalSemaphores = &_PrepareIndirectSemaphore;
	//submit.pCommandBuffers = &_PrepareIndirectCmdBuffer;

	//VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	////grid reset
	//submit.pWaitSemaphores = &_PrepareIndirectSemaphore;
	//submit.pSignalSemaphores = &_GridResetSemaphore;
	//submit.pCommandBuffers = &_GridResetCmdBuffer;

	//VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	//update
	submit.pWaitSemaphores = &_SurfelPositionSemaphore;
	submit.pSignalSemaphores = &_UpdateSurfelsSemaphore;
	submit.pCommandBuffers = &_SurfelRTXCommandBuffer;

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);



	submit.pWaitSemaphores = &_UpdateSurfelsSemaphore;
	submit.pSignalSemaphores = &_SurfelShadeSemaphore;
	submit.pCommandBuffers = &_SurfelShadeCmdBuffer;

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	////Grid Offset
	//submit.pWaitSemaphores = &_UpdateSurfelsSemaphore;
	//submit.pSignalSemaphores = &_GridOffsetSemaphore;
	//submit.pCommandBuffers = &_GridOffsetCmdBuffer;

	//VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	////Grid binning
	//submit.pWaitSemaphores = &_GridOffsetSemaphore;
	//submit.pSignalSemaphores = &_SurfelBinningSemaphore;
	//submit.pCommandBuffers = &_SurfelBinningCmdBuffer;

	//VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	//std::cout << _scene->_camera->_position.x << ", " << _scene->_camera->_position.y << ", " << _scene->_camera->_position.z << std::endl;
	////compute pass
	//submit.pWaitSemaphores = &_SurfelPositionSemaphore;
	////submit.pWaitSemaphores = &_shadowSemaphore;

	//submit.pSignalSemaphores = &_offscreenSemaphore;
	//submit.pCommandBuffers = &_denoiseCommandBuffer;

	//VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	////compute pass
	//submit.pWaitSemaphores = &_offscreenSemaphore;
	//submit.pSignalSemaphores = &_buffer2;
	//submit.pCommandBuffers = &_commandbuffer2;

	//VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));
	//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);


	build_deferred_command_buffer();

	// Second pass
	submit.pWaitSemaphores			= &_SurfelShadeSemaphore;
	submit.pSignalSemaphores		= &get_current_frame()._renderSemaphore;
	submit.pCommandBuffers			= &get_current_frame()._mainCommandBuffer;
	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.pNext				= nullptr;
	presentInfo.swapchainCount		= 1;
	presentInfo.pSwapchains			= swapchain;
	presentInfo.waitSemaphoreCount	= 1;
	presentInfo.pWaitSemaphores		= &get_current_frame()._renderSemaphore;
	presentInfo.pImageIndices		= &VulkanEngine::engine->_indexSwapchainImage;

	result = vkQueuePresentKHR(VulkanEngine::engine->_graphicsQueue, &presentInfo);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		VulkanEngine::engine->recreate_swapchain();
	}
	else if (result != VK_SUCCESS)
		throw std::runtime_error("failed to present swap chain images!");
}

void Renderer::render_gui()
{
	bool changed = false;
	bool changed_material = false;

	// Imgui new frame
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL2_NewFrame(VulkanEngine::engine->_window->_handle);

	ImGui::NewFrame();

	ImGui::Begin("Debug window");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Checkbox("Denoise", &VulkanEngine::engine->_denoise);

	const std::vector<std::string> renderers = { "Deferred Shading Renderer", "Ray Traced Renderer", "Hybrid Shading Renderer" };
	std::vector<const char*> charShadings;
	charShadings.reserve(renderers.size());
	for (size_t i = 0; i < renderers.size(); i++)
	{
		charShadings.push_back(renderers[i].c_str());
	}

	const char* title = "Rendering Mode";
	int renderer = VulkanEngine::engine->_mode;
	if (ImGui::Combo(title, &renderer, &charShadings[0], renderers.size(), renderers.size()))
	{
		std::cout << (renderMode)renderer << std::endl;
		VulkanEngine::engine->_mode = (renderMode)renderer;
	}

	if (VulkanEngine::engine->_mode == DEFERRED)
	{
		const std::vector<std::string> targets = { "Final", "Position", "Normal", "Albedo", "Motion", "Material", "Emissive" };
		std::vector<const char*> charTargets;
		charTargets.reserve(targets.size());
		for (size_t i = 0; i < targets.size(); i++)
		{
			charTargets.push_back(targets[i].c_str());
		}

		title = "Target";
		int index = VulkanEngine::engine->debugTarget;

		if (ImGui::Combo(title, &index, &charTargets[0], targets.size(), targets.size()))
		{
			VulkanEngine::engine->debugTarget = index;
			void* debugData;
			vmaMapMemory(VulkanEngine::engine->_allocator, VulkanEngine::engine->renderer->_debugBuffer._allocation, &debugData);
			memcpy(debugData, &VulkanEngine::engine->debugTarget, sizeof(uint32_t));
			vmaUnmapMemory(VulkanEngine::engine->_allocator, VulkanEngine::engine->renderer->_debugBuffer._allocation);
		}
	}

	ImGui::DragInt("Shadow Samples", &VulkanEngine::engine->_samples, 1.0f, 1, 64);

	for (auto& light : _scene->_lights)
	{
		if (ImGui::TreeNode(&light, "Light")) {
			if (ImGui::Button("Select"))
				gizmoEntity = light;
			changed |= ImGui::SliderFloat3("Position", &((glm::vec3)light->m_matrix[3])[0], -200, 200);
			changed |= ImGui::ColorEdit3("Color", &light->color.x);
			changed |= ImGui::SliderFloat("Intensity", &light->intensity, 0, 1000);
			changed |= ImGui::SliderFloat("Max Distance", &light->maxDistance, 0, 500);
			changed |= ImGui::SliderFloat("Radius", &light->radius, 0, 10);
			ImGui::TreePop();
		}
	}
	for (auto& entity : _scene->_entities)
	{
		if (ImGui::TreeNode(&entity, "Entity")) {
			if (ImGui::Button("Select"))
				gizmoEntity = entity;
			changed_material |= ImGui::SliderFloat3("Color", glm::value_ptr(entity->material->diffuseColor), 0., 1.);
			changed_material |= ImGui::SliderFloat("Metallic", &entity->material->metallicFactor, 0., 1.);
			changed_material |= ImGui::SliderFloat("Roughness", &entity->material->roughnessFactor, 0., 1.);
			ImGui::TreePop();
		}
	}
	ImGui::End();

	if (!gizmoEntity)
		return;

	glm::mat4& matrix = gizmoEntity->m_matrix;
	glm::mat4 aux = matrix;

	ImGuizmo::BeginFrame();

	static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);
	static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::WORLD);
	if (ImGui::IsKeyPressed(90))
		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	if (ImGui::IsKeyPressed(69))
		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	if (ImGui::IsKeyPressed(82)) // r Key
		mCurrentGizmoOperation = ImGuizmo::SCALE;
	if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	ImGui::SameLine();
	if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
		mCurrentGizmoOperation = ImGuizmo::SCALE;
	float matrixTranslation[3], matrixRotation[3], matrixScale[3];
	ImGuizmo::DecomposeMatrixToComponents(&matrix[0][0], matrixTranslation, matrixRotation, matrixScale);
	ImGui::InputFloat3("Tr", matrixTranslation, 3);
	ImGui::InputFloat3("Rt", matrixRotation, 3);
	ImGui::InputFloat3("Sc", matrixScale, 3);
	ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, &matrix[0][0]);


	if (mCurrentGizmoOperation != ImGuizmo::SCALE)
	{
		if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
			mCurrentGizmoMode = ImGuizmo::LOCAL;
		ImGui::SameLine();
		if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
			mCurrentGizmoMode = ImGuizmo::WORLD;
	}
	static bool useSnap(false);
	if (ImGui::IsKeyPressed(83))
		useSnap = !useSnap;
	ImGui::Checkbox("", &useSnap);
	ImGui::SameLine();
	glm::vec3 snap;
	switch (mCurrentGizmoOperation)
	{
	case ImGuizmo::TRANSLATE:
		//snap = config.mSnapTranslation;
		ImGui::InputFloat3("Snap", &snap.x);
		break;
	case ImGuizmo::ROTATE:
		//snap = config.mSnapRotation;
		ImGui::InputFloat("Angle Snap", &snap.x);
		break;
	case ImGuizmo::SCALE:
		//snap = config.mSnapScale;
		ImGui::InputFloat("Scale Snap", &snap.x);
		break;
	}
	ImGuiIO& io = ImGui::GetIO();
	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1712.f / 912.f, 0.1f, 200.0f);
	ImGuizmo::Manipulate(&_scene->_camera->getView()[0][0], &projection[0][0], mCurrentGizmoOperation, mCurrentGizmoMode, &matrix[0][0], NULL, useSnap ? &snap.x : NULL);

	ImGui::EndFrame();

	// If matrix is different, then turn changed to true
	if (memcmp(&matrix[0][0], &aux[0][0], sizeof(glm::mat4)) != 0) {
		changed = true;
	}

	if (changed)
		VulkanEngine::engine->resetFrame();

	// If any material has been modified, update the materials
	if (changed_material)
	{
		std::vector<GPUMaterial> materials;
		for (Material* it : Material::_materials)
		{
			GPUMaterial mat = it->materialToShader();
			materials.push_back(mat);
		}

		void* matData;
		vmaMapMemory(VulkanEngine::engine->_allocator, _matBuffer._allocation, &matData);
		memcpy(matData, materials.data(), sizeof(GPUMaterial) * materials.size());
		vmaUnmapMemory(VulkanEngine::engine->_allocator, _matBuffer._allocation);
	}
}

void Renderer::init_framebuffers()
{
	VkExtent2D extent = { (uint32_t)VulkanEngine::engine->_window->getWidth(), (uint32_t)VulkanEngine::engine->_window->getHeight() };
	VkFramebufferCreateInfo framebufferInfo = vkinit::framebuffer_create_info(_renderPass, extent);

	// Grab how many images we have in the swapchain
	const uint32_t swapchain_imagecount = static_cast<uint32_t>(VulkanEngine::engine->_swapchainImages.size());
	_framebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (unsigned int i = 0; i < swapchain_imagecount; i++)
	{
		VkImageView attachments[2];
		attachments[0] = VulkanEngine::engine->_swapchainImageViews[i];
		attachments[1] = VulkanEngine::engine->_depthImageView;

		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments	= attachments;
		VK_CHECK(vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &_framebuffers[i]));

		VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
			vkDestroyFramebuffer(*device, _framebuffers[i], nullptr);
			vkDestroyImageView(*device, VulkanEngine::engine->_swapchainImageViews[i], nullptr);
			});
	}
}

void Renderer::init_offscreen_framebuffers()
{
	std::array<VkImageView, 7> attachments;
	attachments[0] = _deferredTextures.at(0).imageView;	// Position
	attachments[1] = _deferredTextures.at(1).imageView;	// Normal
	attachments[2] = _deferredTextures.at(2).imageView;	// Color	
	attachments[3] = _deferredTextures.at(3).imageView;	// Motion Vector	
	attachments[4] = _deferredTextures.at(4).imageView;	// Material Properties
	attachments[5] = _deferredTextures.at(5).imageView;	// Emissive Color
	attachments[6] = _deferredTextures.at(6).imageView;	// Depth

	VkExtent2D extent = { (uint32_t)VulkanEngine::engine->_window->getWidth(), (uint32_t)VulkanEngine::engine->_window->getHeight() };

	VkFramebufferCreateInfo framebufferInfo = vkinit::framebuffer_create_info(_offscreenRenderPass, extent);
	framebufferInfo.attachmentCount			= static_cast<uint32_t>(attachments.size());
	framebufferInfo.pAttachments			= attachments.data();

	VK_CHECK(vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &_offscreenFramebuffer));

	VkSamplerCreateInfo sampler = vkinit::sampler_create_info(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	sampler.mipmapMode		= VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler.mipLodBias		= 0.0f;
	sampler.maxAnisotropy	= 1.0f;
	sampler.minLod			= 0.0f;
	sampler.maxLod			= 1.0f;
	sampler.borderColor		= VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	VK_CHECK(vkCreateSampler(*device, &sampler, nullptr, &_offscreenSampler));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyFramebuffer(*device, _offscreenFramebuffer, nullptr);
		vkDestroySampler(*device, _offscreenSampler, nullptr);
		});
}

void Renderer::init_sync_structures()
{
	// Create syncronization structures

	VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
	VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_offscreenSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_rtSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_shadowSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_denoiseSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_GridResetSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_PrepareIndirectSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_SurfelPositionSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_UpdateSurfelsSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_GridOffsetSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_SurfelBinningSemaphore));
	VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_SurfelShadeSemaphore));

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VK_CHECK(vkCreateFence(*device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));

		VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
			vkDestroyFence(*device, _frames[i]._renderFence, nullptr);
			});

		// We do not need any flags for the sempahores

		VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_frames[i]._presentSemaphore));
		VK_CHECK(vkCreateSemaphore(*device, &semaphoreCreateInfo, nullptr, &_frames[i]._renderSemaphore));

		VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
			vkDestroySemaphore(*device, _frames[i]._presentSemaphore, nullptr);
			vkDestroySemaphore(*device, _frames[i]._renderSemaphore, nullptr);
			});
	}

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroySemaphore(*device, _offscreenSemaphore, nullptr);
		vkDestroySemaphore(*device, _rtSemaphore, nullptr);
		vkDestroySemaphore(*device, _shadowSemaphore, nullptr);
		vkDestroySemaphore(*device, _denoiseSemaphore, nullptr);
		vkDestroySemaphore(*device, _GridResetSemaphore, nullptr);
		vkDestroySemaphore(*device, _PrepareIndirectSemaphore, nullptr);
		vkDestroySemaphore(*device, _SurfelPositionSemaphore, nullptr);
		vkDestroySemaphore(*device, _UpdateSurfelsSemaphore, nullptr);
		vkDestroySemaphore(*device, _GridOffsetSemaphore, nullptr);
		vkDestroySemaphore(*device, _SurfelBinningSemaphore, nullptr);
		vkDestroySemaphore(*device, _SurfelShadeSemaphore, nullptr);
		});
}

void Renderer::init_descriptors()
{
	std::vector<VkDescriptorPoolSize> sizes = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100}
	};

	VkDescriptorPoolCreateInfo pool_info = vkinit::descriptor_pool_create_info(sizes, 10, VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	vkCreateDescriptorPool(*device, &pool_info, nullptr, &_descriptorPool);

	uint32_t nText = (uint32_t)Texture::_textures.size();
	VkDescriptorSetLayoutBinding cameraBind		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0);
	VkDescriptorSetLayoutBinding textureBind	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1, nText);
	VkDescriptorSetLayoutBinding materialBind	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);

	// Create descriptors set layouts
	// Set = 0
	// binding camera data at 0
	std::vector<VkDescriptorSetLayoutBinding> bindings = { cameraBind, textureBind };
	VkDescriptorSetLayoutCreateInfo setInfo = vkinit::descriptor_set_layout_create_info(bindings.size(), bindings);

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_offscreenDescriptorSetLayout));

	// Set = 1
	// binding nText textures at 0
	VkDescriptorSetLayoutCreateInfo set1Info = vkinit::descriptor_set_layout_create_info();
	set1Info.bindingCount	= 1;
	set1Info.pBindings		= &textureBind;

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &set1Info, nullptr, &_textureDescriptorSetLayout));

	// Set = 2
	// binding the material info of each entity
	VkDescriptorSetLayoutCreateInfo set2Info = vkinit::descriptor_set_layout_create_info();
	set2Info.bindingCount	= 1;
	set2Info.pBindings		= &materialBind;

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &set2Info, nullptr, &_objectDescriptorSetLayout));

	// Allocate descriptor sets
	// Camera descriptor set
	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptor_set_allocate_info(_descriptorPool, &_offscreenDescriptorSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &allocInfo, &_offscreenDescriptorSet));

	// Textures descriptor set
	VkDescriptorSetAllocateInfo textureAllocInfo = vkinit::descriptor_set_allocate_info(_descriptorPool, &_textureDescriptorSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &textureAllocInfo, &_textureDescriptorSet));

	// Material descriptor set
	VkDescriptorSetAllocateInfo materialAllocInfo = vkinit::descriptor_set_allocate_info(_descriptorPool, &_objectDescriptorSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &materialAllocInfo, &_objectDescriptorSet));

	// Create descriptors infos to write
	// Camera descriptor buffer
	VkDescriptorBufferInfo cameraInfo = vkinit::descriptor_buffer_info(_cameraBuffer._buffer, sizeof(GPUCameraData), 0);

	// Textures descriptor image infos
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSampler sampler;
	vkCreateSampler(*device, &samplerInfo, nullptr, &sampler);

	std::vector<VkDescriptorImageInfo> imageInfos;
	for (auto const& texture : Texture::_textures)
	{
		VkDescriptorImageInfo imageBufferInfo = {};
		imageBufferInfo.sampler		= sampler;
		imageBufferInfo.imageView	= texture.second->imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		imageInfos.push_back(imageBufferInfo);
	}

	// Material descriptor infos
	VkDescriptorBufferInfo materialInfo = vkinit::descriptor_buffer_info(VulkanEngine::engine->_objectBuffer._buffer, sizeof(GPUMaterial), 0);

	// Writes
	VkWriteDescriptorSet cameraWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _offscreenDescriptorSet, &cameraInfo, 0);
	VkWriteDescriptorSet texturesWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _offscreenDescriptorSet, imageInfos.data(), 1, nText);
	VkWriteDescriptorSet materialWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _objectDescriptorSet, &materialInfo, 0);

	std::vector<VkWriteDescriptorSet> writes = { cameraWrite, texturesWrite, materialWrite };

	vkUpdateDescriptorSets(*device, writes.size(), writes.data(), 0, nullptr);

	// SKYBOX DESCRIPTOR --------------------
	// Skybox set = 0
	// binding single texture as skybox and matrix to position the sphere around camera
	VkDescriptorSetLayoutBinding skyBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1);
	VkDescriptorSetLayoutBinding skyBufferBind = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 2);

	std::vector<VkDescriptorSetLayoutBinding> skyboxBindings = {
		cameraBind,		// binding = 0 camera info
		skyBind,		// binding = 1 sky texture
		skyBufferBind	// binding = 2 sphere matrix
	};

	VkDescriptorSetLayoutCreateInfo skyboxSetInfo = {};
	skyboxSetInfo.sType					= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	skyboxSetInfo.pNext					= nullptr;
	skyboxSetInfo.bindingCount			= static_cast<uint32_t>(skyboxBindings.size());
	skyboxSetInfo.pBindings				= skyboxBindings.data();

	vkCreateDescriptorSetLayout(*device, &skyboxSetInfo, nullptr, &_skyboxDescriptorSetLayout);
	
	VkDescriptorSetAllocateInfo skyboxAllocInfo = {};
	skyboxAllocInfo.sType				= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	skyboxAllocInfo.pNext				= nullptr;
	skyboxAllocInfo.descriptorPool		= _descriptorPool;
	skyboxAllocInfo.descriptorSetCount	= 1;
	skyboxAllocInfo.pSetLayouts			= &_skyboxDescriptorSetLayout;

	VK_CHECK(vkAllocateDescriptorSets(*device, &skyboxAllocInfo, &_skyboxDescriptorSet));

	VkDescriptorImageInfo skyboxImageInfo = {};
	skyboxImageInfo.sampler				= sampler;
	skyboxImageInfo.imageView = Texture::GET("data/textures/LA_Downtown_Helipad_GoldenHour_8k.jpg")->imageView; // Texture::GET("data/textures/woods.jpg")->imageView;
	skyboxImageInfo.imageLayout			= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VulkanEngine::engine->create_buffer(sizeof(glm::mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _skyboxBuffer);

	VkDescriptorBufferInfo skyboxBufferInfo = {};
	skyboxBufferInfo.buffer				= _skyboxBuffer._buffer;
	skyboxBufferInfo.offset				= 0;
	skyboxBufferInfo.range				= sizeof(glm::mat4);

	VkWriteDescriptorSet camWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _skyboxDescriptorSet, &cameraInfo, 0);
	VkWriteDescriptorSet skyboxWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _skyboxDescriptorSet, &skyboxImageInfo, 1);
	VkWriteDescriptorSet skyboxBuffer	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _skyboxDescriptorSet, &skyboxBufferInfo, 2);

	std::vector<VkWriteDescriptorSet> skyboxWrites = {
		camWrite,
		skyboxWrite,
		skyboxBuffer
	};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(skyboxWrites.size()), skyboxWrites.data(), 0, nullptr);
	
	// Destroy all objects created
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorPool(*device, _descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(*device, _offscreenDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _textureDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _objectDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _skyboxDescriptorSetLayout, nullptr);
		vkDestroySampler(*device, sampler, nullptr);
		});
}

void Renderer::init_deferred_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10}
	};

	VkDescriptorSetLayoutBinding positionBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0);	// Position
	VkDescriptorSetLayoutBinding normalBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1);	// Normals
	VkDescriptorSetLayoutBinding albedoBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 2);	// Albedo
	VkDescriptorSetLayoutBinding motionBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 3);	// Motion
	VkDescriptorSetLayoutBinding lightBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 4);	// Lights buffer
	VkDescriptorSetLayoutBinding debugBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 5);	// Debug display
	VkDescriptorSetLayoutBinding materialBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 6); // Metallic Roughness
	VkDescriptorSetLayoutBinding cameraBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 7); // Camera position buffer
	VkDescriptorSetLayoutBinding emissiveBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 8); // Emissive
	VkDescriptorSetLayoutBinding environtmentBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 9);
	VkDescriptorSetLayoutBinding surfelDebugBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 10);
	VkDescriptorSetLayoutBinding surfelResultBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 11);

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		positionBinding,
		normalBinding,
		albedoBinding,
		motionBinding,
		lightBinding,
		debugBinding,
		materialBinding,
		cameraBinding,
		emissiveBinding,
		environtmentBinding,
		surfelDebugBinding,
		surfelResultBinding
	};

	VkDescriptorSetLayoutCreateInfo setInfo = {};
	setInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.pNext			= nullptr;
	setInfo.bindingCount	= static_cast<uint32_t>(setLayoutBindings.size());
	setInfo.pBindings		= setLayoutBindings.data();

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_deferredSetLayout));

	const int nLights = _scene->_lights.size();

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSampler sampler;
	vkCreateSampler(*device, &samplerInfo, nullptr, &sampler);

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType					= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.pNext					= nullptr;
		allocInfo.descriptorPool		= _descriptorPool;
		allocInfo.descriptorSetCount	= 1;
		allocInfo.pSetLayouts			= &_deferredSetLayout;

		vkAllocateDescriptorSets(*device, &allocInfo, &_frames[i].deferredDescriptorSet);

		// Binginds 0 to 3 G-Buffers
		VkDescriptorImageInfo texDescriptorPosition = vkinit::descriptor_image_info(
			_deferredTextures[0].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Position
		VkDescriptorImageInfo texDescriptorNormal = vkinit::descriptor_image_info(
			_deferredTextures[1].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Normal
		VkDescriptorImageInfo texDescriptorAlbedo = vkinit::descriptor_image_info(
			_deferredTextures[2].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Albedo
		VkDescriptorImageInfo texDescriptorMotion = vkinit::descriptor_image_info(
			_deferredTextures[3].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Motion
		VkDescriptorImageInfo texDescriptorMaterial = vkinit::descriptor_image_info(
			_deferredTextures[4].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Material
		VkDescriptorImageInfo texDescriptorEmissive = vkinit::descriptor_image_info(
			_deferredTextures[5].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Material

		// Binding = 4 Light buffer
		VkDescriptorBufferInfo lightBufferDesc;
		lightBufferDesc.buffer	= _lightBuffer._buffer;
		lightBufferDesc.offset	= 0;
		lightBufferDesc.range	= sizeof(uboLight) * nLights;

		// Binding = 5 Debug value buffer
		VkDescriptorBufferInfo debugDesc;
		debugDesc.buffer		= _debugBuffer._buffer;
		debugDesc.offset		= 0;
		debugDesc.range			= sizeof(uint32_t);

		// Binding = 7 Camera buffer
		VkDescriptorBufferInfo cameraDesc;
		cameraDesc.buffer		= _cameraPositionBuffer._buffer;
		cameraDesc.offset		= 0;
		cameraDesc.range		= sizeof(glm::vec3);

		// Binding = 9 Environment image
		VkDescriptorImageInfo environmentDesc = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_Env.hdr")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		//B 10 debug gi
		VkDescriptorImageInfo debugGIdesc = vkinit::descriptor_image_info(
			_debugGI.imageView, VK_IMAGE_LAYOUT_GENERAL, _offscreenSampler);	

		//B 11 debug gi
		VkDescriptorImageInfo resultGIdesc = vkinit::descriptor_image_info(
			_result.imageView, VK_IMAGE_LAYOUT_GENERAL, _offscreenSampler);


		VkWriteDescriptorSet positionWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorPosition, 0);
		VkWriteDescriptorSet normalWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorNormal, 1);
		VkWriteDescriptorSet albedoWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorAlbedo, 2);
		VkWriteDescriptorSet motionWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorMotion, 3);
		VkWriteDescriptorSet lightBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].deferredDescriptorSet, &lightBufferDesc, 4);
		VkWriteDescriptorSet debugWrite			= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].deferredDescriptorSet, &debugDesc, 5);
		VkWriteDescriptorSet materialWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorMaterial, 6);
		VkWriteDescriptorSet cameraWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].deferredDescriptorSet, &cameraDesc, 7);
		VkWriteDescriptorSet emissiveWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorEmissive, 8);
		VkWriteDescriptorSet environmentWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &environmentDesc, 9);
		VkWriteDescriptorSet debugGIWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &debugGIdesc, 10);
		VkWriteDescriptorSet resultGIWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &resultGIdesc, 11);

		std::vector<VkWriteDescriptorSet> writes = {
			positionWrite,
			normalWrite,
			albedoWrite,
			motionWrite,
			lightBufferWrite,
			debugWrite,
			materialWrite,
			cameraWrite,
			emissiveWrite,
			environmentWrite,
			debugGIWrite,
			resultGIWrite
		};

		vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _deferredSetLayout, nullptr);
		vkDestroySampler(*device, sampler, nullptr);
		});
}

void Renderer::init_deferred_pipelines()
{
	VulkanEngine* engine = VulkanEngine::engine;

	VkShaderModule offscreenVertexShader;
	if (!engine->load_shader_module(vkutil::findFile("basic.vert.spv", searchPaths, true).c_str(), &offscreenVertexShader)) {
		std::cout << "Could not load geometry vertex shader!" << std::endl;
	}
	VkShaderModule offscreenFragmentShader;
	if (!engine->load_shader_module(vkutil::findFile("geometry_shader.frag.spv", searchPaths, true).c_str(), &offscreenFragmentShader)) {
		std::cout << "Could not load geometry fragment shader!" << std::endl;
	}
	VkShaderModule deferredVertexShader;
	if (!engine->load_shader_module(vkutil::findFile("quad.vert.spv", searchPaths, true).c_str(), &deferredVertexShader)) {
		std::cout << "Could not load deferred vertex shader!" << std::endl;
	}
	VkShaderModule deferredFragmentShader;
	if (!engine->load_shader_module(vkutil::findFile("deferred.frag.spv", searchPaths, true).c_str(), &deferredFragmentShader)) {
		std::cout << "Could not load deferred fragment shader!" << std::endl;
	}
	VkShaderModule skyboxVertexShader;
	if (!engine->load_shader_module(vkutil::findFile("skybox.vert.spv", searchPaths, true).c_str(), &skyboxVertexShader)) {
		std::cout << "Could not load skybox vertex shader!" << std::endl;
	}
	VkShaderModule skyboxFragmentShader;
	if (!engine->load_shader_module(vkutil::findFile("/skybox.frag.spv", searchPaths, true).c_str(), &skyboxFragmentShader)) {
		std::cout << "Could not load skybox fragment shader!" << std::endl;
	}

	PipelineBuilder pipBuilder;
	pipBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, offscreenVertexShader));
	pipBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, offscreenFragmentShader));

	VkDescriptorSetLayout offscreenSetLayouts[] = { _offscreenDescriptorSetLayout, _objectDescriptorSetLayout };

	VkPushConstantRange matrix_constant;
	matrix_constant.offset								= 0;
	matrix_constant.size								= sizeof(glm::mat4) * 2;
	matrix_constant.stageFlags							= VK_SHADER_STAGE_VERTEX_BIT;

	VkPushConstantRange material_constant;
	material_constant.offset							= sizeof(glm::mat4) * 2;
	material_constant.size								= sizeof(GPUMaterial);
	material_constant.stageFlags						= VK_SHADER_STAGE_FRAGMENT_BIT;

	VkPushConstantRange constants[] = { matrix_constant, material_constant };

	VkPipelineLayoutCreateInfo offscreenPipelineLayoutInfo = vkinit::pipeline_layout_create_info();
	offscreenPipelineLayoutInfo.setLayoutCount			= 2;
	offscreenPipelineLayoutInfo.pSetLayouts				= offscreenSetLayouts;
	offscreenPipelineLayoutInfo.pushConstantRangeCount	= 2;
	offscreenPipelineLayoutInfo.pPushConstantRanges		= constants;

	VK_CHECK(vkCreatePipelineLayout(*device, &offscreenPipelineLayoutInfo, nullptr, &_offscreenPipelineLayout));

	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	pipBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	pipBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	pipBuilder._vertexInputInfo.pVertexAttributeDescriptions	= vertexDescription.attributes.data();
	pipBuilder._vertexInputInfo.vertexBindingDescriptionCount	= vertexDescription.bindings.size();
	pipBuilder._vertexInputInfo.pVertexBindingDescriptions		= vertexDescription.bindings.data();

	pipBuilder._pipelineLayout = _offscreenPipelineLayout;

	VkExtent2D extent = { VulkanEngine::engine->_window->getWidth(), VulkanEngine::engine->_window->getHeight() };

	pipBuilder._inputAssembly		= vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipBuilder._rasterizer			= vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	pipBuilder._depthStencil		= vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
	pipBuilder._viewport.x			= 0.0f;
	pipBuilder._viewport.y			= 0.0f;
	pipBuilder._viewport.maxDepth	= 1.0f;
	pipBuilder._viewport.minDepth	= 0.0f;
	pipBuilder._viewport.width		= (float)VulkanEngine::engine->_window->getWidth();
	pipBuilder._viewport.height		= (float)VulkanEngine::engine->_window->getHeight();
	pipBuilder._scissor.offset		= { 0, 0 };
	pipBuilder._scissor.extent		= extent;

	std::array<VkPipelineColorBlendAttachmentState, 6> blendAttachmentStates = {
		vkinit::color_blend_attachment_state(
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, VK_FALSE),
		vkinit::color_blend_attachment_state(0xf, VK_FALSE),
		vkinit::color_blend_attachment_state(0xf, VK_FALSE),
		vkinit::color_blend_attachment_state(0xf, VK_FALSE),
		vkinit::color_blend_attachment_state(0xf, VK_FALSE),
		vkinit::color_blend_attachment_state(0xf, VK_FALSE)
	};

	VkPipelineColorBlendStateCreateInfo colorBlendInfo = vkinit::color_blend_state_create_info(static_cast<uint32_t>(blendAttachmentStates.size()), blendAttachmentStates.data());

	pipBuilder._colorBlendStateInfo = colorBlendInfo;
	pipBuilder._multisampling		= vkinit::multisample_state_create_info();

	_offscreenPipeline = pipBuilder.build_pipeline(*device, _offscreenRenderPass);

	// Skybox pipeline -----------------------------------------------------------------------------

	VkPipelineLayoutCreateInfo skyboxPipelineLayoutInfo = vkinit::pipeline_layout_create_info();
	skyboxPipelineLayoutInfo.setLayoutCount = 1;
	skyboxPipelineLayoutInfo.pSetLayouts	= &_skyboxDescriptorSetLayout;

	VK_CHECK(vkCreatePipelineLayout(*device, &skyboxPipelineLayoutInfo, nullptr, &_skyboxPipelineLayout));

	pipBuilder._shaderStages.clear();
	pipBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, skyboxVertexShader));
	pipBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, skyboxFragmentShader));

	pipBuilder._pipelineLayout					= _skyboxPipelineLayout;
	pipBuilder._depthStencil.depthTestEnable	= VK_FALSE;
	pipBuilder._depthStencil.depthWriteEnable	= VK_TRUE;

	_skyboxPipeline = pipBuilder.build_pipeline(*device, _offscreenRenderPass);

	// Second pipeline -----------------------------------------------------------------------------

	VkPushConstantRange push_constant_final;
	push_constant_final.offset		= 0;
	push_constant_final.size		= sizeof(pushConstants);
	push_constant_final.stageFlags	= VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayout finalSetLayout[] = { _deferredSetLayout };

	VkPipelineLayoutCreateInfo deferredPipelineLayoutInfo = vkinit::pipeline_layout_create_info();
	deferredPipelineLayoutInfo.setLayoutCount			= 1;
	deferredPipelineLayoutInfo.pSetLayouts				= finalSetLayout;
	deferredPipelineLayoutInfo.pushConstantRangeCount	= 1;
	deferredPipelineLayoutInfo.pPushConstantRanges		= &push_constant_final;

	VK_CHECK(vkCreatePipelineLayout(*device, &deferredPipelineLayoutInfo, nullptr, &_finalPipelineLayout));

	VkPipelineColorBlendAttachmentState att = vkinit::color_blend_attachment_state(0xf, VK_FALSE);

	pipBuilder._colorBlendStateInfo.sType			= VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	pipBuilder._colorBlendStateInfo.attachmentCount = 1;
	pipBuilder._colorBlendStateInfo.pAttachments	= &att;

	pipBuilder._shaderStages.clear();
	pipBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, deferredVertexShader));
	pipBuilder._shaderStages.push_back(
		vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, deferredFragmentShader));
	pipBuilder._depthStencil.depthTestEnable = VK_TRUE;
	pipBuilder._depthStencil.depthWriteEnable = VK_TRUE;
	pipBuilder._pipelineLayout = _finalPipelineLayout;

	_finalPipeline = pipBuilder.build_pipeline(*device, _renderPass);

	vkDestroyShaderModule(*device, offscreenVertexShader, nullptr);
	vkDestroyShaderModule(*device, offscreenFragmentShader, nullptr);
	vkDestroyShaderModule(*device, skyboxVertexShader, nullptr);
	vkDestroyShaderModule(*device, skyboxFragmentShader, nullptr);
	vkDestroyShaderModule(*device, deferredVertexShader, nullptr);
	vkDestroyShaderModule(*device, deferredFragmentShader, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipelineLayout(*device, _offscreenPipelineLayout, nullptr);
		vkDestroyPipelineLayout(*device, _skyboxPipelineLayout, nullptr);
		vkDestroyPipelineLayout(*device, _finalPipelineLayout, nullptr);
		vkDestroyPipeline(*device, _offscreenPipeline, nullptr);
		vkDestroyPipeline(*device, _skyboxPipeline, nullptr);
		vkDestroyPipeline(*device, _finalPipeline, nullptr);
		});
}

void Renderer::build_forward_command_buffer()
{
	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));
	VkCommandBuffer *cmd = &get_current_frame()._mainCommandBuffer;

	std::array<VkClearValue, 2> clearValues;
	clearValues[0].color		= { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType						= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass					= _forwardRenderPass;
	renderPassBeginInfo.renderArea.extent.width		= VulkanEngine::engine->_window->getWidth();
	renderPassBeginInfo.renderArea.extent.height	= VulkanEngine::engine->_window->getHeight();
	renderPassBeginInfo.clearValueCount				= static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues				= clearValues.data();
	renderPassBeginInfo.framebuffer					= _framebuffers[VulkanEngine::engine->_indexSwapchainImage];


	std::cout << "hey" << std::endl;


	VK_CHECK(vkBeginCommandBuffer(*cmd, &cmdBufInfo));

	vkCmdBeginRenderPass(*cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
	// Set = 0 Camera data descriptor
	uint32_t uniform_offset = VulkanEngine::engine->pad_uniform_buffer_size(sizeof(GPUSceneData));
	vkCmdBindDescriptorSets(*cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _forwardPipelineLayout, 0, 1, &_offscreenDescriptorSet, 1, &uniform_offset);
	// Set = 1 Object data descriptor
	vkCmdBindDescriptorSets(*cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _forwardPipelineLayout, 1, 1, &_objectDescriptorSet, 0, nullptr);
	// Set = 2 Texture data descriptor
	vkCmdBindDescriptorSets(*cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _forwardPipelineLayout, 2, 1, &_textureDescriptorSet, 0, nullptr);

	Mesh* lastMesh = nullptr;

	for (size_t i = 0; i < _scene->_entities.size(); i++)
	{
		Object* object = _scene->_entities[i];

		vkCmdBindPipeline(*cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _forwardPipeline);

		VkDeviceSize offset = { 0 };

		int constant = object->id;
		int matIdx = object->materialIdx;
		vkCmdPushConstants(_offscreenComandBuffer, _offscreenPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &constant);
		vkCmdPushConstants(_offscreenComandBuffer, _offscreenPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, sizeof(int), sizeof(int), &matIdx);

		if (lastMesh != object->prefab->_mesh) {
			vkCmdBindVertexBuffers(*cmd, 0, 1, &object->prefab->_mesh->_vertexBuffer._buffer, &offset);
			vkCmdBindIndexBuffer(*cmd, object->prefab->_mesh->_indexBuffer._buffer, 0, VK_INDEX_TYPE_UINT32);
			lastMesh = object->prefab->_mesh;
		}
		vkCmdDrawIndexed(*cmd, static_cast<uint32_t>(object->prefab->_mesh->_indices.size()), _scene->_entities.size(), 0, 0, i);
	}

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), *cmd);

	vkCmdEndRenderPass(*cmd);
	VK_CHECK(vkEndCommandBuffer(*cmd));
}

void Renderer::build_previous_command_buffer()
{
	if (_offscreenComandBuffer == VK_NULL_HANDLE)
	{
		VkCommandBufferAllocateInfo allocInfo = vkinit::command_buffer_allocate_info(_commandPool, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		VK_CHECK(vkAllocateCommandBuffers(*device, &allocInfo, &_offscreenComandBuffer));
	}

	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	vkDeviceWaitIdle(*device);
	VK_CHECK(vkBeginCommandBuffer(_offscreenComandBuffer, &cmdBufInfo));

	VkDeviceSize offset = { 0 };

	std::array<VkClearValue, 7> clearValues;
	clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[1].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[2].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[3].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[4].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[5].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[6].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType						= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass					= _offscreenRenderPass;
	renderPassBeginInfo.framebuffer					= _offscreenFramebuffer;
	renderPassBeginInfo.renderArea.extent.width		= VulkanEngine::engine->_window->getWidth();
	renderPassBeginInfo.renderArea.extent.height	= VulkanEngine::engine->_window->getHeight();
	renderPassBeginInfo.clearValueCount				= static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues				= clearValues.data();

	//std::cout << "pito" << std::endl;

	vkCmdBeginRenderPass(_offscreenComandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);


	//std::cout << "puto" << std::endl;

	// Skybox pass
	vkCmdBindDescriptorSets(_offscreenComandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _skyboxPipelineLayout, 0, 1, &_skyboxDescriptorSet, 0, nullptr);
	vkCmdBindPipeline(_offscreenComandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _skyboxPipeline);
	Mesh* sphere = Mesh::GET("sphere.obj");
	vkCmdBindVertexBuffers(_offscreenComandBuffer, 0, 1, &sphere->_vertexBuffer._buffer, &offset);
	vkCmdBindIndexBuffer(_offscreenComandBuffer, sphere->_indexBuffer._buffer, offset, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(_offscreenComandBuffer, static_cast<uint32_t>(sphere->_indices.size()), 1, 0, 0, 1);

	// Geometry pass
	// Set = 0 Camera data descriptor
	vkCmdBindDescriptorSets(_offscreenComandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _offscreenPipelineLayout, 0, 1, &_offscreenDescriptorSet, 0, nullptr);

	vkCmdBindPipeline(_offscreenComandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _offscreenPipeline);

	uint32_t instance = 0;
	for (size_t i = 0; i < _scene->_entities.size(); i++)
	{
		Object* object = _scene->_entities[i];
		object->draw(_offscreenComandBuffer, _offscreenPipelineLayout, object->m_matrix);
	}

	vkCmdEndRenderPass(_offscreenComandBuffer);
	VK_CHECK(vkEndCommandBuffer(_offscreenComandBuffer));
}

void Renderer::build_deferred_command_buffer()
{
	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	std::array<VkClearValue, 2> clearValues;
	clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType						= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass					= _renderPass;
	renderPassBeginInfo.renderArea.extent.width		= VulkanEngine::engine->_window->getWidth();
	renderPassBeginInfo.renderArea.extent.height	= VulkanEngine::engine->_window->getHeight();
	renderPassBeginInfo.clearValueCount				= static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues				= clearValues.data();
	renderPassBeginInfo.framebuffer					= _framebuffers[VulkanEngine::engine->_indexSwapchainImage];

	//std::cout << "hola" << std::endl;

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

	vkBeginCommandBuffer(get_current_frame()._mainCommandBuffer, &cmdBufInfo);

	vkCmdBeginRenderPass(get_current_frame()._mainCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(get_current_frame()._mainCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _finalPipeline);

	VkDeviceSize offset = { 0 };

	Mesh* quad = Mesh::get_quad();

	vkCmdPushConstants(get_current_frame()._mainCommandBuffer, _finalPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushConstants), &_constants);

	vkCmdBindDescriptorSets(get_current_frame()._mainCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _finalPipelineLayout, 0, 1, &get_current_frame().deferredDescriptorSet, 0, nullptr);
	vkCmdBindVertexBuffers(get_current_frame()._mainCommandBuffer, 0, 1, &quad->_vertexBuffer._buffer, &offset);
	vkCmdBindIndexBuffer(get_current_frame()._mainCommandBuffer, quad->_indexBuffer._buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(get_current_frame()._mainCommandBuffer, static_cast<uint32_t>(quad->_indices.size()), 1, 0, 0, 1);

	//ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), get_current_frame()._mainCommandBuffer);

	vkCmdEndRenderPass(get_current_frame()._mainCommandBuffer);
	VK_CHECK(vkEndCommandBuffer(get_current_frame()._mainCommandBuffer));
}

void Renderer::load_data_to_gpu()
{
	// Raster data
	if(!_cameraBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _cameraBuffer);
	if(!VulkanEngine::engine->_objectBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(GPUMaterial), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, VulkanEngine::engine->_objectBuffer);
	if (!_debugBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(uint32_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _debugBuffer);
	if (!_cameraPositionBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(glm::vec3), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _cameraPositionBuffer);
	if (!_frameCountBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(sizeof(int)), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _frameCountBuffer);

	// Raytracing data
	const unsigned int nLights		= _scene->_lights.size();
	const unsigned int nMaterials	= Material::_materials.size();

	if (!_lightBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(uboLight) * nLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _lightBuffer);
	if (!_matBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(GPUMaterial) * nMaterials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _matBuffer);
	if (!_rtCameraBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(RTCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _rtCameraBuffer);

	// TODO: rethink how to update vertex and index for each entity
	for (Object* obj : _scene->_entities)
	{
		for (Node* root : obj->prefab->_root)
		{
			root->fill_matrix_buffer(_scene->_matricesVector, obj->m_matrix);
		}
	}

	if(!_matricesBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(glm::mat4) * _scene->_matricesVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _matricesBuffer);

	void* matricesData;
	vmaMapMemory(VulkanEngine::engine->_allocator, _matricesBuffer._allocation, &matricesData);
	memcpy(matricesData, _scene->_matricesVector.data(), sizeof(glm::mat4) * _scene->_matricesVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _matricesBuffer._allocation);


	// Update material data
	std::vector<GPUMaterial> materials;
	for (Material* it : Material::_materials)
	{
		GPUMaterial mat = it->materialToShader();
		materials.push_back(mat);
	}

	void* matData;
	vmaMapMemory(VulkanEngine::engine->_allocator, _matBuffer._allocation, &matData);
	memcpy(matData, materials.data(), sizeof(GPUMaterial) * materials.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _matBuffer._allocation);

}

void Renderer::create_storage_image()
{
	VkExtent3D extent			= { VulkanEngine::engine->_window->getWidth(), VulkanEngine::engine->_window->getHeight(), 1 };
	VkImageCreateInfo imageInfo = vkinit::image_create_info(VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent);
	imageInfo.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;

	VkImageCreateInfo shadowImageInfo	= vkinit::image_create_info(VK_FORMAT_R8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent);
	shadowImageInfo.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;

	VkImageCreateInfo debugImageInfo = vkinit::image_create_info(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent);
	debugImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;


	VkImageCreateInfo resutlGIinfo = vkinit::image_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent);
	resutlGIinfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage			= VMA_MEMORY_USAGE_GPU_ONLY;
	allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(VulkanEngine::engine->_allocator, &imageInfo, &allocInfo,
		&_rtImage.image._image, &_rtImage.image._allocation, nullptr);

	VkImageViewCreateInfo imageViewInfo = vkinit::image_view_create_info(VK_FORMAT_B8G8R8A8_UNORM, _rtImage.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(*device, &imageViewInfo, nullptr, &_rtImage.imageView));

	vmaCreateImage(VulkanEngine::engine->_allocator, &debugImageInfo, &allocInfo,
		&_debugGI.image._image, &_debugGI.image._allocation, nullptr);

	VkImageViewCreateInfo dubugImageViewInfo = vkinit::image_view_create_info(VK_FORMAT_R8G8B8A8_UNORM, _debugGI.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(*device, &dubugImageViewInfo, nullptr, &_debugGI.imageView));


	vmaCreateImage(VulkanEngine::engine->_allocator, &resutlGIinfo, &allocInfo,
		&_result.image._image, &_result.image._allocation, nullptr);

	VkImageViewCreateInfo resultImageViewInfo = vkinit::image_view_create_info(VK_FORMAT_R16G16B16A16_SFLOAT, _result.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(*device, &resultImageViewInfo, nullptr, &_result.imageView));


	_shadowImages.reserve(_scene->_lights.size());

	for (decltype(_scene->_lights.size()) i = 0; i < _scene->_lights.size(); i++)
	{
		Texture image, denoisedImage;
		vmaCreateImage(VulkanEngine::engine->_allocator, &shadowImageInfo, &allocInfo,
			&image.image._image, &image.image._allocation, nullptr);
		VkImageViewCreateInfo shadowImageViewInfo = vkinit::image_view_create_info(VK_FORMAT_R8_UNORM, image.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
		VK_CHECK(vkCreateImageView(*device, &shadowImageViewInfo, nullptr, &image.imageView));
		_shadowImages.emplace_back(image);
	
		vmaCreateImage(VulkanEngine::engine->_allocator, &shadowImageInfo, &allocInfo,
			&denoisedImage.image._image, &denoisedImage.image._allocation, nullptr);
		VkImageViewCreateInfo denoiseImageViewInfo = vkinit::image_view_create_info(VK_FORMAT_R8_UNORM, denoisedImage.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
		VK_CHECK(vkCreateImageView(*device, &denoiseImageViewInfo, nullptr, &denoisedImage.imageView));
		_denoisedImages.emplace_back(denoisedImage);
	}

	VulkanEngine::engine->immediate_submit([&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier{};
		imageMemoryBarrier.sType					= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.image					= _rtImage.image._image;
		imageMemoryBarrier.oldLayout				= VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout				= VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.subresourceRange			= { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		VkImageMemoryBarrier barrier[] = { imageMemoryBarrier};

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, barrier);
	});

	VulkanEngine::engine->immediate_submit([&](VkCommandBuffer cmd) {
		std::vector<VkImageMemoryBarrier> shadowBarriers(_shadowImages.size());
		for (int i = 0; i < _shadowImages.size(); i++)
		{
			VkImageMemoryBarrier shadowImageMemoryBarrier{};
			shadowImageMemoryBarrier.sType				= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			shadowImageMemoryBarrier.image				= _shadowImages[i].image._image;
			shadowImageMemoryBarrier.oldLayout			= VK_IMAGE_LAYOUT_UNDEFINED;
			shadowImageMemoryBarrier.newLayout			= VK_IMAGE_LAYOUT_GENERAL;
			shadowImageMemoryBarrier.subresourceRange	= { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			shadowBarriers[i] = shadowImageMemoryBarrier;
		}
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, _shadowImages.size(), shadowBarriers.data());
	});

	VulkanEngine::engine->immediate_submit([&](VkCommandBuffer cmd) {
		std::vector<VkImageMemoryBarrier> denoisedBarriers(_shadowImages.size());
		for (int i = 0; i < _shadowImages.size(); i++)
		{
			VkImageMemoryBarrier denoiseImageMemoryBarrier{};
			denoiseImageMemoryBarrier.sType				= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			denoiseImageMemoryBarrier.image				= _denoisedImages[i].image._image;
			denoiseImageMemoryBarrier.oldLayout			= VK_IMAGE_LAYOUT_UNDEFINED;
			denoiseImageMemoryBarrier.newLayout			= VK_IMAGE_LAYOUT_GENERAL;
			denoiseImageMemoryBarrier.subresourceRange	= { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			denoisedBarriers[i] = denoiseImageMemoryBarrier;
		}
		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, _shadowImages.size(), denoisedBarriers.data());
	});

	VulkanEngine::engine->immediate_submit([&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier{};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.image = _debugGI.image._image;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		VkImageMemoryBarrier barrier[] = { imageMemoryBarrier };

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, barrier);
		});


	VulkanEngine::engine->immediate_submit([&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier{};
		imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.image = _result.image._image;
		imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		VkImageMemoryBarrier barrier[] = { imageMemoryBarrier };

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, barrier);
		});

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vmaDestroyImage(VulkanEngine::engine->_allocator, _rtImage.image._image, _rtImage.image._allocation);
		vkDestroyImageView(*device, _rtImage.imageView, nullptr);

		vmaDestroyImage(VulkanEngine::engine->_allocator, _debugGI.image._image, _debugGI.image._allocation);
		vkDestroyImageView(*device, _debugGI.imageView, nullptr);

		vmaDestroyImage(VulkanEngine::engine->_allocator, _result.image._image, _result.image._allocation);
		vkDestroyImageView(*device, _result.imageView, nullptr);

		for (int i = 0; i < _shadowImages.size(); i++)
		{
			vmaDestroyImage(VulkanEngine::engine->_allocator, _shadowImages[i].image._image, _shadowImages[i].image._allocation);
			vmaDestroyImage(VulkanEngine::engine->_allocator, _denoisedImages[i].image._image, _denoisedImages[i].image._allocation);
			vkDestroyImageView(*device, _shadowImages[i].imageView, nullptr);
			vkDestroyImageView(*device, _denoisedImages[i].imageView, nullptr);

		}
	});
}

void Renderer::recreate_renderer()
{
	init_render_pass();
	init_forward_render_pass();
	init_offscreen_render_pass();
	init_deferred_pipelines();
	init_raytracing_pipeline();
	init_framebuffers();
	init_offscreen_framebuffers();
}

// VKRAY
// ---------------------------------------------------------------------------------------
// Create all the BLAS
// - Go through all meshes in the scene and convert them to BlasInput (holds geometry and rangeInfo)
// - Build as many BLAS as BlasInput (geometries defined in the scene)

void Renderer::create_bottom_acceleration_structure()
{
	std::vector<BlasInput> allBlas;
	allBlas.reserve(_scene->get_drawable_nodes_size());
	for (Object* obj : _scene->_entities)
	{
		Prefab* p = obj->prefab;
		if (!p->_root.empty())
		{
			VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
			VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

			VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
			bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
			bufferDeviceAddressInfo.buffer = obj->prefab->_mesh->_vertexBuffer._buffer;


			vertexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);

			bufferDeviceAddressInfo.buffer = obj->prefab->_mesh->_indexBuffer._buffer;

			indexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);

			for (Node* root : p->_root)
			{
				root->node_to_geometry(allBlas, vertexBufferDeviceAddress, indexBufferDeviceAddress);
			}
		}
	}

 	buildBlas(allBlas);
}

// ---------------------------------------------------------------------------------------
// Create all the TLAS
// - Go through all meshes in the scene and convert them to Instances (holds matrices)
// - Build as many Instances as BlasInput (geometries defined in the scene) and pass them to build the TLAS
void Renderer::create_top_acceleration_structure()
{
	int instanceIndex = 0;
	for (auto& entity : _scene->_entities)
	{
		Prefab* p = entity->prefab;
		if (!p->_root.empty())
		{
			for (Node* root : p->_root)
			{
				root->node_to_instance(_tlas, instanceIndex, entity->m_matrix);
			}
		}
	}

	buildTlas(_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

// ---------------------------------------------------------------------------------------
// This function will create as many BLAS as input objects.
// - Create a buildGeometryInfo for each input object and add the necessary information
// - Create the AS object where handle and device addres is stored
// - Find the max scratch size for all the inputs in order to only use one buffer
// - Finally submit the creation commands
void Renderer::buildBlas(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags)
{
	// Make own copy of the information coming from input
	assert(_blas.empty());	// Make sure that we are only building blas once
	_blas = std::vector<BlasInput>(input.begin(), input.end());
	uint32_t blasSize = static_cast<uint32_t>(_blas.size());

	_bottomLevelAS.resize(blasSize);	// Prepare all necessary BLAS to create

	// We will prepare the building information for each of the blas
	std::vector<VkAccelerationStructureBuildGeometryInfoKHR> asBuildGeoInfos(blasSize);
	for (uint32_t i = 0; i < blasSize; i++)
	{
		asBuildGeoInfos[i] = vkinit::acceleration_structure_build_geometry_info();
		asBuildGeoInfos[i].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		asBuildGeoInfos[i].flags = flags;
		asBuildGeoInfos[i].geometryCount = 1;
		asBuildGeoInfos[i].pGeometries = &_blas[i].asGeometry;
		asBuildGeoInfos[i].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		asBuildGeoInfos[i].srcAccelerationStructure = VK_NULL_HANDLE;
	}

	// Used to search for the maximum scratch size to only use one scratch for the whole build
	VkDeviceSize maxScratch{ 0 };

	for (uint32_t i = 0; i < blasSize; i++)
	{
		VkAccelerationStructureBuildSizesInfoKHR asBuildSizesInfo{};
		asBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		vkGetAccelerationStructureBuildSizesKHR(*device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &asBuildGeoInfos[i], &_blas[i].nTriangles, &asBuildSizesInfo);

		create_acceleration_structure(_bottomLevelAS[i], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, asBuildSizesInfo);

		asBuildGeoInfos[i].dstAccelerationStructure = _bottomLevelAS[i].handle;

		maxScratch = std::max(maxScratch, asBuildSizesInfo.buildScratchSize);
	}

	RayTracingScratchBuffer scratchBuffer{};
	VulkanEngine::engine->create_buffer(maxScratch, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY, scratchBuffer.buffer, false);
	
	VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
	bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAddressInfo.buffer = scratchBuffer.buffer._buffer;
	scratchBuffer.deviceAddress = VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);

	// Once the scratch buffer is created we finally end inflating the asBuildGeosInfo struct and can proceed to submit the creation command
	for (uint32_t i = 0; i < blasSize; i++) {

		asBuildGeoInfos[i].scratchData.deviceAddress = scratchBuffer.deviceAddress;

		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> asBuildStructureRangeInfo = { &_blas[i].asBuildRangeInfo };

		VulkanEngine::engine->immediate_submit([=](VkCommandBuffer cmd) {
			vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildGeoInfos[i], asBuildStructureRangeInfo.data());
			});

		//	VkCommandPoolCreateInfo poolInfo = vkinit::command_pool_create_info(VulkanEngine::engine->_graphicsQueueFamily);
//
		//VkCommandPoolCreateInfo poolInfo = vkinit::command_pool_create_info(VulkanEngine::engine->_graphicsQueueFamily);
		//VkCommandPool pool;
		//vkCreateCommandPool(VulkanEngine::engine->_device, &poolInfo, nullptr, &pool);
		//VkCommandBufferAllocateInfo alloc = vkinit::command_buffer_allocate_info(pool, 1);

		//VkCommandBuffer cmd = VulkanEngine::engine->beginSingleTimeCommands();

		//vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildGeoInfos[i], asBuildStructureRangeInfo.data());

		//VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
		//barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
		//barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
		//vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		//	VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr, 0, nullptr);

		//VulkanEngine::engine->endSingleTimeCommands(cmd);

		//VkSubmitInfo submit = vkinit::submit_info(&cmd);
		//vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, nullptr);

		//vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

		//vkDestroyCommandPool(VulkanEngine::engine->_device, pool, nullptr);
//

	}

	// Finally we can free the scratch buffer
	vmaDestroyBuffer(VulkanEngine::engine->_allocator, scratchBuffer.buffer._buffer, scratchBuffer.buffer._allocation);
}

// ---------------------------------------------------------------------------------------
// This creates the TLAS from the input instances
void Renderer::buildTlas(const std::vector<TlasInstance>& instances, VkBuildAccelerationStructureFlagsKHR flags, bool update)
{
	// Cannot be built twice
	assert(_topLevelAS.handle == VK_NULL_HANDLE || update);

	std::vector<VkAccelerationStructureInstanceKHR> geometryInstances;
	//geometryInstances.reserve(instances.size());

	for (const TlasInstance& instance : instances)
	{
		geometryInstances.push_back(object_to_instance(instance));
	}

	VkDeviceSize instancesSize = geometryInstances.size() * sizeof(VkAccelerationStructureInstanceKHR);

	// Create buffer if not already created 
	if (!update)
	{
		VulkanEngine::engine->create_buffer(instancesSize,
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			VMA_MEMORY_USAGE_CPU_TO_GPU, _instanceBuffer);
	}

	void* instanceData;
	vmaMapMemory(VulkanEngine::engine->_allocator, _instanceBuffer._allocation, &instanceData);
	memcpy(instanceData, geometryInstances.data(), instancesSize);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _instanceBuffer._allocation);

	VkDeviceOrHostAddressConstKHR instanceDataDeviceAddress{};
	//instanceDataDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(_instanceBuffer._buffer);



	VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfoInstances{};
	bufferDeviceAddressInfoInstances.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAddressInfoInstances.buffer = _instanceBuffer._buffer;
	instanceDataDeviceAddress.deviceAddress = VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfoInstances);

	//// Create a stucture that holds a device pointer to the uploaded instances.
	//VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
	//instancesData.sType					= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	//instancesData.arrayOfPointers		= VK_FALSE;
	//instancesData.data					= instanceDataDeviceAddress;

	//// Put the above structure to the structure geometry
	//VkAccelerationStructureGeometryKHR asGeometry = vkinit::acceleration_structure_geometry_khr();
	//asGeometry.geometryType				= VK_GEOMETRY_TYPE_INSTANCES_KHR;
	//asGeometry.flags					= VK_GEOMETRY_OPAQUE_BIT_KHR;
	//asGeometry.geometry.instances		= instancesData;


	VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
	accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	accelerationStructureGeometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	accelerationStructureGeometry.geometry.instances.arrayOfPointers = VK_FALSE;
	accelerationStructureGeometry.geometry.instances.data = instanceDataDeviceAddress;




	// Find sizes
	//VkAccelerationStructureBuildGeometryInfoKHR asBuildGeometryInfo = vkinit::acceleration_structure_build_geometry_info();
	//asBuildGeometryInfo.type			= VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	//asBuildGeometryInfo.mode			= update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	//asBuildGeometryInfo.flags			= flags;
	//asBuildGeometryInfo.geometryCount	= 1;
	//asBuildGeometryInfo.pGeometries		= &accelerationStructureGeometry;
	//asBuildGeometryInfo.srcAccelerationStructure = update ? _topLevelAS.handle : VK_NULL_HANDLE;
	//asBuildGeometryInfo.dstAccelerationStructure = _topLevelAS.handle;


	VkAccelerationStructureBuildGeometryInfoKHR accelerationStructureBuildGeometryInfo{};
	accelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	accelerationStructureBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationStructureBuildGeometryInfo.geometryCount = 1;
	accelerationStructureBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;


	uint32_t									primitiveCount = static_cast<uint32_t>(instances.size());
	VkAccelerationStructureBuildSizesInfoKHR accelerationStructureBuildSizesInfo{};
	accelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(
		VulkanEngine::engine->_device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&accelerationStructureBuildGeometryInfo,
		&primitiveCount,
		&accelerationStructureBuildSizesInfo
	);


	create_acceleration_structure(_topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, accelerationStructureBuildSizesInfo);

	
	RayTracingScratchBuffer scratchBuffer{};
	VulkanEngine::engine->create_buffer(accelerationStructureBuildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY, scratchBuffer.buffer, false);

	VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
	bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bufferDeviceAddressInfo.buffer = scratchBuffer.buffer._buffer;
	scratchBuffer.deviceAddress = VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);


	VkAccelerationStructureBuildGeometryInfoKHR accelerationBuildGeometryInfo{};
	accelerationBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	accelerationBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	accelerationBuildGeometryInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	accelerationBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	accelerationBuildGeometryInfo.dstAccelerationStructure = _topLevelAS.handle;
	accelerationBuildGeometryInfo.geometryCount = 1;
	accelerationBuildGeometryInfo.pGeometries = &accelerationStructureGeometry;
	accelerationBuildGeometryInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;



	VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};
	asBuildRangeInfo.primitiveCount		= static_cast<uint32_t>(instances.size());
	asBuildRangeInfo.primitiveOffset	= 0;
	asBuildRangeInfo.firstVertex		= 0;
	asBuildRangeInfo.transformOffset	= 0;

	std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &asBuildRangeInfo };


	//const VkAccelerationStructureBuildRangeInfoKHR* pAsBuildRangeInfo = &asBuildRangeInfo;
	
	VkCommandPoolCreateInfo poolInfo = vkinit::command_pool_create_info(VulkanEngine::engine->_graphicsQueueFamily);

	VkCommandPool pool;
	vkCreateCommandPool(VulkanEngine::engine->_device, &poolInfo, nullptr, &pool);
	VkCommandBufferAllocateInfo alloc = vkinit::command_buffer_allocate_info(pool, 1);

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(VulkanEngine::engine->_device, &alloc, &cmd);

	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	vkBeginCommandBuffer(cmd, &beginInfo);

	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &accelerationBuildGeometryInfo, accelerationBuildStructureRangeInfos.data());

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, nullptr);

	vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	vkDestroyCommandPool(VulkanEngine::engine->_device, pool, nullptr);
	
	vmaDestroyBuffer(VulkanEngine::engine->_allocator, scratchBuffer.buffer._buffer, scratchBuffer.buffer._allocation);
}
//
void Renderer::create_acceleration_structure(AccelerationStructure& accelerationStructure, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
{

	VulkanEngine::engine->create_buffer(buildSizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY, accelerationStructure.buffer, false);

	VkAccelerationStructureCreateInfoKHR asCreateInfo{};
	asCreateInfo.sType				= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	asCreateInfo.buffer				= accelerationStructure.buffer._buffer;
	asCreateInfo.size				= buildSizeInfo.accelerationStructureSize;
	asCreateInfo.type				= type;

	vkCreateAccelerationStructureKHR(*device,
		&asCreateInfo, nullptr, &accelerationStructure.handle);

	VkAccelerationStructureDeviceAddressInfoKHR asDeviceAddressInfo{};
	asDeviceAddressInfo.sType					= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	asDeviceAddressInfo.accelerationStructure	= accelerationStructure.handle;

	accelerationStructure.deviceAddress = vkGetAccelerationStructureDeviceAddressKHR(*device, &asDeviceAddressInfo);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vmaDestroyBuffer(VulkanEngine::engine->_allocator, accelerationStructure.buffer._buffer, accelerationStructure.buffer._allocation);
		vkDestroyAccelerationStructureKHR(VulkanEngine::engine->_device, accelerationStructure.handle, nullptr);
		});
}

// Pass the information from our instance to the vk instance to function in the TLAS
VkAccelerationStructureInstanceKHR Renderer::object_to_instance(const TlasInstance& instance)
{
	assert(size_t(instance.blasId) < _blas.size());

	glm::mat4 aux = glm::transpose(instance.transform);

	VkTransformMatrixKHR transform = {
		aux[0].x, aux[0].y, aux[0].z, aux[0].w,
		aux[1].x, aux[1].y, aux[1].z, aux[1].w,
		aux[2].x, aux[2].y, aux[2].z, aux[2].w,
	};

	VkAccelerationStructureInstanceKHR vkInst{};
	vkInst.transform								= transform;
	vkInst.instanceCustomIndex						= instance.instanceId;
	vkInst.mask										= instance.mask;
	vkInst.instanceShaderBindingTableRecordOffset	= instance.hitGroupId;
	vkInst.flags									= instance.flags;
	vkInst.accelerationStructureReference			= _bottomLevelAS[instance.blasId].deviceAddress;

	return vkInst;
}
//
// TODO: Erase if not necessary
void Renderer::create_shadow_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_shadowDescPool));

	// First set
	// binding 0 = AS
	// binding 1 = Storage image
	// binding 2 = Camera data
	// binding 3 = Lights buffer
	// binding 4 = Samples buffer
	// binding 5 = Position, Normal, Material, Motion Gbuffer
	// binding 6 = Material buffer
	//const unsigned int nInstances	= _scene->_entities.size();
	//const unsigned int nLights		= _scene->_lights.size();

	//VkDescriptorSetLayoutBinding accelerationStructureLayoutBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);
	//VkDescriptorSetLayoutBinding storageImageLayoutBinding			= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 1, nLights);
	//VkDescriptorSetLayoutBinding uniformBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 2);
	//VkDescriptorSetLayoutBinding lightBufferBinding					= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3);
	//VkDescriptorSetLayoutBinding sampleBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4);	// Samples buffer
	//VkDescriptorSetLayoutBinding gbuffersBinding					= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 5, 3);
	//VkDescriptorSetLayoutBinding materialBinding					= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 6);

	//std::vector<VkDescriptorSetLayoutBinding> bindings({
	//	accelerationStructureLayoutBinding,
	//	storageImageLayoutBinding,
	//	uniformBufferBinding,
	//	lightBufferBinding,
	//	sampleBufferBinding,
	//	gbuffersBinding,
	//	materialBinding
	//});

	//// Allocate Descriptor
	//VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
	//descriptorSetLayoutCreateInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	//descriptorSetLayoutCreateInfo.bindingCount	= static_cast<uint32_t>(bindings.size());
	//descriptorSetLayoutCreateInfo.pBindings		= bindings.data();
	//VK_CHECK(vkCreateDescriptorSetLayout(*device, &descriptorSetLayoutCreateInfo, nullptr, &_shadowDescSetLayout));

	//VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_shadowDescPool, &_shadowDescSetLayout, 1);
	//VK_CHECK(vkAllocateDescriptorSets(*device, &descriptorSetAllocateInfo, &_shadowDescSet));

	//// Binding = 0 AS
	//VkWriteDescriptorSetAccelerationStructureKHR descriptorSetAS{};
	//descriptorSetAS.sType						= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	//descriptorSetAS.accelerationStructureCount	= 1;
	//descriptorSetAS.pAccelerationStructures		= &_topLevelAS.handle;

	//// Binding = 1 Storage Image
	//std::vector<VkDescriptorImageInfo> shadowsInfo(nLights);
	//for (int i = 0; i < nLights; i++)
	//{
	//	shadowsInfo.at(i).imageView = _shadowImages.at(i).imageView;
	//	shadowsInfo.at(i).imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	//}

	//// Binding = 2 Camera data
	//VkDescriptorBufferInfo cameraBufferInfo = vkinit::descriptor_buffer_info(_rtCameraBuffer._buffer, sizeof(RTCameraData));

	//// Binding = 3 lights
	//VkDescriptorBufferInfo lightBufferInfo = vkinit::descriptor_buffer_info(_lightBuffer._buffer, sizeof(uboLight) * nLights);

	//// Binding = 4 Samples
	//if (!_shadowSamplesBuffer._buffer)
	//	VulkanEngine::engine->create_buffer(sizeof(int), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _shadowSamplesBuffer);

	//VkDescriptorBufferInfo samplesDescInfo = vkinit::descriptor_buffer_info(_shadowSamplesBuffer._buffer, sizeof(unsigned int));

	//// Binding = 5 Gbuffers
	//VkDescriptorImageInfo positionDescInfo = vkinit::descriptor_image_info(
	//	_deferredTextures[0].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);
	//VkDescriptorImageInfo normalDescInfo = vkinit::descriptor_image_info(
	//	_deferredTextures[1].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);
	//VkDescriptorImageInfo motionDescInfo = vkinit::descriptor_image_info(
	//	_deferredTextures[3].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Motion GBuffer

	//std::vector<VkDescriptorImageInfo> gbuffersDescInfo = {positionDescInfo, normalDescInfo, motionDescInfo};

	//VkDescriptorBufferInfo materialDescInfo = vkinit::descriptor_buffer_info(_matBuffer._buffer, sizeof(GPUMaterial) * Material::_materials.size());

	//// WRITES ---
	//VkWriteDescriptorSet accelerationStructureWrite = vkinit::write_descriptor_acceleration_structure(_shadowDescSet, &descriptorSetAS, 0);
	//VkWriteDescriptorSet resultImageWrite			= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _shadowDescSet, shadowsInfo.data(), 1, nLights);
	//VkWriteDescriptorSet uniformBufferWrite			= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _shadowDescSet, &cameraBufferInfo, 2);
	//VkWriteDescriptorSet lightsBufferWrite			= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _shadowDescSet, &lightBufferInfo, 3);
	//VkWriteDescriptorSet samplesWrite				= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _shadowDescSet, &samplesDescInfo, 4);
	//VkWriteDescriptorSet gbuffersWrite				= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _shadowDescSet, gbuffersDescInfo.data(), 5, gbuffersDescInfo.size());
	//VkWriteDescriptorSet materialWrite				= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _shadowDescSet, &materialDescInfo, 6);

	//std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
	//	accelerationStructureWrite,
	//	resultImageWrite,
	//	uniformBufferWrite,
	//	lightsBufferWrite,
	//	samplesWrite,
	//	gbuffersWrite,
	//	materialWrite
	//};

	//vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);
	
	// COMPUTE PASS
	//-------------
	//VkDescriptorSetLayoutBinding inputImageLayoutBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 0, nLights);
	//VkDescriptorSetLayoutBinding resultImageLayoutBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 1, nLights);
	VkDescriptorSetLayoutBinding frameLayoutBinding			= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	//VkDescriptorSetLayoutBinding motionLayoutBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 3);
	
	//std::vector<VkDescriptorSetLayoutBinding> denoiseBindings({
	//	inputImageLayoutBinding,
	//	resultImageLayoutBinding,
	//	frameLayoutBinding,
	//	motionLayoutBinding
	//});

	// Allocate Descriptor
	VkDescriptorSetLayoutCreateInfo denoiseDescriptorSetLayoutCreateInfo = {};
	denoiseDescriptorSetLayoutCreateInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	denoiseDescriptorSetLayoutCreateInfo.bindingCount = 1; // static_cast<uint32_t>(denoiseBindings.size());
	denoiseDescriptorSetLayoutCreateInfo.pBindings = &frameLayoutBinding; //denoiseBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &denoiseDescriptorSetLayoutCreateInfo, nullptr, &_sPostDescSetLayout));

	VkDescriptorSetAllocateInfo denoiseDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_shadowDescPool, &_sPostDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &denoiseDescriptorSetAllocateInfo, &_sPostDescSet));

	// Binding = 1 Storage Image
	/*std::vector<VkDescriptorImageInfo> inputImagesInfo(nLights);
	std::vector<VkDescriptorImageInfo> outputImagesInfo(nLights);
	for (int i = 0; i < nLights; i++)
	{
		inputImagesInfo.at(i).imageView = _shadowImages.at(i).imageView;
		inputImagesInfo.at(i).imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		outputImagesInfo.at(i).imageView = _denoisedImages.at(i).imageView;
		outputImagesInfo.at(i).imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	}*/

	// Binding = 2 Frame Count Buffer
	VulkanEngine::engine->create_buffer(sizeof(Surfel)* SURFEL_CAPACITY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _frameCountBuffer);
	VkDescriptorBufferInfo frameDescInfo = vkinit::descriptor_buffer_info(_frameCountBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);

	//VkWriteDescriptorSet inputImageWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _sPostDescSet, inputImagesInfo.data(), 0, nLights);
	//VkWriteDescriptorSet outputImageWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _sPostDescSet, outputImagesInfo.data(), 1, nLights);
	VkWriteDescriptorSet frameBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _sPostDescSet, &frameDescInfo, 0);
	//VkWriteDescriptorSet motionImageWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _sPostDescSet, &motionDescInfo, 3);

	//std::vector<VkWriteDescriptorSet> writeDenoiseDescriptorSets = {
	//	inputImageWrite,
	//	outputImageWrite,
	//	frameBufferWrite,
	//	motionImageWrite
	//};

	vkUpdateDescriptorSets(*device, 1, &frameBufferWrite, 0, VK_NULL_HANDLE);
		//, static_cast<uint32_t>(writeDenoiseDescriptorSets.size()), writeDenoiseDescriptorSets.data(), 0, VK_NULL_HANDLE);
	
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _shadowDescSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _sPostDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _shadowDescPool, nullptr);
		});

}

void Renderer::create_rt_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 1);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_rtDescriptorPool));

	// First set:
	//	binding 0 = AS
	//	binding 1 = storage image
	//	binding 2 = Camera data
	//  binding 3 = Vertex buffer
	//  binding 4 = Index buffer
	//  binding 5 = matrices buffer
	//  binding 6 = light buffer
	//  binding 7 = material buffer
	//  binding 8 = material indices
	//  binding 9 = textures
	//  binding 10 = skybox texture
	//  binding 11 = shadow texture

	const unsigned int nInstances	= _scene->_entities.size();
	const unsigned int nLights		= _scene->_lights.size();
	const unsigned int nMaterials	= Material::_materials.size();
	const unsigned int nTextures	= Texture::_textures.size();

	VkDescriptorSetLayoutBinding accelerationStructureLayoutBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);
	VkDescriptorSetLayoutBinding resultImageLayoutBinding			= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1);
	VkDescriptorSetLayoutBinding uniformBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 2);
	VkDescriptorSetLayoutBinding vertexBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 3, nInstances);
	VkDescriptorSetLayoutBinding indexBufferBinding					= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 4, nInstances);
	VkDescriptorSetLayoutBinding matrixBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 5);
	VkDescriptorSetLayoutBinding lightBufferBinding					= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 6);
	VkDescriptorSetLayoutBinding materialBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 7);
	VkDescriptorSetLayoutBinding matIdxBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 8);
	VkDescriptorSetLayoutBinding texturesBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 9, nTextures);
	VkDescriptorSetLayoutBinding skyboxBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 10, 2);
	VkDescriptorSetLayoutBinding textureBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 11, nLights);
	VkDescriptorSetLayoutBinding sampleBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 12);

	std::vector<VkDescriptorSetLayoutBinding> bindings({
		accelerationStructureLayoutBinding,
		resultImageLayoutBinding,
		uniformBufferBinding,
		vertexBufferBinding,
		indexBufferBinding,
		matrixBufferBinding,
		lightBufferBinding,
		materialBufferBinding,
		matIdxBufferBinding,
		texturesBufferBinding,
		skyboxBufferBinding,
		textureBufferBinding,
		sampleBufferBinding
		});

	// Allocate Descriptor
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = vkinit::descriptor_set_layout_create_info(static_cast<uint32_t>(bindings.size()), bindings);
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &descriptorSetLayoutCreateInfo, nullptr, &_rtDescriptorSetLayout));

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_rtDescriptorPool, &_rtDescriptorSetLayout);
	VK_CHECK(vkAllocateDescriptorSets(*device, &descriptorSetAllocateInfo, &_rtDescriptorSet));

	// Binding = 0 AS
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
	descriptorAccelerationStructureInfo.sType						= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	descriptorAccelerationStructureInfo.accelerationStructureCount	= 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures		= &_topLevelAS.handle;

	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	accelerationStructureWrite.pNext			= &descriptorAccelerationStructureInfo;
	accelerationStructureWrite.dstSet			= _rtDescriptorSet;
	accelerationStructureWrite.dstBinding		= 0;
	accelerationStructureWrite.descriptorCount	= 1;
	accelerationStructureWrite.descriptorType	= VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	// Binding = 1 Storage Image
	VkDescriptorImageInfo storageImageDescriptor = vkinit::descriptor_image_info(_rtImage.imageView, VK_IMAGE_LAYOUT_GENERAL);

	// Binding = 2 Camera 
	VkDescriptorBufferInfo _rtDescriptorBufferInfo = vkinit::descriptor_buffer_info(_rtCameraBuffer._buffer, sizeof(RTCameraData));

	std::vector<VkDescriptorBufferInfo> vertexDescInfo;
	std::vector<VkDescriptorBufferInfo> indexDescInfo;
	std::vector<glm::vec4> idVector;
	for (Object* obj : _scene->_entities)
	{
		std::vector<Vertex> vertices = obj->prefab->_mesh->_vertices;
		std::vector<uint32_t> indices = obj->prefab->_mesh->_indices;
		size_t vertexBufferSize = sizeof(rtVertexAttribute) * vertices.size();
		size_t indexBufferSize = sizeof(uint32_t) * indices.size();
		AllocatedBuffer vBuffer;
		VulkanEngine::engine->create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, vBuffer);

		std::vector<rtVertexAttribute> vAttr;
		vAttr.reserve(vertices.size());
		for (Vertex& v : vertices) {
			vAttr.push_back({ {v.normal.x, v.normal.y, v.normal.z, 1}, {v.color.x, v.color.y, v.color.z, 1}, {v.uv.x, v.uv.y, 1, 1} });
		}

		void* vdata;
		vmaMapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation, &vdata);
		memcpy(vdata, vAttr.data(), vertexBufferSize);
		vmaUnmapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation);

		// Binding = 3 Vertices buffer
		VkDescriptorBufferInfo vertexBufferDescriptor = vkinit::descriptor_buffer_info(vBuffer._buffer, vertexBufferSize);
		vertexDescInfo.push_back(vertexBufferDescriptor);

		// Binding = 4 Indices buffer
		VkDescriptorBufferInfo indexBufferDescriptor = vkinit::descriptor_buffer_info(obj->prefab->_mesh->_indexBuffer._buffer, indexBufferSize);
		indexDescInfo.push_back(indexBufferDescriptor);

		for (Node* root : obj->prefab->_root)
		{
			root->fill_index_buffer(idVector);
		}
	}

	// Binding = 5 Matrix buffer
	VkDescriptorBufferInfo matrixDescInfo = vkinit::descriptor_buffer_info(_matricesBuffer._buffer, sizeof(glm::mat4) * _scene->_matricesVector.size());

	// Binding = 6 lights
	VkDescriptorBufferInfo lightBufferInfo = vkinit::descriptor_buffer_info(_lightBuffer._buffer, sizeof(uboLight) * nLights);

	// Binding = 7 ID buffer
	if (!_idBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(glm::vec4) * idVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _idBuffer);

	void* idData;
	vmaMapMemory(VulkanEngine::engine->_allocator, _idBuffer._allocation, &idData);
	memcpy(idData, idVector.data(), sizeof(glm::vec4) * idVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _idBuffer._allocation);

	VkDescriptorBufferInfo idDescInfo = vkinit::descriptor_buffer_info(_idBuffer._buffer, sizeof(glm::vec4) * idVector.size());

	// Binding = 8 Materials
	VkDescriptorBufferInfo materialBufferInfo = vkinit::descriptor_buffer_info(_matBuffer._buffer, sizeof(GPUMaterial) * nMaterials);

	// Binding = 9 Textures
	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSampler sampler;
	vkCreateSampler(*device, &samplerInfo, nullptr, &sampler);

	std::vector<VkDescriptorImageInfo> imageInfos;
	for (auto const& texture : Texture::_textures)
	{
		VkDescriptorImageInfo imageBufferInfo = vkinit::descriptor_image_info(texture.second->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sampler);
		imageInfos.push_back(imageBufferInfo);
	}

	// Binding = 10 Skybox
	VkDescriptorImageInfo skyboxImagesDesc[2];
	skyboxImagesDesc[0] = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_8k.jpg")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	skyboxImagesDesc[1] = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_Env.hdr")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	// Binding = 11 Shadow texture
	std::vector<VkDescriptorImageInfo> shadowImagesDesc(_denoisedImages.size());
	for (decltype(_denoisedImages.size()) i = 0; i < _denoisedImages.size(); i++)
	{
		shadowImagesDesc[i] = { nullptr, _denoisedImages[i].imageView, VK_IMAGE_LAYOUT_GENERAL };
	}

	// Binding = 12 Sample buffer
	if (!_shadowSamplesBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(int), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _shadowSamplesBuffer);

	VkDescriptorBufferInfo samplesDescInfo = vkinit::descriptor_buffer_info(_shadowSamplesBuffer._buffer, sizeof(unsigned int));

	// WRITES ---
	VkWriteDescriptorSet resultImageWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _rtDescriptorSet, &storageImageDescriptor, 1);
	VkWriteDescriptorSet uniformBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _rtDescriptorSet, &_rtDescriptorBufferInfo, 2);
	VkWriteDescriptorSet vertexBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _rtDescriptorSet, vertexDescInfo.data(), 3, nInstances);
	VkWriteDescriptorSet indexBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _rtDescriptorSet, indexDescInfo.data(), 4, nInstances);
	VkWriteDescriptorSet matrixBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _rtDescriptorSet, &matrixDescInfo, 5);
	VkWriteDescriptorSet lightsBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _rtDescriptorSet, &lightBufferInfo, 6);
	VkWriteDescriptorSet matBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _rtDescriptorSet, &materialBufferInfo, 7);
	VkWriteDescriptorSet matIdxBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _rtDescriptorSet, &idDescInfo, 8);
	VkWriteDescriptorSet textureBufferWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _rtDescriptorSet, imageInfos.data(), 9, nTextures);
	VkWriteDescriptorSet skyboxBufferWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _rtDescriptorSet, skyboxImagesDesc, 10, 2);
	VkWriteDescriptorSet shadowBufferWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _rtDescriptorSet, shadowImagesDesc.data(), 11, nLights);
	VkWriteDescriptorSet sampleWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _rtDescriptorSet, &samplesDescInfo, 12);

	std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
		accelerationStructureWrite,
		resultImageWrite,
		uniformBufferWrite,
		vertexBufferWrite,
		indexBufferWrite,
		matrixBufferWrite,
		lightsBufferWrite,
		matBufferWrite,
		matIdxBufferWrite,
		textureBufferWrite,
		skyboxBufferWrite,
		shadowBufferWrite,
		sampleWrite
	};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, VK_NULL_HANDLE);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _rtDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _rtDescriptorPool, nullptr);
		vkDestroySampler(*device, sampler, nullptr);
		});
}

void Renderer::init_raytracing_pipeline()
{
	VulkanEngine* engine = VulkanEngine::engine;

	// Setup ray tracing shader groups
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages = {};
	std::vector<VkPipelineShaderStageCreateInfo> hybridShaderStages = {};
	std::vector<VkPipelineShaderStageCreateInfo> shadowShaderStages = {};

	// Ray generation group
	VkShaderModule rayGenModule, hraygenModule, sraygenModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("raygen.rgen.spv", searchPaths, true).c_str(), &rayGenModule, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("hybridRaygen.rgen.spv", searchPaths, true).c_str(), &hraygenModule, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
		shadowShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadowRaygen.rgen.spv", searchPaths, true).c_str(), &sraygenModule, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType				= VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type				= VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		shaderGroup.generalShader		= 0;
		shaderGroup.closestHitShader	= VK_SHADER_UNUSED_KHR;
		shaderGroup.anyHitShader		= VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader	= VK_SHADER_UNUSED_KHR;

		shaderGroups.push_back(shaderGroup);
		hybridShaderGroups.push_back(shaderGroup);
		shadowShaderGroups.push_back(shaderGroup);
	}

	// Miss group
	VkShaderModule missModule, hmissModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("miss.rmiss.spv", searchPaths, true).c_str(), &missModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("miss.rmiss.spv", searchPaths, true).c_str(), &hmissModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType				= VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type				= VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		shaderGroup.generalShader		= static_cast<uint32_t>(shaderStages.size()) - 1;
		shaderGroup.closestHitShader	= VK_SHADER_UNUSED_KHR;
		shaderGroup.anyHitShader		= VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader	= VK_SHADER_UNUSED_KHR;

		shaderGroups.push_back(shaderGroup);
		shaderGroup.generalShader = static_cast<uint32_t>(hybridShaderStages.size()) - 1;
		hybridShaderGroups.push_back(shaderGroup);
	}

	// Shadow miss
	VkShaderModule shadowModule, hshadowModule, sshadowModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadow.rmiss.spv", searchPaths, true).c_str(), &shadowModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadow.rmiss.spv", searchPaths, true).c_str(), &hshadowModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		shadowShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadow.rmiss.spv", searchPaths, true).c_str(), &sshadowModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType				= VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type				= VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		shaderGroup.generalShader		= static_cast<uint32_t>(shaderStages.size()) - 1;
		shaderGroup.closestHitShader	= VK_SHADER_UNUSED_KHR;
		shaderGroup.anyHitShader		= VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader	= VK_SHADER_UNUSED_KHR;
		shaderGroups.push_back(shaderGroup);

		shaderGroup.generalShader = static_cast<uint32_t>(hybridShaderStages.size()) - 1;
		hybridShaderGroups.push_back(shaderGroup);
		shaderGroup.generalShader = static_cast<uint32_t>(shadowShaderStages.size()) - 1;
		shadowShaderGroups.push_back(shaderGroup);
	}

	// Hit group
	VkShaderModule hitModule, hhitModule, shitModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("closesthit.rchit.spv", searchPaths, true).c_str(), &hitModule, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("hybridHit.rchit.spv", searchPaths, true).c_str(), &hhitModule, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
		shadowShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadowHit.rchit.spv", searchPaths, true).c_str(), &shitModule, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType				= VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type				= VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
		shaderGroup.generalShader		= VK_SHADER_UNUSED_KHR;
		shaderGroup.closestHitShader	= static_cast<uint32_t>(shaderStages.size()) - 1;
		shaderGroup.anyHitShader		= VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader	= VK_SHADER_UNUSED_KHR;
		shaderGroups.push_back(shaderGroup);
		
		shaderGroup.closestHitShader = static_cast<uint32_t>(hybridShaderStages.size()) - 1;
		hybridShaderGroups.push_back(shaderGroup);
		shaderGroup.closestHitShader = static_cast<uint32_t>(shadowShaderStages.size()) - 1;
		shadowShaderGroups.push_back(shaderGroup);
	}

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
	pipelineLayoutCreateInfo.sType				= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount		= 1;
	pipelineLayoutCreateInfo.pSetLayouts		= &_rtDescriptorSetLayout;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCreateInfo, nullptr, &_rtPipelineLayout));

	// Create RT pipeline 
	VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo{};
	rtPipelineCreateInfo.sType							= VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	rtPipelineCreateInfo.stageCount						= static_cast<uint32_t>(shaderStages.size());
	rtPipelineCreateInfo.pStages						= shaderStages.data();
	rtPipelineCreateInfo.groupCount						= static_cast<uint32_t>(shaderGroups.size());
	rtPipelineCreateInfo.pGroups						= shaderGroups.data();
	rtPipelineCreateInfo.maxPipelineRayRecursionDepth	= 4;
	rtPipelineCreateInfo.layout							= _rtPipelineLayout;

	VK_CHECK(vkCreateRayTracingPipelinesKHR(*device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rtPipelineCreateInfo, nullptr, &_rtPipeline));

	// HYBRID PIPELINE CREATION - using the deferred pass
	VkPipelineLayoutCreateInfo hybridPipelineLayoutInfo{};
	hybridPipelineLayoutInfo.sType					= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	hybridPipelineLayoutInfo.setLayoutCount			= 1;
	hybridPipelineLayoutInfo.pSetLayouts			= &_hybridDescSetLayout;
	VK_CHECK(vkCreatePipelineLayout(*device, &hybridPipelineLayoutInfo, nullptr, &_hybridPipelineLayout));

	VkRayTracingPipelineCreateInfoKHR hybridPipelineInfo{};
	hybridPipelineInfo.sType						= VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	hybridPipelineInfo.stageCount					= static_cast<uint32_t>(hybridShaderStages.size());
	hybridPipelineInfo.pStages						= hybridShaderStages.data();
	hybridPipelineInfo.groupCount					= static_cast<uint32_t>(hybridShaderGroups.size());
	hybridPipelineInfo.pGroups						= hybridShaderGroups.data();
	hybridPipelineInfo.maxPipelineRayRecursionDepth = 4;
	hybridPipelineInfo.layout						= _hybridPipelineLayout;

	VK_CHECK(vkCreateRayTracingPipelinesKHR(*device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &hybridPipelineInfo, nullptr, &_hybridPipeline));

	// SHADOW PIPELINA CREATION
	VkPipelineLayoutCreateInfo shadowPipelineLayoutInfo = {};
	shadowPipelineLayoutInfo.sType					= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	shadowPipelineLayoutInfo.setLayoutCount			= 1;
	shadowPipelineLayoutInfo.pSetLayouts			= &_shadowDescSetLayout;
	VK_CHECK(vkCreatePipelineLayout(*device, &shadowPipelineLayoutInfo, nullptr, &_shadowPipelineLayout));

	VkRayTracingPipelineCreateInfoKHR shadowPipelineInfo{};
	shadowPipelineInfo.sType						= VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	shadowPipelineInfo.stageCount					= static_cast<uint32_t>(shadowShaderStages.size());
	shadowPipelineInfo.pStages						= shadowShaderStages.data();
	shadowPipelineInfo.groupCount					= static_cast<uint32_t>(shadowShaderGroups.size());
	shadowPipelineInfo.pGroups						= shadowShaderGroups.data();
	shadowPipelineInfo.maxPipelineRayRecursionDepth = 2;
	shadowPipelineInfo.layout						= _shadowPipelineLayout;

	VK_CHECK(vkCreateRayTracingPipelinesKHR(*device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &shadowPipelineInfo, nullptr, &_shadowPipeline));

	vkDestroyShaderModule(*device, rayGenModule, nullptr);
	vkDestroyShaderModule(*device, hitModule, nullptr);
	vkDestroyShaderModule(*device, missModule, nullptr);
	vkDestroyShaderModule(*device, shadowModule, nullptr);
	vkDestroyShaderModule(*device, hraygenModule, nullptr);
	vkDestroyShaderModule(*device, hmissModule, nullptr);
	vkDestroyShaderModule(*device, hshadowModule, nullptr);
	vkDestroyShaderModule(*device, hhitModule, nullptr);
	vkDestroyShaderModule(*device, sraygenModule, nullptr);
	vkDestroyShaderModule(*device, sshadowModule, nullptr);
	vkDestroyShaderModule(*device, shitModule, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _rtPipeline, nullptr);
		vkDestroyPipeline(*device, _hybridPipeline, nullptr);
		vkDestroyPipeline(*device, _shadowPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _rtPipelineLayout, nullptr);
		vkDestroyPipelineLayout(*device, _hybridPipelineLayout, nullptr);
		vkDestroyPipelineLayout(*device, _shadowPipelineLayout, nullptr);
		});
}

void Renderer::init_compute_pipeline()
{
	VkShaderModule computeShaderModule;
	
	//system("glslc ./data/prueba.comp -o ./data/output/prueba.comp.spv");

	VulkanEngine::engine->load_shader_module(vkutil::findFile("prueba.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	VkPushConstantRange _constantRangeCI = {};
	_constantRangeCI.offset = 0;
	_constantRangeCI.size = sizeof(int);
	_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts	= &_sPostDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = &_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_sPostPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType		= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage		= shaderStageCI;
	computePipelineCI.layout	= _sPostPipelineLayout;
	
	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_sPostPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _sPostPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _sPostPipelineLayout, nullptr);
		});
}
//
void Renderer::create_shader_binding_table()
{
	// RAYTRACING BUFFERS
	const uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());	// 4 shaders: raygen, miss, shadowmiss and hit
	const uint32_t handleSize = VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;	// Size of a programm identifier
	const uint32_t handleAlignment = VulkanEngine::engine->_rtProperties.shaderGroupHandleAlignment;
	const uint32_t sbtSize = groupCount * handleSize;

	std::vector<uint8_t> shaderHandleStorage(sbtSize);
	VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(VulkanEngine::engine->_device, _rtPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()));

	const VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, raygenShaderBindingTable);
	VulkanEngine::engine->create_buffer(handleSize * 2, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, missShaderBindingTable);
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, hitShaderBindingTable);

	void* rayGenData, *missData, *hitData;
	vmaMapMemory(VulkanEngine::engine->_allocator, raygenShaderBindingTable._allocation, &rayGenData);
	memcpy(rayGenData, shaderHandleStorage.data(), handleSize);
	vmaMapMemory(VulkanEngine::engine->_allocator, missShaderBindingTable._allocation, &missData);
	memcpy(missData, shaderHandleStorage.data() + handleAlignment, handleSize * 2);
	vmaMapMemory(VulkanEngine::engine->_allocator, hitShaderBindingTable._allocation, &hitData);
	memcpy(hitData, shaderHandleStorage.data() + handleAlignment * 3, handleSize);

	vmaUnmapMemory(VulkanEngine::engine->_allocator, raygenShaderBindingTable._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, missShaderBindingTable._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, hitShaderBindingTable._allocation);

	// HYBRID BUFFERS
	const uint32_t hybridCount		= static_cast<uint32_t>(hybridShaderGroups.size());
	const uint32_t hybridSbtSize	= hybridCount * handleSize;

	std::vector<uint8_t> hybridShaderHandleStorage(hybridSbtSize);
	VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(*device, _hybridPipeline, 0, hybridCount, hybridSbtSize, hybridShaderHandleStorage.data()));

	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, raygenSBT);
	VulkanEngine::engine->create_buffer(handleSize * 2, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, missSBT);
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, hitSBT);

	vmaMapMemory(VulkanEngine::engine->_allocator, raygenSBT._allocation, &rayGenData);
	memcpy(rayGenData, hybridShaderHandleStorage.data(), handleSize);
	vmaMapMemory(VulkanEngine::engine->_allocator, missSBT._allocation, &missData);
	memcpy(missData, hybridShaderHandleStorage.data() + handleAlignment, handleSize * 2);
	vmaMapMemory(VulkanEngine::engine->_allocator, hitSBT._allocation, &hitData);
	memcpy(hitData, hybridShaderHandleStorage.data() + handleAlignment * 3, handleSize);

	vmaUnmapMemory(VulkanEngine::engine->_allocator, raygenSBT._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, missSBT._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, hitSBT._allocation);

	// SHADOW BUFFERS
	const uint32_t shadowCount		= static_cast<uint32_t>(shadowShaderGroups.size());
	const uint32_t shadowSbtSize	= shadowCount * handleSize;

	std::vector<uint8_t> shadowShaderHandleStorage(shadowSbtSize);
	VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(*device, _shadowPipeline, 0, shadowCount, shadowSbtSize, shadowShaderHandleStorage.data()));

	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, sraygenSBT);
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, smissSBT);
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, shitSBT);

	vmaMapMemory(VulkanEngine::engine->_allocator, sraygenSBT._allocation, &rayGenData);
	memcpy(rayGenData, shadowShaderHandleStorage.data(), handleSize);
	vmaMapMemory(VulkanEngine::engine->_allocator, smissSBT._allocation, &missData);
	memcpy(missData, shadowShaderHandleStorage.data() + handleAlignment, handleSize);
	vmaMapMemory(VulkanEngine::engine->_allocator, shitSBT._allocation, &hitData);
	memcpy(hitData, shadowShaderHandleStorage.data() + handleAlignment * 2, handleSize);

	vmaUnmapMemory(VulkanEngine::engine->_allocator, sraygenSBT._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, smissSBT._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, shitSBT._allocation);
}
//
//void Renderer::build_raytracing_command_buffers()
//{
//	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
//
//	VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
//	
//	VkCommandBuffer& cmd = _rtCommandBuffer;
//
//	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBufInfo));
//
//	VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
//	raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(raygenShaderBindingTable._buffer);
//	raygenShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	raygenShaderSbtEntry.size			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	
//	VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
//	missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(missShaderBindingTable._buffer);
//	missShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize * 2;
//	
//	VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
//	hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(hitShaderBindingTable._buffer);
//	hitShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	hitShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//
//	VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};
//
//	uint32_t width = VulkanEngine::engine->_window->getWidth(), height = VulkanEngine::engine->_window->getHeight();
//
//	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _rtPipeline);
//	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _rtPipelineLayout, 0, 1, &_rtDescriptorSet, 0, nullptr);
//	
//	vkCmdTraceRaysKHR(
//		cmd,
//		&raygenShaderSbtEntry,
//		&missShaderSbtEntry,
//		&hitShaderSbtEntry,
//		&callableShaderSbtEntry,
//		width,
//		height,
//		1
//	);
//	
//	VK_CHECK(vkEndCommandBuffer(cmd));
//}
//
//void Renderer::build_shadow_command_buffer()
//{
//	VkCommandBufferBeginInfo cmdBuffInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
//	VkCommandBuffer& cmd = _shadowCommandBuffer;
//
//	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBuffInfo));
//
//	VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
//	raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(sraygenSBT._buffer);
//	raygenShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	raygenShaderSbtEntry.size			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//
//	VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
//	missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(smissSBT._buffer);
//	missShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//
//	VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
//	hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(shitSBT._buffer);
//	hitShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	hitShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//
//	VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};
//
//	uint32_t width = VulkanEngine::engine->_window->getWidth(), height = VulkanEngine::engine->_window->getHeight();
//
//	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _shadowPipeline);
//	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _shadowPipelineLayout, 0, 1, &_shadowDescSet, 0, nullptr);
//
//	vkCmdTraceRaysKHR(
//		cmd,
//		&raygenShaderSbtEntry,
//		&missShaderSbtEntry,
//		&hitShaderSbtEntry,
//		&callableShaderSbtEntry,
//		width,
//		height,
//		1
//	);
//
//	VK_CHECK(vkEndCommandBuffer(cmd));
//}

void Renderer::build_compute_command_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	VkCommandBuffer &cmd = _denoiseCommandBuffer;

	VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _sPostPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _sPostPipelineLayout, 0, 1, &_sPostDescSet, 0, nullptr);

	//shaderIf = 1;

	int x = 1;

	vkCmdPushConstants(cmd, _sPostPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &x);

	vkCmdDispatch(cmd, VulkanEngine::engine->_window->getWidth() / 16, VulkanEngine::engine->_window->getHeight() / 16, 1);

	VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::create_SurfelGi_resources()
{
	//preguntar pau cómo crear texturas
	//sizeof(SurfelGridCell)* SURFEL_TABLE_SIZE
	VulkanEngine::engine->create_buffer(
		(VulkanEngine::engine->_window->getWidth() / 16) * (VulkanEngine::engine->_window->getHeight() / 16 * sizeof(glm::vec2)), 
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, _SurfelPositionBuffer);
	VulkanEngine::engine->create_buffer(sizeof(Surfel) * SURFEL_CAPACITY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, _SurfelBuffer);
	VulkanEngine::engine->create_buffer(sizeof(SurfelData) * SURFEL_CAPACITY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, _SurfelDataBuffer);
	VulkanEngine::engine->create_buffer(sizeof(unsigned int) * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, _SurfelStatsBuffer);
	VulkanEngine::engine->create_buffer(sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, _SurfelGridBuffer);
	VulkanEngine::engine->create_buffer(sizeof(unsigned int) * SURFEL_CAPACITY * 27, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY, _SurfelCellBuffer);
}

void Renderer::todo_de_nuevo()
{
	//std::vector<VkDescriptorPoolSize> poolSize = {
	//	{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
	//	{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
	//	{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
	//	{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
	//	{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
	//};

	//VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	//VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_buffer2DescPool));

	//VkDescriptorSetLayoutBinding frameLayoutBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);

	//VkDescriptorSetLayoutCreateInfo denoiseDescriptorSetLayoutCreateInfo = {};
	//denoiseDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	//denoiseDescriptorSetLayoutCreateInfo.bindingCount = 1; // static_cast<uint32_t>(denoiseBindings.size());
	//denoiseDescriptorSetLayoutCreateInfo.pBindings = &frameLayoutBinding; //denoiseBindings.data();
	//VK_CHECK(vkCreateDescriptorSetLayout(*device, &denoiseDescriptorSetLayoutCreateInfo, nullptr, &_buffer2DescSetLayout));

	//VkDescriptorSetAllocateInfo denoiseDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_buffer2DescPool, &_buffer2DescSetLayout, 1);
	//VK_CHECK(vkAllocateDescriptorSets(*device, &denoiseDescriptorSetAllocateInfo, &_buffer2DescSet));

	////Binding = 2 Frame Count Buffer
	//VulkanEngine::engine->create_buffer(sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _actualBuffer2);
	//VkDescriptorBufferInfo frameDescInfo = vkinit::descriptor_buffer_info(_actualBuffer2._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);
	//
	////VkDescriptorBufferInfo frameDescInfo = vkinit::descriptor_buffer_info(_frameCountBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);


	//VkWriteDescriptorSet frameBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _buffer2DescSet, &frameDescInfo, 0);


	//vkUpdateDescriptorSets(*device, 1, &frameBufferWrite, 0, VK_NULL_HANDLE);

	//VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
	//	//vkDestroyDescriptorSetLayout(*device, _buffer2DescSetLayout, nullptr);
	//	vkDestroyDescriptorSetLayout(*device, _buffer2DescSetLayout, nullptr);
	//	vkDestroyDescriptorPool(*device, _buffer2DescPool, nullptr);
	//	});


	////-----------------------------------------------------------------------------------------------------------------------------------------------------

	//VkShaderModule computeShaderModule;

	////system("glslc ./data/prueba.comp -o ./data/output/prueba.comp.spv");

	//VulkanEngine::engine->load_shader_module(vkutil::findFile("gridreset.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	//VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	//VkPushConstantRange _constantRangeCI = {};
	//_constantRangeCI.offset = 0;
	//_constantRangeCI.size = sizeof(int);
	//_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	//VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	//pipelineLayoutCI.setLayoutCount = 1;
	//pipelineLayoutCI.pSetLayouts = &_buffer2DescSetLayout;
	//pipelineLayoutCI.pPushConstantRanges = &_constantRangeCI;
	//pipelineLayoutCI.pushConstantRangeCount = 1;
	//VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_buffer2PipelineLayout));

	//VkComputePipelineCreateInfo computePipelineCI = {};
	//computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	//computePipelineCI.stage = shaderStageCI;
	//computePipelineCI.layout = _buffer2PipelineLayout;

	//VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_buffer2Pipeline));

	//// Fill the buffer

	//vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	//VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
	//	vkDestroyPipeline(*device, _buffer2Pipeline, nullptr);
	//	vkDestroyPipelineLayout(*device, _buffer2PipelineLayout, nullptr);
	//	});

	////-------------------------------------------------------------------------------------------------------------------------------------


	//VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _commandbuffer2;

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	//vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _buffer2Pipeline);
	//vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _buffer2PipelineLayout, 0, 1, &_buffer2DescSet, 0, nullptr);


	////shaderIf = 2;

	//int x = 2;

	//vkCmdPushConstants(cmd, _buffer2PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &x);

	//vkCmdDispatch(cmd, (SURFEL_TABLE_SIZE + 63) / 64, 1, 1);

	//VK_CHECK(vkEndCommandBuffer(cmd));

}


void Renderer::surfel_position()
{
	
	create_surfel_position_descriptors();

	//-----------------------------------------------------------------------------------------------------------------------------------------------------

	init_surfel_position_pipeline();

	//-------------------------------------------------------------------------------------------------------------------------------------

	build_surfel_position_command_buffer();

}

void Renderer::prepare_indirect()
{
	create_prepare_indirect_descriptors();

	init_prepare_indirect_pipeline();

	build_prepare_indirect_buffer();
}

void Renderer::grid_reset()
{
	create_grid_reset_descriptors();

	init_grid_reset_pipeline();

	build_grid_reset_buffer();
}

void Renderer::update_surfels()
{
	create_update_surfels_descriptors();

	init_update_surfels_pipeline();

	build_update_surfels_buffer();
}

void Renderer::grid_offset()
{
	create_grid_offset_descriptors();

	init_grid_offset_pipeline();

	build_grid_offset_buffer();
}

void Renderer::surfel_binning()
{
	create_surfel_binning_descriptors();

	init_surfel_binning_pipeline();

	build_surfel_binning_buffer();
}

void Renderer::surfel_ray_tracing()
{
	create_surfel_rtx_descriptors();

	create_surfel_rtx_pipeline();

	create_surfel_rtx_SBT();

	create_surfel_rtx_cmd_buffer();
}

void Renderer::surfel_shade()
{
	create_surfel_shade_descriptors();

	init_surfel_shade_pipeline();

	build_surfel_shade_buffer();
}


//------------------------------------------------------------------- Coverage

void Renderer::create_surfel_position_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
	{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
	{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
	{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
	{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
	{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
	{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_SurfelPositionDescPool));

	VkDescriptorSetLayoutBinding _PositionBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	VkDescriptorSetLayoutBinding normalBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 1);
	VkDescriptorSetLayoutBinding statsBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2);
	//VkDescriptorSetLayoutBinding positionBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 3);
	VkDescriptorSetLayoutBinding _GridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4);
	VkDescriptorSetLayoutBinding _CellBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 5);
	VkDescriptorSetLayoutBinding cameraBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 6);			// Camera buffer
	VkDescriptorSetLayoutBinding depthBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 7);

	VkDescriptorSetLayoutBinding debugImageLayoutBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 8);
	VkDescriptorSetLayoutBinding resultImageLayoutBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT, 9);


	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		_PositionBufferBinding,
		normalBinding,
		statsBinding,
		depthBinding,
		//positionBinding,
		_GridBufferBinding,
		_CellBufferBinding,
		cameraBufferBinding,
		debugImageLayoutBinding,
		resultImageLayoutBinding
	};

	
	VkDescriptorSetLayoutCreateInfo positionDescriptorSetLayoutCreateInfo = {};
	positionDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	positionDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	positionDescriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &positionDescriptorSetLayoutCreateInfo, nullptr, &_SurfelPositionDescSetLayout));

	VkDescriptorSetAllocateInfo positionDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_SurfelPositionDescPool, &_SurfelPositionDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &positionDescriptorSetAllocateInfo, &_SurfelPositionDescSet));


	VkSamplerCreateInfo sampler = vkinit::sampler_create_info(VK_FILTER_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
	sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	sampler.mipLodBias = 0.0f;
	sampler.maxAnisotropy = 1.0f;
	sampler.minLod = 0.0f;
	sampler.maxLod = 1.0f;
	sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	VK_CHECK(vkCreateSampler(*device, &sampler, nullptr, &_SurfelPositionNormalSampler));

	//std::cout << _scene->_camera->_position.x << ", " << _scene->_camera->_position.y << ", " << _scene->_camera->_position.z << std::endl;

	VkDescriptorBufferInfo surfelDescInfo = vkinit::descriptor_buffer_info(_SurfelBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);

	VkDescriptorImageInfo texDescriptorNormal = vkinit::descriptor_image_info(_deferredTextures[1].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _SurfelPositionNormalSampler);

	VkDescriptorImageInfo depthDescriptorDepth = vkinit::descriptor_image_info(_deferredTextures[6].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _SurfelPositionNormalSampler);
	//VkDescriptorImageInfo positionDescriptorInfo = vkinit::descriptor_image_info(_deferredTextures[0].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _SurfelPositionNormalSampler);
	//VkDescriptorImageInfo positionDescriptorInfo = { _SurfelPositionNormalSampler, Texture::GET("blueNoise.png")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	
	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);
	
	VkDescriptorBufferInfo cellDescInfo = vkinit::descriptor_buffer_info(_SurfelCellBuffer._buffer, sizeof(unsigned int) * SURFEL_CAPACITY * 27);

	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	VkDescriptorBufferInfo cameraBufferInfo = vkinit::descriptor_buffer_info(_cameraBuffer._buffer, sizeof(GPUCameraData));

	VkDescriptorImageInfo debugImageDescriptor = vkinit::descriptor_image_info(_debugGI.imageView, VK_IMAGE_LAYOUT_GENERAL);

	VkDescriptorImageInfo resultImageDescriptor = vkinit::descriptor_image_info(_result.imageView, VK_IMAGE_LAYOUT_GENERAL);


	VkWriteDescriptorSet surfelBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelPositionDescSet, &surfelDescInfo, 0);
	VkWriteDescriptorSet normalWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _SurfelPositionDescSet, &texDescriptorNormal, 1);
	VkWriteDescriptorSet statsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelPositionDescSet, &statsDescInfo, 2);
	//VkWriteDescriptorSet positionWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _SurfelPositionDescSet, &positionDescriptorInfo, 3);
	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelPositionDescSet, &gridDescInfo, 4);
	VkWriteDescriptorSet CellWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelPositionDescSet, &cellDescInfo, 5);
	VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _SurfelPositionDescSet, &cameraBufferInfo, 6);
	VkWriteDescriptorSet depthWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _SurfelPositionDescSet, &depthDescriptorDepth, 7);
	VkWriteDescriptorSet debugWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _SurfelPositionDescSet, &debugImageDescriptor, 8);
	VkWriteDescriptorSet resultWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _SurfelPositionDescSet, &resultImageDescriptor, 9);

	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		surfelBufferWrite,
		normalWrite,
		statsWrite,
		depthWrite,
		//positionWrite,
		GridWrite,
		CellWrite,
		cameraWrite,
		debugWrite,
		resultWrite
	};

	//vkUpdateDescriptorSets(*device, 1, &_PositionBufferWrite, 0, VK_NULL_HANDLE);

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);
	
	//vkUpdateDescriptorSets(*device, 1, &normalWrite, 0, VK_NULL_HANDLE);
	//vkUpdateDescriptorSets(*device, 1, &resultWrite, 0, VK_NULL_HANDLE);


	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		//vkDestroyDescriptorSetLayout(*device, _buffer3DescSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _SurfelPositionDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _SurfelPositionDescPool, nullptr);
		vkDestroySampler(*device, _SurfelPositionNormalSampler, nullptr);
		});
}

void Renderer::init_surfel_position_pipeline()
{
	VkShaderModule computeShaderModule;

	//system("glslc ./data/prueba.comp -o ./data/output/prueba.comp.spv");

	VulkanEngine::engine->load_shader_module(vkutil::findFile("surfelRandomPos.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	VkPushConstantRange _constantRangeCI = {};
	_constantRangeCI.offset = 0;
	_constantRangeCI.size = sizeof(int);
	_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_SurfelPositionDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr; //&_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0; //1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_SurfelPositionPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _SurfelPositionPipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_SurfelPositionPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _SurfelPositionPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _SurfelPositionPipelineLayout, nullptr);
		});
}

void Renderer::build_surfel_position_command_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	VkCommandBuffer& cmd = _SurfelPositionCmd;

	VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _SurfelPositionPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _SurfelPositionPipelineLayout, 0, 1, &_SurfelPositionDescSet, 0, nullptr);

	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;


	VkBufferMemoryBarrier bufferbarrierdesc = {};
	bufferbarrierdesc.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc.pNext = nullptr;
	bufferbarrierdesc.buffer = _SurfelStatsBuffer._buffer;
	bufferbarrierdesc.size = sizeof(unsigned int) * 8;
	bufferbarrierdesc.offset = 0;
	bufferbarrierdesc.srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;


	//shaderIf = 2;


	

	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		dstStage,
		0,
		0, nullptr,
		1, &bufferbarrierdesc,
		0, nullptr
	);

	//int t = static_cast<int> (time(NULL));


	//vkCmdPushConstants(cmd, _SurfelPositionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);

	vkCmdDispatch(cmd, ((uint32_t)VulkanEngine::engine->_window->getWidth() + 15) / 16, ((uint32_t)VulkanEngine::engine->_window->getHeight() + 15) / 16, 1);



	VkMemoryBarrier memorybarrierdesc = {};
	memorybarrierdesc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memorybarrierdesc.pNext = nullptr;
	memorybarrierdesc.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	memorybarrierdesc.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		1, &memorybarrierdesc,
		0, nullptr,
		0, nullptr
	);


	//VK_CHECK(vkEndCommandBuffer(cmd));
}


//------------------------------------------------------------------- Indirect Prepare


void Renderer::create_prepare_indirect_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
	{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_PrepareIndirectDescPool));

	VkDescriptorSetLayoutBinding _StatsBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		_StatsBufferBinding
	};

	VkDescriptorSetLayoutCreateInfo prepareIndirectDescriptorSetLayoutCreateInfo = {};
	prepareIndirectDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	prepareIndirectDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	prepareIndirectDescriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &prepareIndirectDescriptorSetLayoutCreateInfo, nullptr, &_PrepareIndirectDescSetLayout));

	VkDescriptorSetAllocateInfo prepareIndirectDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_PrepareIndirectDescPool, &_PrepareIndirectDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &prepareIndirectDescriptorSetAllocateInfo, &_PrepareIndirectDescSet));


	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	VkWriteDescriptorSet StatsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _PrepareIndirectDescSet, &statsDescInfo, 0);


	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		StatsWrite
	};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);



	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		//vkDestroyDescriptorSetLayout(*device, _buffer3DescSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _PrepareIndirectDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _PrepareIndirectDescPool, nullptr);
		});
}

void Renderer::init_prepare_indirect_pipeline()
{
	VkShaderModule computeShaderModule;

	//system("glslc ./data/prueba.comp -o ./data/output/prueba.comp.spv");

	VulkanEngine::engine->load_shader_module(vkutil::findFile("prepareIndirect.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	//VkPushConstantRange _constantRangeCI = {};
	//_constantRangeCI.offset = 0;
	//_constantRangeCI.size = sizeof(int);
	//_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_PrepareIndirectDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr;// &_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0;// 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_PrepareIndirectPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _PrepareIndirectPipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_PrepareIndirectPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _PrepareIndirectPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _PrepareIndirectPipelineLayout, nullptr);
		});
}

void Renderer::build_prepare_indirect_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _PrepareIndirectCmdBuffer;
	VkCommandBuffer& cmd = _SurfelPositionCmd;

	//VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _SurfelPositionCmd;

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _PrepareIndirectPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _PrepareIndirectPipelineLayout, 0, 1, &_PrepareIndirectDescSet, 0, nullptr);



	//shaderIf = 2;


	/*int t = static_cast<int> (time(NULL));


	vkCmdPushConstants(cmd, _PrepareIndirectPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);*/

	vkCmdDispatch(cmd, 1, 1, 1);

	//VK_CHECK(vkEndCommandBuffer(cmd));
}


//------------------------------------------------------------------- Grid reset


void Renderer::create_grid_reset_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_GridResetDescPool));

	VkDescriptorSetLayoutBinding _GridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		_GridBufferBinding
	};

	VkDescriptorSetLayoutCreateInfo gridResetDescriptorSetLayoutCreateInfo = {};
	gridResetDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	gridResetDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	gridResetDescriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &gridResetDescriptorSetLayoutCreateInfo, nullptr, &_GridResetDescSetLayout));

	VkDescriptorSetAllocateInfo gridResetDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_GridResetDescPool, &_GridResetDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &gridResetDescriptorSetAllocateInfo, &_GridResetDescSet));


	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);

	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _GridResetDescSet, &gridDescInfo, 0);


	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		GridWrite
	};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);



	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		//vkDestroyDescriptorSetLayout(*device, _buffer3DescSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _GridResetDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _GridResetDescPool, nullptr);
		});
}

void Renderer::init_grid_reset_pipeline()
{
	VkShaderModule computeShaderModule;

	VulkanEngine::engine->load_shader_module(vkutil::findFile("gridReset.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	//VkPushConstantRange _constantRangeCI = {};
	//_constantRangeCI.offset = 0;
	//_constantRangeCI.size = sizeof(int);
	//_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_GridResetDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr;// &_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0;// 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_GridResetPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _GridResetPipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_GridResetPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _GridResetPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _GridResetPipelineLayout, nullptr);
		});
}

void Renderer::build_grid_reset_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _GridResetCmdBuffer;
	VkCommandBuffer& cmd = _SurfelPositionCmd;

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _GridResetPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _GridResetPipelineLayout, 0, 1, &_GridResetDescSet, 0, nullptr);

	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc = {};
	bufferbarrierdesc.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc.pNext = nullptr;
	bufferbarrierdesc.buffer = _SurfelGridBuffer._buffer;
	bufferbarrierdesc.size = sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE;
	bufferbarrierdesc.offset = 0;
	bufferbarrierdesc.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;


	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		dstStage,
		0,
		0, nullptr,
		1, &bufferbarrierdesc,
		0, nullptr
	);

	//shaderIf = 2;


	//int t = static_cast<int> (time(NULL));


	//vkCmdPushConstants(cmd, _GridResetPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);

	vkCmdDispatch(cmd, (SURFEL_TABLE_SIZE + 63) / 64, 1, 1);

	//VK_CHECK(vkEndCommandBuffer(cmd));
}


//------------------------------------------------------------------- Surfel Update


void Renderer::create_update_surfels_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_UpdateSurfelsDescPool));

	VkDescriptorSetLayoutBinding _PositionBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	VkDescriptorSetLayoutBinding _SurfelDataBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1);
	
	VkDescriptorSetLayoutBinding statsBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2);
	//VkDescriptorSetLayoutBinding depthBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_COMPUTE_BIT, 3);
	VkDescriptorSetLayoutBinding _GridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3);
	VkDescriptorSetLayoutBinding cameraBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4);			// Camera buffer

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		_PositionBufferBinding,
		_SurfelDataBufferBinding,
		statsBinding,
		_GridBufferBinding,
		cameraBufferBinding
	};



	//transitionImageLayout(_deferredTextures[6].image._image, VulkanEngine::engine->_depthFormat, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	VkDescriptorSetLayoutCreateInfo updateDescriptorSetLayoutCreateInfo = {};
	updateDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	updateDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	updateDescriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &updateDescriptorSetLayoutCreateInfo, nullptr, &_UpdateSurfelsDescSetLayout));

	VkDescriptorSetAllocateInfo updateDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_UpdateSurfelsDescPool, &_UpdateSurfelsDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &updateDescriptorSetAllocateInfo, &_UpdateSurfelsDescSet));


	VkDescriptorBufferInfo surfelDescInfo = vkinit::descriptor_buffer_info(_SurfelBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);

	VkDescriptorBufferInfo surfelDataDescInfo = vkinit::descriptor_buffer_info(_SurfelDataBuffer._buffer, sizeof(SurfelData) * SURFEL_CAPACITY);
	
	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	//VkDescriptorImageInfo depthDescriptorDepth = vkinit::descriptor_image_info(_deferredTextures[6].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _SurfelPositionNormalSampler);

	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);


	VkDescriptorBufferInfo cameraBufferInfo = vkinit::descriptor_buffer_info(_cameraBuffer._buffer, sizeof(GPUCameraData));



	VkWriteDescriptorSet surfelBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _UpdateSurfelsDescSet, &surfelDescInfo, 0);
	VkWriteDescriptorSet surfelDataBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _UpdateSurfelsDescSet, &surfelDataDescInfo, 1);
	VkWriteDescriptorSet statsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _UpdateSurfelsDescSet, &statsDescInfo, 2);
	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _UpdateSurfelsDescSet, &gridDescInfo, 3);
	VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _UpdateSurfelsDescSet, &cameraBufferInfo, 4);


	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		surfelBufferWrite,
		surfelDataBufferWrite,
		statsWrite,
		GridWrite,
		cameraWrite
	};


	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);

	//vkUpdateDescriptorSets(*device, 1, &normalWrite, 0, VK_NULL_HANDLE);
	//vkUpdateDescriptorSets(*device, 1, &resultWrite, 0, VK_NULL_HANDLE);


	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		//vkDestroyDescriptorSetLayout(*device, _buffer3DescSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(*device, _UpdateSurfelsDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _UpdateSurfelsDescPool, nullptr);
		});
}

void Renderer::init_update_surfels_pipeline()
{
	VkShaderModule computeShaderModule;

	VulkanEngine::engine->load_shader_module(vkutil::findFile("updateSurfels.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	//VkPushConstantRange _constantRangeCI = {};
	//_constantRangeCI.offset = 0;
	//_constantRangeCI.size = sizeof(int);
	//_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_UpdateSurfelsDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr;//&_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0;// 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_UpdateSurfelsPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _UpdateSurfelsPipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_UpdateSurfelsPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _UpdateSurfelsPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _UpdateSurfelsPipelineLayout, nullptr);
		});
}

void Renderer::build_update_surfels_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _UpdateSurfelsCmdBuffer;
	VkCommandBuffer& cmd = _SurfelPositionCmd;

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _UpdateSurfelsPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _UpdateSurfelsPipelineLayout, 0, 1, &_UpdateSurfelsDescSet, 0, nullptr);


	//shaderIf = 2;


	int t = static_cast<int> (time(NULL));

	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc1 = {};
	bufferbarrierdesc1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc1.pNext = nullptr;
	bufferbarrierdesc1.buffer = _SurfelBuffer._buffer;
	bufferbarrierdesc1.size = sizeof(Surfel) * SURFEL_CAPACITY;
	bufferbarrierdesc1.offset = 0;
	bufferbarrierdesc1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;


	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		dstStage,
		0,
		0, nullptr,
		1, &bufferbarrierdesc1,
		0, nullptr
	);



	//vkCmdPushConstants(cmd, _UpdateSurfelsPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);

	vkCmdDispatchIndirect(cmd, _SurfelStatsBuffer._buffer, sizeof(unsigned int) * 2);

	VkMemoryBarrier memorybarrierdesc = {};
	memorybarrierdesc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memorybarrierdesc.pNext = nullptr;
	memorybarrierdesc.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	memorybarrierdesc.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc2 = {};
	bufferbarrierdesc2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc2.pNext = nullptr;
	bufferbarrierdesc2.buffer = _SurfelBuffer._buffer;
	bufferbarrierdesc2.size = sizeof(Surfel) * SURFEL_CAPACITY;
	bufferbarrierdesc2.offset = 0;
	bufferbarrierdesc2.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; // 

	bufferbarrierdesc2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		dstStage,
		0,
		1, &memorybarrierdesc,
		1, &bufferbarrierdesc2,
		0, nullptr
	);

	VK_CHECK(vkEndCommandBuffer(cmd));

	//VK_CHECK(vkEndCommandBuffer(cmd));
}


//------------------------------------------------------------------- Grid Offset


void Renderer::create_grid_offset_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_GridOffsetDescPool));


	VkDescriptorSetLayoutBinding statsBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	VkDescriptorSetLayoutBinding _GridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1);
	VkDescriptorSetLayoutBinding _CellBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2);

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		statsBinding,
		_GridBufferBinding,
		_CellBufferBinding
	};


	VkDescriptorSetLayoutCreateInfo gridOffsetSetLayoutCreateInfo = {};
	gridOffsetSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	gridOffsetSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	gridOffsetSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &gridOffsetSetLayoutCreateInfo, nullptr, &_GridOffsetDescSetLayout));

	VkDescriptorSetAllocateInfo gridOffsetDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_GridOffsetDescPool, &_GridOffsetDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &gridOffsetDescriptorSetAllocateInfo, &_GridOffsetDescSet));


	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);

	VkDescriptorBufferInfo cellDescInfo = vkinit::descriptor_buffer_info(_SurfelCellBuffer._buffer, sizeof(unsigned int) * SURFEL_CAPACITY * 27);


	VkWriteDescriptorSet statsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _GridOffsetDescSet, &statsDescInfo, 0);
	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _GridOffsetDescSet, &gridDescInfo, 1);
	VkWriteDescriptorSet CellWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _GridOffsetDescSet, &cellDescInfo, 2);


	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		statsWrite,
		GridWrite,
		CellWrite
	};


	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);


	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _GridOffsetDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _GridOffsetDescPool, nullptr);
		});
}

void Renderer::init_grid_offset_pipeline()
{
	VkShaderModule computeShaderModule;

	VulkanEngine::engine->load_shader_module(vkutil::findFile("gridOffset.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	//VkPushConstantRange _constantRangeCI = {};
	//_constantRangeCI.offset = 0;
	//_constantRangeCI.size = sizeof(int);
	//_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_GridOffsetDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr; // &_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0;// 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_GridOffsetPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _GridOffsetPipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_GridOffsetPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _GridOffsetPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _GridOffsetPipelineLayout, nullptr);
		});
}

void Renderer::build_grid_offset_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _GridOffsetCmdBuffer;
	VkCommandBuffer& cmd = _SurfelPositionCmd;

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _GridOffsetPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _GridOffsetPipelineLayout, 0, 1, &_GridOffsetDescSet, 0, nullptr);


	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc1 = {};
	bufferbarrierdesc1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc1.pNext = nullptr;
	bufferbarrierdesc1.buffer = _SurfelCellBuffer._buffer;
	bufferbarrierdesc1.size = sizeof(unsigned int) * SURFEL_CAPACITY * 27;
	bufferbarrierdesc1.offset = 0;
	bufferbarrierdesc1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	VkBufferMemoryBarrier bufferbarrierdesc2 = {};
	bufferbarrierdesc2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc2.pNext = nullptr;
	bufferbarrierdesc2.buffer = _SurfelStatsBuffer._buffer;
	bufferbarrierdesc2.size = sizeof(unsigned int) * 8;
	bufferbarrierdesc2.offset = 0;
	bufferbarrierdesc2.srcAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;


	std::vector<VkBufferMemoryBarrier> bufferBarriers =
	{
		bufferbarrierdesc1,
		bufferbarrierdesc2
	};


	vkCmdPipelineBarrier(
		cmd,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		dstStage,
		0,
		0, nullptr,
		bufferBarriers.size(), bufferBarriers.data(),
		0, nullptr
	);




	/*int t = static_cast<int> (time(NULL));

	vkCmdPushConstants(cmd, _GridOffsetPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);*/

	vkCmdDispatch(cmd, (SURFEL_TABLE_SIZE + 63) / 64, 1, 1);


	VkMemoryBarrier memorybarrierdesc = {};
	memorybarrierdesc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memorybarrierdesc.pNext = nullptr;
	memorybarrierdesc.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	memorybarrierdesc.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc = {};
	bufferbarrierdesc.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc.pNext = nullptr;
	bufferbarrierdesc.buffer = _SurfelStatsBuffer._buffer;
	bufferbarrierdesc.size = sizeof(unsigned int) * 8;
	bufferbarrierdesc.offset = 0;
	bufferbarrierdesc.srcAccessMask =  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		0,
		1, &memorybarrierdesc,
		1, &bufferbarrierdesc,
		0, nullptr
	);

	//VK_CHECK(vkEndCommandBuffer(cmd));
}


//------------------------------------------------------------------- Surfel Binning


void Renderer::create_surfel_binning_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_SurfelBinningDescPool));

	VkDescriptorSetLayoutBinding _SurfelBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	VkDescriptorSetLayoutBinding statsBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1);
	VkDescriptorSetLayoutBinding _GridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2);
	VkDescriptorSetLayoutBinding _CellBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3);
	VkDescriptorSetLayoutBinding cameraBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4);			// Camera buffer

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		_SurfelBufferBinding,
		statsBinding,
		_GridBufferBinding,
		_CellBufferBinding,
		cameraBufferBinding
	};


	VkDescriptorSetLayoutCreateInfo surfelBinningDescriptorSetLayoutCreateInfo = {};
	surfelBinningDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	surfelBinningDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	surfelBinningDescriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &surfelBinningDescriptorSetLayoutCreateInfo, nullptr, &_SurfelBinningDescSetLayout));

	VkDescriptorSetAllocateInfo surfelBinningDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_SurfelBinningDescPool, &_SurfelBinningDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &surfelBinningDescriptorSetAllocateInfo, &_SurfelBinningDescSet));


	VkDescriptorBufferInfo surfelDescInfo = vkinit::descriptor_buffer_info(_SurfelBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);

	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);

	VkDescriptorBufferInfo cellDescInfo = vkinit::descriptor_buffer_info(_SurfelCellBuffer._buffer, sizeof(unsigned int) * SURFEL_CAPACITY * 27);

	VkDescriptorBufferInfo cameraBufferInfo = vkinit::descriptor_buffer_info(_cameraBuffer._buffer, sizeof(GPUCameraData));



	VkWriteDescriptorSet surfelBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelBinningDescSet, &surfelDescInfo, 0);
	VkWriteDescriptorSet statsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelBinningDescSet, &statsDescInfo, 1);
	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelBinningDescSet, &gridDescInfo, 2);
	VkWriteDescriptorSet cellWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelBinningDescSet, &cellDescInfo, 3);
	VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _SurfelBinningDescSet, &cameraBufferInfo, 4);


	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		surfelBufferWrite,
		statsWrite,
		GridWrite,
		cellWrite,
		cameraWrite
	};


	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);


	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _SurfelBinningDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _SurfelBinningDescPool, nullptr);
		});
}

void Renderer::init_surfel_binning_pipeline()
{
	VkShaderModule computeShaderModule;

	VulkanEngine::engine->load_shader_module(vkutil::findFile("surfelbinning.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);

	//VkPushConstantRange _constantRangeCI = {};
	//_constantRangeCI.offset = 0;
	//_constantRangeCI.size = sizeof(int);
	//_constantRangeCI.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_SurfelBinningDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr;// &_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0;// 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_SurfelBinningPipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _SurfelBinningPipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_SurfelBinningPipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _SurfelBinningPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _SurfelBinningPipelineLayout, nullptr);
		});
}

void Renderer::build_surfel_binning_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	//VkCommandBuffer& cmd = _SurfelBinningCmdBuffer;
	VkCommandBuffer& cmd = _SurfelPositionCmd;

	//VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _SurfelBinningPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _SurfelBinningPipelineLayout, 0, 1, &_SurfelBinningDescSet, 0, nullptr);


	int t = static_cast<int> (time(NULL));


	//vkCmdPushConstants(cmd, _GridResetPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);

	vkCmdDispatchIndirect(cmd, _SurfelStatsBuffer._buffer, sizeof(unsigned int) * 2);

	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc1 = {};
	bufferbarrierdesc1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc1.pNext = nullptr;
	bufferbarrierdesc1.buffer = _SurfelCellBuffer._buffer;
	bufferbarrierdesc1.size = sizeof(unsigned int) * SURFEL_CAPACITY * 27;
	bufferbarrierdesc1.offset = 0;
	bufferbarrierdesc1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	VkBufferMemoryBarrier bufferbarrierdesc2 = {};
	bufferbarrierdesc2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc2.pNext = nullptr;
	bufferbarrierdesc2.buffer = _SurfelGridBuffer._buffer;
	bufferbarrierdesc2.size = sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE;
	bufferbarrierdesc2.offset = 0;
	bufferbarrierdesc2.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;


	std::vector<VkBufferMemoryBarrier> bufferBarriers =
	{
		bufferbarrierdesc1,
		bufferbarrierdesc2
	};


	VkMemoryBarrier memorybarrierdesc = {};
	memorybarrierdesc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memorybarrierdesc.pNext = nullptr;
	memorybarrierdesc.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	memorybarrierdesc.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		dstStage,
		0,
		1, &memorybarrierdesc,
		bufferBarriers.size(), bufferBarriers.data(),
		0, nullptr
	);



	VK_CHECK(vkEndCommandBuffer(cmd));
}


void Renderer::create_surfel_rtx_descriptors()
{

	std::vector<VkDescriptorPoolSize> poolSizes = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSizes, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_SurfelRTXDescPool));

	const uint32_t nInstances = static_cast<uint32_t>(_scene->_entities.size());
	const uint32_t nDrawables = static_cast<uint32_t>(_scene->get_drawable_nodes_size());
	const uint32_t nMaterials = static_cast<uint32_t>(Material::_materials.size());
	const uint32_t nTextures = static_cast<uint32_t>(Texture::_textures.size());
	const uint32_t nLights = static_cast<uint32_t>(_scene->_lights.size());

	// binding = 0 TLAS
	// binding = 1 Storage image
	// binding = 2 Camera buffer
	// binding = 3 Gbuffers
	// binding = 4 Lights buffer
	// binding = 5 Vertices buffer
	// binding = 6 Indices buffer
	// binding = 7 Textures buffer
	// binding = 8 Skybox buffer
	// binding = 9 Materials buffer
	// binding = 10 Scene indices
	// binding = 11 Matrices buffer
	// binding = 12 Shadow image

	VkDescriptorSetLayoutBinding TLASBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);			// TLAS
	VkDescriptorSetLayoutBinding cameraBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 2);			// Camera buffer
	VkDescriptorSetLayoutBinding lightsBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 4);	// Lights
	VkDescriptorSetLayoutBinding vertexBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 5, nInstances);	// Vertices
	VkDescriptorSetLayoutBinding indexBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 6, nInstances);	// Indices
	VkDescriptorSetLayoutBinding texturesBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 7, nTextures); // Textures buffer
	VkDescriptorSetLayoutBinding matIdxBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 8); // Scene indices
	VkDescriptorSetLayoutBinding materialBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 9);	// Materials buffer
	VkDescriptorSetLayoutBinding skyboxBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 10, 2);
	VkDescriptorSetLayoutBinding matrixBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 11);	// Matrices
	VkDescriptorSetLayoutBinding surfelBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 13);	// surfels
	VkDescriptorSetLayoutBinding surfelStatsBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 14);	// stats
	VkDescriptorSetLayoutBinding surfelGridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 15);	// grid
	VkDescriptorSetLayoutBinding surfelCellBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 16);	// cell
	VkDescriptorSetLayoutBinding surfelDataBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 17);	// data

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		TLASBinding,
		cameraBufferBinding,
		lightsBufferBinding,
		vertexBufferBinding,
		indexBufferBinding,
		texturesBufferBinding,
		matrixBufferBinding,
		materialBufferBinding,
		matIdxBufferBinding,
		skyboxBufferBinding,
		surfelBufferBinding,
		surfelStatsBufferBinding,
		surfelGridBufferBinding,
		surfelCellBufferBinding,
		surfelDataBufferBinding
	};

	VkDescriptorSetLayoutCreateInfo setInfo = vkinit::descriptor_set_layout_create_info(static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings);
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_SurfelRTXDescSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptor_set_allocate_info(_SurfelRTXDescPool, &_SurfelRTXDescSetLayout);
	vkAllocateDescriptorSets(*device, &allocInfo, &_SurfelRTXDescSet);

	// Binding = 0 TLAS write
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
	descriptorAccelerationStructureInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures = &_topLevelAS.handle;

	// Binding = 1 Camera write
	VkDescriptorBufferInfo cameraBufferInfo = vkinit::descriptor_buffer_info(_rtCameraBuffer._buffer, sizeof(RTCameraData));

	// Binding = 4 Lights buffer descriptor
	VkDescriptorBufferInfo lightDescBuffer = vkinit::descriptor_buffer_info(_lightBuffer._buffer, sizeof(uboLight) * nLights);

	std::vector<VkDescriptorBufferInfo> vertexDescInfo;
	std::vector<VkDescriptorBufferInfo> indexDescInfo;
	std::vector<glm::vec4> idVector;
	for (Object* obj : _scene->_entities)
	{
		AllocatedBuffer vBuffer;
		size_t bufferSize = sizeof(rtVertexAttribute) * obj->prefab->_mesh->_vertices.size();
		VulkanEngine::engine->create_buffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, vBuffer);

		std::vector<rtVertexAttribute> vAttr;
		std::vector<Vertex> vertices = obj->prefab->_mesh->_vertices;
		vAttr.reserve(vertices.size());
		for (Vertex& v : vertices) {
			vAttr.push_back({ {v.normal.x, v.normal.y, v.normal.z, 1}, {v.color.x, v.color.y, v.color.z, 1}, {v.uv.x, v.uv.y, 1, 1} });
		}

		void* vdata;
		vmaMapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation, &vdata);
		memcpy(vdata, vAttr.data(), bufferSize);
		vmaUnmapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation);

		// Binding = 5 Vertices Info
		VkDescriptorBufferInfo vertexBufferDescriptor = vkinit::descriptor_buffer_info(vBuffer._buffer, bufferSize);
		vertexDescInfo.push_back(vertexBufferDescriptor);

		// Binding = 6 Indices Info
		VkDescriptorBufferInfo indexBufferDescriptor = vkinit::descriptor_buffer_info(obj->prefab->_mesh->_indexBuffer._buffer, sizeof(uint32_t) * obj->prefab->_mesh->_indices.size());
		indexDescInfo.push_back(indexBufferDescriptor);

		for (Node* root : obj->prefab->_root)
		{
			root->fill_index_buffer(idVector);
		}
	}

	// Binding = 7 Textures info
	VkDescriptorSetAllocateInfo textureAllocInfo = {};
	textureAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	textureAllocInfo.pNext = nullptr;
	textureAllocInfo.descriptorSetCount = 1;
	textureAllocInfo.pSetLayouts = &_textureDescriptorSetLayout;
	textureAllocInfo.descriptorPool = _SurfelRTXDescPool;

	VK_CHECK(vkAllocateDescriptorSets(*device, &textureAllocInfo, &_textureDescriptorSet));

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSampler sampler;
	vkCreateSampler(*device, &samplerInfo, nullptr, &sampler);

	std::vector<VkDescriptorImageInfo> imageInfos;
	for (auto const& texture : Texture::_textures)
	{
		VkDescriptorImageInfo imageBufferInfo = vkinit::descriptor_image_info(texture.second->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sampler);
		imageInfos.push_back(imageBufferInfo);
	}

	// Binding = 8 Skybox
	VkDescriptorImageInfo skyboxImagesDesc[2];
	skyboxImagesDesc[0] = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_8k.jpg")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	skyboxImagesDesc[1] = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_Env.hdr")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	// Binding = 9 Material info
	VkDescriptorBufferInfo materialBufferInfo = vkinit::descriptor_buffer_info(_matBuffer._buffer, sizeof(GPUMaterial) * nMaterials);

	// Binding = 10 ID info
	if (!_idBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(glm::vec4) * idVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _idBuffer);

	void* idData;
	vmaMapMemory(VulkanEngine::engine->_allocator, _idBuffer._allocation, &idData);
	memcpy(idData, idVector.data(), sizeof(glm::vec4) * idVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _idBuffer._allocation);



	VkDescriptorBufferInfo idDescInfo = vkinit::descriptor_buffer_info(_idBuffer._buffer, sizeof(glm::vec4) * idVector.size());

	// Binding = 11 Matrices info
	VkDescriptorBufferInfo matrixDescInfo = vkinit::descriptor_buffer_info(_matricesBuffer._buffer, sizeof(glm::mat4) * _scene->_matricesVector.size());


	VkDescriptorBufferInfo surfelDescInfo = vkinit::descriptor_buffer_info(_SurfelBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);

	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);

	VkDescriptorBufferInfo cellDescInfo = vkinit::descriptor_buffer_info(_SurfelCellBuffer._buffer, sizeof(unsigned int) * SURFEL_CAPACITY * 27);

	VkDescriptorBufferInfo surfelDataDescInfo = vkinit::descriptor_buffer_info(_SurfelDataBuffer._buffer, sizeof(SurfelData) * SURFEL_CAPACITY);


	// Writes list
	VkWriteDescriptorSet accelerationStructureWrite = vkinit::write_descriptor_acceleration_structure(_SurfelRTXDescSet, &descriptorAccelerationStructureInfo, 0);
	VkWriteDescriptorSet cameraWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _SurfelRTXDescSet, &cameraBufferInfo, 2);
	VkWriteDescriptorSet lightWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &lightDescBuffer, 4);
	VkWriteDescriptorSet vertexBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, vertexDescInfo.data(), 5, nInstances);
	VkWriteDescriptorSet indexBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, indexDescInfo.data(), 6, nInstances);
	VkWriteDescriptorSet texturesBufferWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _SurfelRTXDescSet, imageInfos.data(), 7, nTextures);
	VkWriteDescriptorSet matIdxBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &idDescInfo, 8);
	VkWriteDescriptorSet materialBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &materialBufferInfo, 9);
	VkWriteDescriptorSet skyboxBufferWrite = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _SurfelRTXDescSet, skyboxImagesDesc, 10, 2);
	VkWriteDescriptorSet matrixBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &matrixDescInfo, 11);
	VkWriteDescriptorSet surfelBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &surfelDescInfo, 13);
	VkWriteDescriptorSet statsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &statsDescInfo, 14);
	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &gridDescInfo, 15);
	VkWriteDescriptorSet cellWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &cellDescInfo, 16);
	VkWriteDescriptorSet surfelDataBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelRTXDescSet, &surfelDataDescInfo, 17);

	std::vector<VkWriteDescriptorSet> writes = {
		accelerationStructureWrite,	// 0 TLAS
		cameraWrite,
		lightWrite,
		vertexBufferWrite,
		indexBufferWrite,
		texturesBufferWrite,
		matrixBufferWrite, //este
		materialBufferWrite,
		matIdxBufferWrite, //este
		skyboxBufferWrite,
		surfelBufferWrite,
		statsWrite,
		GridWrite,
		cellWrite,
		surfelDataBufferWrite
		};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _SurfelRTXDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _SurfelRTXDescPool, nullptr);
		vkDestroySampler(*device, sampler, nullptr);
		});
}

void Renderer::create_surfel_rtx_pipeline()
{
	VulkanEngine* engine = VulkanEngine::engine;

	// Setup ray tracing shader groups

	std::vector<VkPipelineShaderStageCreateInfo> SurfelShaderStages = {};


	// Ray generation group
	VkShaderModule hraygenModule;
	{
		SurfelShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("surfelRayGen.rgen.spv", searchPaths, true).c_str(), &hraygenModule, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		shaderGroup.generalShader = 0;
		shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
		shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

		surfelShaderGroups.push_back(shaderGroup);
	}

	// Miss group
	VkShaderModule hmissModule;
	{
		SurfelShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("surfelMiss.rmiss.spv", searchPaths, true).c_str(), &hmissModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		shaderGroup.generalShader = static_cast<uint32_t>(SurfelShaderStages.size()) - 1;
		shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
		shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

		shaderGroup.generalShader = static_cast<uint32_t>(SurfelShaderStages.size()) - 1;
		surfelShaderGroups.push_back(shaderGroup);
	}


	// Hit group
	VkShaderModule hhitModule;
	{
		SurfelShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("surfelHit.rchit.spv", searchPaths, true).c_str(), &hhitModule, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
		shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
		shaderGroup.closestHitShader = static_cast<uint32_t>(SurfelShaderStages.size()) - 1;
		shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
		surfelShaderGroups.push_back(shaderGroup);
	}


	// HYBRID PIPELINE CREATION - using the deferred pass
	VkPipelineLayoutCreateInfo SurfelPipelineLayoutInfo{};
	SurfelPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	SurfelPipelineLayoutInfo.setLayoutCount = 1;
	SurfelPipelineLayoutInfo.pSetLayouts = &_SurfelRTXDescSetLayout;
	VK_CHECK(vkCreatePipelineLayout(*device, &SurfelPipelineLayoutInfo, nullptr, &_SurfelRTXPipelineLayout));

	VkRayTracingPipelineCreateInfoKHR SurfelPipelineInfo{};
	SurfelPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	SurfelPipelineInfo.stageCount = static_cast<uint32_t>(SurfelShaderStages.size());
	SurfelPipelineInfo.pStages = SurfelShaderStages.data();
	SurfelPipelineInfo.groupCount = static_cast<uint32_t>(surfelShaderGroups.size());
	SurfelPipelineInfo.pGroups = surfelShaderGroups.data();
	SurfelPipelineInfo.maxPipelineRayRecursionDepth = 1;
	SurfelPipelineInfo.layout = _SurfelRTXPipelineLayout;

	VK_CHECK(vkCreateRayTracingPipelinesKHR(*device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &SurfelPipelineInfo, nullptr, &_SurfelRTXPipeline));

	


	vkDestroyShaderModule(*device, hraygenModule, nullptr);
	vkDestroyShaderModule(*device, hmissModule, nullptr);
	vkDestroyShaderModule(*device, hhitModule, nullptr);


	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {

		vkDestroyPipeline(*device, _SurfelRTXPipeline, nullptr);

		vkDestroyPipelineLayout(*device, _SurfelRTXPipelineLayout, nullptr);

		});
}

void Renderer::create_surfel_rtx_SBT()
{
	const uint32_t handleSize = VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	const uint32_t handleSizeAligned = alignedSize(VulkanEngine::engine->_rtProperties.shaderGroupHandleSize, VulkanEngine::engine->_rtProperties.shaderGroupHandleAlignment);

	const uint32_t groupCount = static_cast<uint32_t>(surfelShaderGroups.size());
	const uint32_t sbtSize = groupCount * handleSizeAligned;



	std::vector<uint8_t> shaderHandleStorage(sbtSize);
	VK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(*device, _SurfelRTXPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data()));

	const VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, _SurfelRTXraygenSBT);
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, _SurfelRTXmissSBT);
	//VulkanEngine::engine->create_buffer(handleSize * 2, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, _SurfelRTXmissSBT);
	VulkanEngine::engine->create_buffer(handleSize, bufferUsageFlags, VMA_MEMORY_USAGE_CPU_TO_GPU, _SurfelRTXhitSBT);

	void* rayGenData, * missData, * hitData;

	vmaMapMemory(VulkanEngine::engine->_allocator, _SurfelRTXraygenSBT._allocation, &rayGenData);
	memcpy(rayGenData, shaderHandleStorage.data(), handleSize);
	vmaMapMemory(VulkanEngine::engine->_allocator, _SurfelRTXmissSBT._allocation, &missData);
	memcpy(missData, shaderHandleStorage.data() + handleSizeAligned, handleSize);
	vmaMapMemory(VulkanEngine::engine->_allocator, _SurfelRTXhitSBT._allocation, &hitData);
	memcpy(hitData, shaderHandleStorage.data() + handleSizeAligned * 2, handleSize);

	vmaUnmapMemory(VulkanEngine::engine->_allocator, _SurfelRTXraygenSBT._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _SurfelRTXmissSBT._allocation);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _SurfelRTXhitSBT._allocation);


}

void Renderer::create_surfel_rtx_cmd_buffer()
{
		VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
	
		VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	
		VkCommandBuffer& cmd = _SurfelRTXCommandBuffer;
		//VkCommandBuffer& cmd = _SurfelPositionCmd;
	
		VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBufInfo));
	
		VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
		bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		bufferDeviceAddressInfo.buffer = _SurfelRTXraygenSBT._buffer;


		VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
		//raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(_SurfelRTXraygenSBT._buffer);
		raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);
		raygenShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
		raygenShaderSbtEntry.size			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	

		bufferDeviceAddressInfo.buffer = _SurfelRTXmissSBT._buffer;

		VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
		//missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(_SurfelRTXmissSBT._buffer);
		missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);
		missShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
		//missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize * 2;
		missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	
		bufferDeviceAddressInfo.buffer = _SurfelRTXhitSBT._buffer;

		VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
		//hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(_SurfelRTXhitSBT._buffer);
		hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->vkGetBufferDeviceAddressKHR(*device, &bufferDeviceAddressInfo);
		hitShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
		hitShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	
		VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};
	
		uint32_t width = VulkanEngine::engine->_window->getWidth(), height = VulkanEngine::engine->_window->getHeight();
	


		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _SurfelRTXPipeline);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _SurfelRTXPipelineLayout, 0, 1, &_SurfelRTXDescSet, 0, nullptr);
	
		vkCmdTraceRaysKHR(
			cmd,
			&raygenShaderSbtEntry,
			&missShaderSbtEntry,
			&hitShaderSbtEntry,
			&callableShaderSbtEntry,
			SURFEL_CAPACITY,
			1,
			1
		);
	
		VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
		VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

		VkMemoryBarrier memorybarrierdesc = {};
		memorybarrierdesc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		memorybarrierdesc.pNext = nullptr;
		memorybarrierdesc.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
		memorybarrierdesc.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

		vkCmdPipelineBarrier(
			cmd,
			srcStage,
			dstStage,
			0,
			1, &memorybarrierdesc,
			0, nullptr,
			0, nullptr
		);


		VK_CHECK(vkEndCommandBuffer(cmd));
	}

void Renderer::create_surfel_shade_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}
	};

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = vkinit::descriptor_pool_create_info(poolSize, 2);
	VK_CHECK(vkCreateDescriptorPool(*device, &descriptorPoolCreateInfo, nullptr, &_SurfelShadeDescPool));

	VkDescriptorSetLayoutBinding _SurfelBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0);
	VkDescriptorSetLayoutBinding statsBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1);
	VkDescriptorSetLayoutBinding _GridBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2);
	VkDescriptorSetLayoutBinding _CellBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 3);
	VkDescriptorSetLayoutBinding _DataBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 4);


	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		_SurfelBufferBinding,
		statsBinding,
		_GridBufferBinding,
		_CellBufferBinding,
		_DataBufferBinding
	};


	VkDescriptorSetLayoutCreateInfo surfelShadeDescriptorSetLayoutCreateInfo = {};
	surfelShadeDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	surfelShadeDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	surfelShadeDescriptorSetLayoutCreateInfo.pBindings = setLayoutBindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &surfelShadeDescriptorSetLayoutCreateInfo, nullptr, &_SurfelShadeDescSetLayout));

	VkDescriptorSetAllocateInfo surfelShadeDescriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_SurfelShadeDescPool, &_SurfelShadeDescSetLayout, 1);
	VK_CHECK(vkAllocateDescriptorSets(*device, &surfelShadeDescriptorSetAllocateInfo, &_SurfelShadeDescSet));


	VkDescriptorBufferInfo surfelDescInfo = vkinit::descriptor_buffer_info(_SurfelBuffer._buffer, sizeof(Surfel) * SURFEL_CAPACITY);

	VkDescriptorBufferInfo statsDescInfo = vkinit::descriptor_buffer_info(_SurfelStatsBuffer._buffer, sizeof(unsigned int) * 8);

	VkDescriptorBufferInfo gridDescInfo = vkinit::descriptor_buffer_info(_SurfelGridBuffer._buffer, sizeof(SurfelGridCell) * SURFEL_TABLE_SIZE);

	VkDescriptorBufferInfo cellDescInfo = vkinit::descriptor_buffer_info(_SurfelCellBuffer._buffer, sizeof(unsigned int) * SURFEL_CAPACITY * 27);

	VkDescriptorBufferInfo dataBufferInfo = vkinit::descriptor_buffer_info(_SurfelDataBuffer._buffer, sizeof(SurfelData) * SURFEL_CAPACITY);



	VkWriteDescriptorSet surfelBufferWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelShadeDescSet, &surfelDescInfo, 0);
	VkWriteDescriptorSet statsWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelShadeDescSet, &statsDescInfo, 1);
	VkWriteDescriptorSet GridWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelShadeDescSet, &gridDescInfo, 2);
	VkWriteDescriptorSet cellWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelShadeDescSet, &cellDescInfo, 3);
	VkWriteDescriptorSet dataWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _SurfelShadeDescSet, &dataBufferInfo, 4);


	std::vector<VkWriteDescriptorSet> DescriptorWrites =
	{
		surfelBufferWrite,
		statsWrite,
		GridWrite,
		cellWrite,
		dataWrite
	};


	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(DescriptorWrites.size()), DescriptorWrites.data(), 0, VK_NULL_HANDLE);


	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _SurfelShadeDescSetLayout, nullptr);
		vkDestroyDescriptorPool(*device, _SurfelShadeDescPool, nullptr);
		});
}

void Renderer::init_surfel_shade_pipeline()
{
	VkShaderModule computeShaderModule;

	VulkanEngine::engine->load_shader_module(vkutil::findFile("surfelshade.comp.spv", searchPaths, true).c_str(), &computeShaderModule);

	VkPipelineShaderStageCreateInfo shaderStageCI = vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_COMPUTE_BIT, computeShaderModule);


	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &_SurfelShadeDescSetLayout;
	pipelineLayoutCI.pPushConstantRanges = nullptr;// &_constantRangeCI;
	pipelineLayoutCI.pushConstantRangeCount = 0;// 1;
	VK_CHECK(vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_SurfelShadePipelineLayout));

	VkComputePipelineCreateInfo computePipelineCI = {};
	computePipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	computePipelineCI.stage = shaderStageCI;
	computePipelineCI.layout = _SurfelShadePipelineLayout;

	VK_CHECK(vkCreateComputePipelines(*device, VK_NULL_HANDLE, 1, &computePipelineCI, nullptr, &_SurfelShadePipeline));

	// Fill the buffer

	vkDestroyShaderModule(*device, computeShaderModule, nullptr);
	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _SurfelShadePipeline, nullptr);
		vkDestroyPipelineLayout(*device, _SurfelShadePipelineLayout, nullptr);
		});
}

void Renderer::build_surfel_shade_buffer()
{
	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	VkCommandBuffer& cmd = _SurfelShadeCmdBuffer;

	VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _SurfelShadePipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _SurfelShadePipelineLayout, 0, 1, &_SurfelShadeDescSet, 0, nullptr);


	int t = static_cast<int> (time(NULL));


	//vkCmdPushConstants(cmd, _GridResetPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &t);

	vkCmdDispatchIndirect(cmd, _SurfelStatsBuffer._buffer, sizeof(unsigned int) * 2);

	VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

	VkBufferMemoryBarrier bufferbarrierdesc1 = {};
	bufferbarrierdesc1.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	bufferbarrierdesc1.pNext = nullptr;
	bufferbarrierdesc1.buffer = _SurfelDataBuffer._buffer;
	bufferbarrierdesc1.size = sizeof(SurfelData) * SURFEL_CAPACITY;
	bufferbarrierdesc1.offset = 0;
	bufferbarrierdesc1.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; // VK_ACCESS_INDIRECT_COMMAND_READ_BIT

	bufferbarrierdesc1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; //RESOURCE_STATE_UNORDERED_ACCESS
																				//flags |= VK_ACCESS_SHADER_READ_BIT;
																				//flags |= VK_ACCESS_SHADER_WRITE_BIT;
	bufferbarrierdesc1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bufferbarrierdesc1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;


	VkMemoryBarrier memorybarrierdesc = {};
	memorybarrierdesc.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	memorybarrierdesc.pNext = nullptr;
	memorybarrierdesc.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
	memorybarrierdesc.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

	vkCmdPipelineBarrier(
		cmd,
		srcStage,
		dstStage,
		0,
		1, &memorybarrierdesc,
		1, &bufferbarrierdesc1,
		0, nullptr
	);



	VK_CHECK(vkEndCommandBuffer(cmd));
}

// POST
// -------------------------------------------------------

void Renderer::create_post_renderPass()
{
	if (_postRenderPass)
		vkDestroyRenderPass(*device, _postRenderPass, nullptr);

	std::array<VkAttachmentDescription, 2> attachments;
	attachments[0].format			= VulkanEngine::engine->_swapchainImageFormat;
	attachments[0].samples			= VK_SAMPLE_COUNT_1_BIT;
	attachments[0].flags			= 0;
	attachments[0].loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout		= VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	attachments[1].format			= VulkanEngine::engine->_depthFormat;
	attachments[1].samples			= VK_SAMPLE_COUNT_1_BIT;
	attachments[1].flags			= 0;
	attachments[1].loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	const VkAttachmentReference colorReference{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
	const VkAttachmentReference depthReference{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

	VkSubpassDependency dependency{};
	dependency.srcSubpass				= VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass				= 0;
	dependency.srcStageMask				= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependency.dstStageMask				= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask			= VK_ACCESS_MEMORY_READ_BIT;
	dependency.dstAccessMask			= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependency.dependencyFlags			= VK_DEPENDENCY_BY_REGION_BIT;

	VkSubpassDescription subpassDesc{};
	subpassDesc.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDesc.colorAttachmentCount	= 1;
	subpassDesc.pColorAttachments		= &colorReference;
	subpassDesc.pDepthStencilAttachment = &depthReference;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType				= VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount		= static_cast<uint32_t>(attachments.size());
	renderPassInfo.pAttachments			= attachments.data();
	renderPassInfo.subpassCount			= 1;
	renderPassInfo.pSubpasses			= &subpassDesc;
	renderPassInfo.dependencyCount		= 1;
	renderPassInfo.pDependencies		= &dependency;

	VK_CHECK(vkCreateRenderPass(*device, &renderPassInfo, nullptr, &_postRenderPass));

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyRenderPass(*device, _postRenderPass, nullptr);
		});
}

void Renderer::create_post_framebuffers()
{
	VkExtent2D extent = { (uint32_t)VulkanEngine::engine->_window->getWidth(), (uint32_t)VulkanEngine::engine->_window->getHeight() };
	VkFramebufferCreateInfo framebufferInfo = vkinit::framebuffer_create_info(_renderPass, extent);

	// Grab how many images we have in the swapchain
	const uint32_t swapchain_imagecount = static_cast<uint32_t>(VulkanEngine::engine->_swapchainImages.size());
	_postFramebuffers = std::vector<VkFramebuffer>(swapchain_imagecount);

	for (unsigned int i = 0; i < swapchain_imagecount; i++)
	{
		VkImageView attachments[2];
		attachments[0] = VulkanEngine::engine->_swapchainImageViews[i];
		attachments[1] = VulkanEngine::engine->_depthImageView;

		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments	= attachments;
		VK_CHECK(vkCreateFramebuffer(*device, &framebufferInfo, nullptr, &_postFramebuffers[i]));

		VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
			vkDestroyFramebuffer(*device, _postFramebuffers[i], nullptr);
			});
	}

}

void Renderer::create_post_pipeline()
{
	// First of all load the shader modules and store them in the builder
	VkShaderModule postVertexShader, postFragmentShader;
	if (!VulkanEngine::engine->load_shader_module(vkutil::findFile("postVertex.vert.spv", searchPaths, true).c_str(), &postVertexShader)) {
		std::cout << "Post vertex failed to load" << std::endl;
	}
	if(!VulkanEngine::engine->load_shader_module(vkutil::findFile("postFragment.frag.spv", searchPaths, true).c_str(), &postFragmentShader)){
		std::cout << "Post fragment failed to load" << std::endl;
	}

	PipelineBuilder builder;
	builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, postVertexShader));
	builder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, postFragmentShader));

	// Create pipeline layout
	VkPipelineLayoutCreateInfo pipelineLayoutCI = vkinit::pipeline_layout_create_info();
	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts	= &_postDescSetLayout;

	VK_CHECK( vkCreatePipelineLayout(*device, &pipelineLayoutCI, nullptr, &_postPipelineLayout));

	builder._pipelineLayout = _postPipelineLayout;

	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	builder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	builder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	builder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	builder._vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();
	builder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();

	VkExtent2D extent = { VulkanEngine::engine->_window->getWidth(), VulkanEngine::engine->_window->getHeight() };

	builder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	builder._rasterizer = vkinit::rasterization_state_create_info(VK_POLYGON_MODE_FILL);
	builder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
	builder._viewport.x = 0.0f;
	builder._viewport.y = 0.0f;
	builder._viewport.maxDepth = 1.0f;
	builder._viewport.minDepth = 0.0f;
	builder._viewport.width = (float)VulkanEngine::engine->_window->getWidth();
	builder._viewport.height = (float)VulkanEngine::engine->_window->getHeight();
	builder._scissor.offset = { 0, 0 };
	builder._scissor.extent = extent;

	VkPipelineColorBlendAttachmentState att = vkinit::color_blend_attachment_state(0xf, VK_FALSE);

	builder._colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	builder._colorBlendStateInfo.attachmentCount = 1;
	builder._colorBlendStateInfo.pAttachments = &att;
	//builder._colorBlendStateInfo = vkinit::color_blend_state_create_info(1, &vkinit::color_blend_attachment_state(0xf, VK_FALSE));
	builder._multisampling = vkinit::multisample_state_create_info();

	_postPipeline = builder.build_pipeline(*device, _forwardRenderPass);

	vkDestroyShaderModule(*device, postVertexShader, nullptr);
	vkDestroyShaderModule(*device, postFragmentShader, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipelineLayout(*device, _postPipelineLayout, nullptr);
		vkDestroyPipeline(*device, _postPipeline, nullptr);
		});
}

void Renderer::create_post_descriptor()
{
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10}
	};

	std::vector<VkDescriptorSetLayoutBinding> bindings = {
		vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0)
	};

	VkDescriptorSetLayoutCreateInfo setInfo = {};
	setInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.pNext			= nullptr;
	setInfo.bindingCount	= static_cast<uint32_t>(bindings.size());
	setInfo.pBindings		= bindings.data();

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_postDescSetLayout));

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType					= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.pNext					= nullptr;
		allocInfo.descriptorPool		= _descriptorPool;
		allocInfo.descriptorSetCount	= 1;
		allocInfo.pSetLayouts			= &_postDescSetLayout;

		vkAllocateDescriptorSets(*device, &allocInfo, &_frames[i].postDescriptorSet);

		VkDescriptorImageInfo postDescriptor = vkinit::descriptor_image_info(
			_rtImage.imageView, VK_IMAGE_LAYOUT_GENERAL, _offscreenSampler);	// final image from rtx

		std::vector<VkWriteDescriptorSet> writes = {
			vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].postDescriptorSet, &postDescriptor, 0),
		};

		vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _postDescSetLayout, nullptr);
		});
}

void Renderer::build_post_command_buffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	std::array<VkClearValue, 2> clearValues;
	clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[1].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType						= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass					= _postRenderPass;
	renderPassBeginInfo.renderArea.extent.width		= VulkanEngine::engine->_window->getWidth();
	renderPassBeginInfo.renderArea.extent.height	= VulkanEngine::engine->_window->getHeight();
	renderPassBeginInfo.clearValueCount				= static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues				= clearValues.data();
	renderPassBeginInfo.framebuffer					= _postFramebuffers[VulkanEngine::engine->_indexSwapchainImage];

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

	vkBeginCommandBuffer(get_current_frame()._mainCommandBuffer, &cmdBufInfo);

	vkCmdBeginRenderPass(get_current_frame()._mainCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
	vkCmdBindPipeline(get_current_frame()._mainCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _postPipeline);

	VkDeviceSize offset = { 0 };

	Mesh* quad = Mesh::get_quad();

	vkCmdBindDescriptorSets(get_current_frame()._mainCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _postPipelineLayout, 0, 1, &get_current_frame().postDescriptorSet, 0, nullptr);
	vkCmdBindVertexBuffers(get_current_frame()._mainCommandBuffer, 0, 1, &quad->_vertexBuffer._buffer, &offset);
	vkCmdBindIndexBuffer(get_current_frame()._mainCommandBuffer, quad->_indexBuffer._buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(get_current_frame()._mainCommandBuffer, static_cast<uint32_t>(quad->_indices.size()), 1, 0, 0, 1);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), get_current_frame()._mainCommandBuffer);

	vkCmdEndRenderPass(get_current_frame()._mainCommandBuffer);
	VK_CHECK(vkEndCommandBuffer(get_current_frame()._mainCommandBuffer));
}

// HYBRID
// -------------------------------------------------------

void Renderer::create_hybrid_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSizes = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}
	};

	const uint32_t nInstances	= static_cast<uint32_t>(_scene->_entities.size());
	const uint32_t nDrawables	= static_cast<uint32_t>(_scene->get_drawable_nodes_size());
	const uint32_t nMaterials	= static_cast<uint32_t>(Material::_materials.size());
	const uint32_t nTextures	= static_cast<uint32_t>(Texture::_textures.size());
	const uint32_t nLights		= static_cast<uint32_t>(_scene->_lights.size());

	// binding = 0 TLAS
	// binding = 1 Storage image
	// binding = 2 Camera buffer
	// binding = 3 Gbuffers
	// binding = 4 Lights buffer
	// binding = 5 Vertices buffer
	// binding = 6 Indices buffer
	// binding = 7 Textures buffer
	// binding = 8 Skybox buffer
	// binding = 9 Materials buffer
	// binding = 10 Scene indices
	// binding = 11 Matrices buffer
	// binding = 12 Shadow image

	VkDescriptorSetLayoutBinding TLASBinding			= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);			// TLAS
	VkDescriptorSetLayoutBinding storageImageBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1);			// storage image
	VkDescriptorSetLayoutBinding cameraBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 2);			// Camera buffer
	VkDescriptorSetLayoutBinding gBuffersBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3, 6);
	VkDescriptorSetLayoutBinding lightsBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 4);	// Lights
	VkDescriptorSetLayoutBinding vertexBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 5, nInstances);	// Vertices
	VkDescriptorSetLayoutBinding indexBufferBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 6, nInstances);	// Indices
	VkDescriptorSetLayoutBinding texturesBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 7, nTextures); // Textures buffer
	VkDescriptorSetLayoutBinding matIdxBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 8); // Scene indices
	VkDescriptorSetLayoutBinding materialBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 9);	// Materials buffer
	VkDescriptorSetLayoutBinding skyboxBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 10, 2);
	VkDescriptorSetLayoutBinding matrixBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 11);	// Matrices
	VkDescriptorSetLayoutBinding shadowImageBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 12, nLights);	// Shadow image

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		TLASBinding,
		storageImageBinding,
		cameraBufferBinding,
		gBuffersBinding,
		lightsBufferBinding,
		vertexBufferBinding,
		indexBufferBinding,
		texturesBufferBinding,
		matrixBufferBinding,
		materialBufferBinding,
		matIdxBufferBinding,
		skyboxBufferBinding,
		shadowImageBinding
	};

	VkDescriptorSetLayoutCreateInfo setInfo = vkinit::descriptor_set_layout_create_info(static_cast<uint32_t>(setLayoutBindings.size()), setLayoutBindings);
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_hybridDescSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = vkinit::descriptor_set_allocate_info(_descriptorPool, &_hybridDescSetLayout);
	vkAllocateDescriptorSets(*device, &allocInfo, &_hybridDescSet);

	// Binding = 0 TLAS write
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
	descriptorAccelerationStructureInfo.sType						= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	descriptorAccelerationStructureInfo.accelerationStructureCount	= 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures		= &_topLevelAS.handle;

	// Binding = 1 Camera write
	VkDescriptorBufferInfo cameraBufferInfo = vkinit::descriptor_buffer_info(_rtCameraBuffer._buffer, sizeof(RTCameraData));

	// Binding = 2 Output image write
	VkDescriptorImageInfo storageImageDescriptor = vkinit::descriptor_image_info(_rtImage.imageView, VK_IMAGE_LAYOUT_GENERAL);

	// Binding = 3
	// Input deferred images write
	VkDescriptorImageInfo texDescriptorPosition = vkinit::descriptor_image_info(
		_deferredTextures[0].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Position
	VkDescriptorImageInfo texDescriptorNormal = vkinit::descriptor_image_info(
		_deferredTextures[1].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Normal
	VkDescriptorImageInfo texDescriptorAlbedo = vkinit::descriptor_image_info(
		_deferredTextures[2].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Albedo
	VkDescriptorImageInfo texDescriptorMotion = vkinit::descriptor_image_info(
		_deferredTextures[3].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Motion
	VkDescriptorImageInfo texDescriptorMaterial = vkinit::descriptor_image_info(
		_deferredTextures[4].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Material
	VkDescriptorImageInfo texDescriptorEmissive = vkinit::descriptor_image_info(
		_deferredTextures[5].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, _offscreenSampler);	// Emissive

	std::vector<VkDescriptorImageInfo> gbuffersDescInfo = { texDescriptorPosition, texDescriptorNormal, texDescriptorAlbedo, texDescriptorMotion, texDescriptorMaterial, texDescriptorEmissive };

	// Binding = 4 Lights buffer descriptor
	VkDescriptorBufferInfo lightDescBuffer = vkinit::descriptor_buffer_info(_lightBuffer._buffer, sizeof(uboLight) * nLights);

	std::vector<VkDescriptorBufferInfo> vertexDescInfo;
	std::vector<VkDescriptorBufferInfo> indexDescInfo;
	std::vector<glm::vec4> idVector;
	for (Object* obj : _scene->_entities)
	{
		AllocatedBuffer vBuffer;
		size_t bufferSize = sizeof(rtVertexAttribute) * obj->prefab->_mesh->_vertices.size();
		VulkanEngine::engine->create_buffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, vBuffer);

		std::vector<rtVertexAttribute> vAttr;
		std::vector<Vertex> vertices = obj->prefab->_mesh->_vertices;
		vAttr.reserve(vertices.size());
		for (Vertex& v : vertices) {
			vAttr.push_back({ {v.normal.x, v.normal.y, v.normal.z, 1}, {v.color.x, v.color.y, v.color.z, 1}, {v.uv.x, v.uv.y, 1, 1} });
		}

		void* vdata;
		vmaMapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation, &vdata);
		memcpy(vdata, vAttr.data(), bufferSize);
		vmaUnmapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation);

		// Binding = 5 Vertices Info
		VkDescriptorBufferInfo vertexBufferDescriptor = vkinit::descriptor_buffer_info(vBuffer._buffer, bufferSize);
		vertexDescInfo.push_back(vertexBufferDescriptor);

		// Binding = 6 Indices Info
		VkDescriptorBufferInfo indexBufferDescriptor = vkinit::descriptor_buffer_info(obj->prefab->_mesh->_indexBuffer._buffer, sizeof(uint32_t) * obj->prefab->_mesh->_indices.size());
		indexDescInfo.push_back(indexBufferDescriptor);

		for (Node* root : obj->prefab->_root)
		{
			root->fill_index_buffer(idVector);
		}
	}
	
	// Binding = 7 Textures info
	VkDescriptorSetAllocateInfo textureAllocInfo = {};
	textureAllocInfo.sType				= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	textureAllocInfo.pNext				= nullptr;
	textureAllocInfo.descriptorSetCount = 1;
	textureAllocInfo.pSetLayouts		= &_textureDescriptorSetLayout;
	textureAllocInfo.descriptorPool		= _descriptorPool;

	VK_CHECK(vkAllocateDescriptorSets(*device, &textureAllocInfo, &_textureDescriptorSet));

	VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
	VkSampler sampler;
	vkCreateSampler(*device, &samplerInfo, nullptr, &sampler);

	std::vector<VkDescriptorImageInfo> imageInfos;
	for (auto const& texture : Texture::_textures)
	{
		VkDescriptorImageInfo imageBufferInfo = vkinit::descriptor_image_info(texture.second->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, sampler);
		imageInfos.push_back(imageBufferInfo);
	}

	// Binding = 8 Skybox
	VkDescriptorImageInfo skyboxImagesDesc[2];
	skyboxImagesDesc[0] = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_8k.jpg")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	skyboxImagesDesc[1] = { sampler, Texture::GET("LA_Downtown_Helipad_GoldenHour_Env.hdr")->imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	// Binding = 9 Material info
	VkDescriptorBufferInfo materialBufferInfo = vkinit::descriptor_buffer_info(_matBuffer._buffer, sizeof(GPUMaterial) * nMaterials);

	// Binding = 10 ID info
	VkDescriptorBufferInfo idDescInfo = vkinit::descriptor_buffer_info(_idBuffer._buffer, sizeof(glm::vec4) * idVector.size());

	// Binding = 11 Matrices info
	VkDescriptorBufferInfo matrixDescInfo = vkinit::descriptor_buffer_info(_matricesBuffer._buffer, sizeof(glm::mat4) * _scene->_matricesVector.size());

	// Binding = 12 Shadow image
	std::vector<VkDescriptorImageInfo> shadowImagesDesc(_denoisedImages.size());
	for (decltype(_denoisedImages.size()) i = 0; i < _denoisedImages.size(); i++)
	{
		shadowImagesDesc[i] = { VK_NULL_HANDLE, _denoisedImages[i].imageView, VK_IMAGE_LAYOUT_GENERAL };
	}

	// Writes list
	VkWriteDescriptorSet accelerationStructureWrite = vkinit::write_descriptor_acceleration_structure(_hybridDescSet, &descriptorAccelerationStructureInfo, 0);
	VkWriteDescriptorSet storageImageWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _hybridDescSet, &storageImageDescriptor, 1);
	VkWriteDescriptorSet cameraWrite			= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _hybridDescSet, &cameraBufferInfo, 2);
	VkWriteDescriptorSet gbuffersWrite			= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, gbuffersDescInfo.data(), 3, gbuffersDescInfo.size());
	VkWriteDescriptorSet lightWrite				= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &lightDescBuffer, 4);
	VkWriteDescriptorSet vertexBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, vertexDescInfo.data(), 5, nInstances);
	VkWriteDescriptorSet indexBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, indexDescInfo.data(), 6, nInstances);
	VkWriteDescriptorSet texturesBufferWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, imageInfos.data(), 7, nTextures);
	VkWriteDescriptorSet matIdxBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &idDescInfo, 8);
	VkWriteDescriptorSet materialBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &materialBufferInfo, 9);
	VkWriteDescriptorSet skyboxBufferWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, skyboxImagesDesc, 10, 2);
	VkWriteDescriptorSet matrixBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &matrixDescInfo, 11);
	VkWriteDescriptorSet shadowImageWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _hybridDescSet, shadowImagesDesc.data(), 12, nLights);
	
	std::vector<VkWriteDescriptorSet> writes = {
		accelerationStructureWrite,	// 0 TLAS
		storageImageWrite,
		cameraWrite, 
		gbuffersWrite,
		lightWrite,
		vertexBufferWrite,
		indexBufferWrite,
		texturesBufferWrite,
		matrixBufferWrite,
		materialBufferWrite,
		matIdxBufferWrite,
		skyboxBufferWrite,
		shadowImageWrite,
	};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _hybridDescSetLayout, nullptr);
		vkDestroySampler(*device, sampler, nullptr);
		});
}

uint32_t Renderer::alignedSize(uint32_t value, uint32_t alignment)
{
	return (value + alignment - 1) & ~(alignment - 1);
}



//
//void Renderer::build_hybrid_command_buffers()
//{
//	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);
//
//	VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
//
//	VkCommandBuffer& cmd = _hybridCommandBuffer;
//
//	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBufInfo));
//
//	VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
//	raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(raygenSBT._buffer);
//	raygenShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	raygenShaderSbtEntry.size			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//
//	VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
//	missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(missSBT._buffer);
//	missShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize * 2;
//
//	VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
//	hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(hitSBT._buffer);
//	hitShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//	hitShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
//
//	VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};
//
//	uint32_t width = VulkanEngine::engine->_window->getWidth(), height = VulkanEngine::engine->_window->getHeight();
//
//	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _hybridPipeline);
//	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _hybridPipelineLayout, 0, 1, &_hybridDescSet, 0, nullptr);
//
//	vkCmdTraceRaysKHR(
//		cmd,
//		&raygenShaderSbtEntry,
//		&missShaderSbtEntry,
//		&hitShaderSbtEntry,
//		&callableShaderSbtEntry,
//		width,
//		height,
//		1
//	);
//
//	VK_CHECK(vkEndCommandBuffer(cmd));
//}