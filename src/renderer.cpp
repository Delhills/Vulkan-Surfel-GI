
#include <renderer.h>
#include <vk_engine.h>
#include <vk_initializers.h>
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
	
	init_descriptors();
	init_deferred_descriptors();
	//init_forward_pipeline();
	init_deferred_pipelines();
	build_previous_command_buffer();

	// Ray tracing
	vkCreateAccelerationStructureKHR			= reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(*device, "vkCreateAccelerationStructureKHR"));
	vkBuildAccelerationStructuresKHR			= reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(*device, "vkBuildAccelerationStructuresKHR"));
	vkGetAccelerationStructureBuildSizesKHR		= reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(*device, "vkGetAccelerationStructureBuildSizesKHR"));
	vkGetAccelerationStructureDeviceAddressKHR	= reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(*device, "vkGetAccelerationStructureDeviceAddressKHR"));
	vkCmdBuildAccelerationStructuresKHR			= reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(*device, "vkCmdBuildAccelerationStructuresKHR"));
	vkGetRayTracingShaderGroupHandlesKHR		= reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(*device, "vkGetRayTracingShaderGroupHandlesKHR"));
	vkCreateRayTracingPipelinesKHR				= reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(*device, "vkCreateRayTracingPipelinesKHR"));
	vkCmdTraceRaysKHR							= reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(*device, "vkCmdTraceRaysKHR"));
	vkDestroyAccelerationStructureKHR			= reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(*device, "vkDestroyAccelerationStructureKHR"));

	create_storage_image();

	// post
	create_post_renderPass();
	create_post_framebuffers();
	create_post_descriptor();
	create_post_pipeline();
	
	// VKRay
	create_bottom_acceleration_structure();
	create_top_acceleration_structure();
	create_rt_descriptors();
	create_hybrid_descriptors();
	init_raytracing_pipeline();
	create_shader_binding_table();
	build_raytracing_command_buffers();
	build_hybrid_command_buffers();
	
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
	Texture position, normal, albedo, material, depth;
	VulkanEngine::engine->create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &position);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &normal);
	VulkanEngine::engine->create_attachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &albedo);
	//VulkanEngine::engine->create_attachment(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &material);
	VulkanEngine::engine->create_attachment(VulkanEngine::engine->_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, &depth);

	_deferredTextures.push_back(position);
	_deferredTextures.push_back(normal);
	_deferredTextures.push_back(albedo);
	//_deferredTextures.push_back(material);
	_deferredTextures.push_back(depth);

	std::array<VkAttachmentDescription, 4> attachmentDescs = {};

	// Init attachment properties
	for (uint32_t i = 0; i < 4; i++)
	{
		attachmentDescs[i].samples			= VK_SAMPLE_COUNT_1_BIT;
		attachmentDescs[i].loadOp			= VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachmentDescs[i].storeOp			= VK_ATTACHMENT_STORE_OP_STORE;
		attachmentDescs[i].stencilLoadOp	= VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescs[i].stencilStoreOp	= VK_ATTACHMENT_STORE_OP_DONT_CARE;
		if (i == 3)
		{
			attachmentDescs[i].initialLayout	= VK_IMAGE_LAYOUT_UNDEFINED;
			attachmentDescs[i].finalLayout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
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
	//attachmentDescs[3].format = VK_FORMAT_R8G8B8A8_UNORM;
	attachmentDescs[3].format = VulkanEngine::engine->_depthFormat;

	std::vector<VkAttachmentReference> colorReferences;
	colorReferences.push_back({ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	colorReferences.push_back({ 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	//colorReferences.push_back({ 3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkAttachmentReference depthReference;
	depthReference.attachment	= 3;
	depthReference.layout		= VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint		= VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.pColorAttachments		= colorReferences.data();
	subpass.colorAttachmentCount	= static_cast<uint32_t>(colorReferences.size());
	subpass.pDepthStencilAttachment = &depthReference;

	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass		= VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass		= 0;
	dependencies[0].srcStageMask	= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[0].dstStageMask	= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask	= VK_ACCESS_MEMORY_READ_BIT;
	dependencies[0].dstAccessMask	= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass		= 0;
	dependencies[1].dstSubpass		= VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask	= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask	= VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
	dependencies[1].srcAccessMask	= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dstAccessMask	= VK_ACCESS_MEMORY_READ_BIT;
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
	ImGui::Render();

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
	submit.pSignalSemaphores		= &_offscreenSemaphore;
	submit.commandBufferCount		= 1;
	submit.pCommandBuffers			= &_offscreenComandBuffer;

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));

	build_deferred_command_buffer();

	// Second pass
	submit.pWaitSemaphores			= &_offscreenSemaphore;
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

void Renderer::raytrace()
{
	ImGui::Render();

	VK_CHECK(vkWaitForFences(*device, 1, &get_current_frame()._renderFence, VK_TRUE, UINT64_MAX));
	VK_CHECK(vkResetFences(*device, 1, &get_current_frame()._renderFence));

	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkResult result;

	result = vkAcquireNextImageKHR(*device, *swapchain, UINT64_MAX, get_current_frame()._presentSemaphore, VK_NULL_HANDLE, &VulkanEngine::engine->_indexSwapchainImage);

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR ) {
		VulkanEngine::engine->recreate_swapchain();
		return;
	}
	else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("Failed to acquire swap chain image");
	}

	VkSubmitInfo submit{};
	submit.sType					= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext					= nullptr;
	submit.pWaitDstStageMask		= waitStages;
	submit.waitSemaphoreCount		= 1;
	submit.pWaitSemaphores			= &get_current_frame()._presentSemaphore;
	submit.signalSemaphoreCount		= 1;
	submit.pSignalSemaphores		= &_rtSemaphore;
	submit.commandBufferCount		= 1;
	submit.pCommandBuffers			= &_rtCommandBuffer;

	VK_CHECK(vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE));

	build_post_command_buffers();

	submit.pWaitSemaphores		= &_rtSemaphore;
	submit.pSignalSemaphores	= &get_current_frame()._renderSemaphore;
	submit.pCommandBuffers		= &get_current_frame()._mainCommandBuffer;

	result = vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, get_current_frame()._renderFence);

	VkPresentInfoKHR present{};
	present.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present.pNext				= nullptr;
	present.swapchainCount		= 1;
	present.pSwapchains			= swapchain;
	present.waitSemaphoreCount	= 1;
	present.pWaitSemaphores		= &get_current_frame()._renderSemaphore;
	present.pImageIndices		= &VulkanEngine::engine->_indexSwapchainImage;

	result = vkQueuePresentKHR(VulkanEngine::engine->_graphicsQueue, &present);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		VulkanEngine::engine->recreate_swapchain();
	}
	else if (result != VK_SUCCESS)
		throw std::runtime_error("failed to present swap chain images!");
}

void Renderer::rasterize_hybrid()
{
	ImGui::Render();

	VK_CHECK(vkWaitForFences(*device, 1, &get_current_frame()._renderFence, VK_TRUE, 1000000000));
	VK_CHECK(vkResetFences(*device, 1, &get_current_frame()._renderFence));

	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

	VkResult result;

	result = vkAcquireNextImageKHR(*device, *swapchain, UINT64_MAX, get_current_frame()._presentSemaphore, VK_NULL_HANDLE, &VulkanEngine::engine->_indexSwapchainImage);

	VK_CHECK(vkResetCommandBuffer(get_current_frame()._mainCommandBuffer, 0));

	if (VulkanEngine::engine->_indexSwapchainImage > 1)
	{
		//VK_CHECK(vkResetCommandBuffer(_offscreenComandBuffer, 0));
		build_previous_command_buffer();
	}

	// First pass
	VkSubmitInfo submit = {};
	submit.sType				= VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.pNext				= nullptr;
	submit.pWaitDstStageMask	= waitStages;
	submit.waitSemaphoreCount	= 1;
	submit.pWaitSemaphores		= &get_current_frame()._presentSemaphore;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores	= &_offscreenSemaphore;
	submit.commandBufferCount	= 1;
	submit.pCommandBuffers		= &_offscreenComandBuffer;

	result = vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE);

	// Second pass raytrace
	submit.pWaitSemaphores		= &_offscreenSemaphore;
	submit.pSignalSemaphores	= &_rtSemaphore;
	submit.pCommandBuffers		= &_hybridCommandBuffer;
	result = vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, VK_NULL_HANDLE);

	build_post_command_buffers();
	submit.pWaitSemaphores		= &_rtSemaphore;
	submit.pSignalSemaphores	= &get_current_frame()._renderSemaphore;
	submit.pCommandBuffers		= &get_current_frame()._mainCommandBuffer;
	result = vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, get_current_frame()._renderFence);

	VkPresentInfoKHR present{};
	present.sType				= VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	present.pNext				= nullptr;
	present.swapchainCount		= 1;
	present.pSwapchains			= swapchain;
	present.waitSemaphoreCount	= 1;
	present.pWaitSemaphores		= &get_current_frame()._renderSemaphore;
	present.pImageIndices		= &VulkanEngine::engine->_indexSwapchainImage;

	result = vkQueuePresentKHR(VulkanEngine::engine->_graphicsQueue, &present);
	if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
		VulkanEngine::engine->recreate_swapchain();
	}
	else if (result != VK_SUCCESS)
		throw std::runtime_error("failed to present swap chain images!");
}

void Renderer::render_gui()
{
	// Imgui new frame
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplSDL2_NewFrame(VulkanEngine::engine->_window->_handle);

	ImGui::NewFrame();

	ImGui::Begin("Debug window");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::Checkbox("Denoise", &VulkanEngine::engine->_denoise);

	const std::vector<std::string> targets = { "Final", "Position", "Normal", "Albedo" };
	std::vector<const char*> charTargets;
	charTargets.reserve(targets.size());
	for (size_t i = 0; i < targets.size(); i++)
	{
		charTargets.push_back(targets[i].c_str());
	}

	const char* title = "Target";
	int index = VulkanEngine::engine->debugTarget;

	if (ImGui::Combo(title, &index, &charTargets[0], targets.size(), targets.size()))
	{
		VulkanEngine::engine->debugTarget = index;
		void* debugData;
		vmaMapMemory(VulkanEngine::engine->_allocator, VulkanEngine::engine->renderer->_debugBuffer._allocation, &debugData);
		memcpy(debugData, &VulkanEngine::engine->debugTarget, sizeof(uint32_t));
		vmaUnmapMemory(VulkanEngine::engine->_allocator, VulkanEngine::engine->renderer->_debugBuffer._allocation);
	}

	for (auto& light : _scene->_lights)
	{
		if (ImGui::TreeNode(&light, "Light")) {
			if (ImGui::Button("Select"))
				gizmoEntity = light;
			ImGui::SliderFloat3("Position", &((glm::vec3)light->m_matrix[3])[0], -200, 200);
			ImGui::ColorEdit3("Color", &light->color.x);
			ImGui::SliderFloat("Intensity", &light->intensity, 0, 1000);
			ImGui::SliderFloat("Max Distance", &light->maxDistance, 0, 500);
			ImGui::SliderFloat("Radius", &light->radius, 0, 10);
			ImGui::TreePop();
		}
	}
	for (auto& entity : _scene->_entities)
	{
		if (ImGui::TreeNode(&entity, "Entity")) {
			if (ImGui::Button("Select"))
				gizmoEntity = entity;
			ImGui::SliderFloat3("Position", &((glm::vec3)entity->m_matrix[3])[0], -200, 200);
			ImGui::TreePop();
		}
	}
	ImGui::End();

	if (!gizmoEntity)
		return;

	glm::mat4& matrix = gizmoEntity->m_matrix;

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
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1700.f / 900.f, 0.1f, 200.0f);
	ImGuizmo::Manipulate(&_scene->_camera->getView()[0][0], &projection[0][0], mCurrentGizmoOperation, mCurrentGizmoMode, &matrix[0][0], NULL, useSnap ? &snap.x : NULL);

	ImGui::EndFrame();
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
	std::array<VkImageView, 4> attachments;
	attachments[0] = _deferredTextures.at(0).imageView;	// Position
	attachments[1] = _deferredTextures.at(1).imageView;	// Normal
	attachments[2] = _deferredTextures.at(2).imageView;	// Color	
	//attachments[3] = _deferredTextures.at(3).imageView;	// Material	
	attachments[3] = _deferredTextures.at(3).imageView;	// Depth

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

	// Once the layouts have been created, we allocate the descriptor sets
	// First create necessary buffers
	VulkanEngine::engine->create_buffer(sizeof(GPUCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, VulkanEngine::engine->_cameraBuffer);
	VulkanEngine::engine->create_buffer(sizeof(GPUMaterial), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, VulkanEngine::engine->_objectBuffer);

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
	VkDescriptorBufferInfo cameraInfo = vkinit::descriptor_buffer_info(VulkanEngine::engine->_cameraBuffer._buffer, sizeof(GPUCameraData), 0);

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
	skyboxImageInfo.imageView			= Texture::GET("data/textures/woods.jpg")->imageView;
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
	VkDescriptorSetLayoutBinding lightBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 4);	// Lights buffer
	VkDescriptorSetLayoutBinding debugBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 5);	// Debug display

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		positionBinding,
		normalBinding,
		albedoBinding,
		lightBinding,
		debugBinding
	};

	VkDescriptorSetLayoutCreateInfo setInfo = {};
	setInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.pNext			= nullptr;
	setInfo.bindingCount	= static_cast<uint32_t>(setLayoutBindings.size());
	setInfo.pBindings		= setLayoutBindings.data();

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_deferredSetLayout));

	for (int i = 0; i < FRAME_OVERLAP; i++)
	{
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType					= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.pNext					= nullptr;
		allocInfo.descriptorPool		= _descriptorPool;
		allocInfo.descriptorSetCount	= 1;
		allocInfo.pSetLayouts			= &_deferredSetLayout;

		vkAllocateDescriptorSets(*device, &allocInfo, &_frames[i].deferredDescriptorSet);

		VkDescriptorImageInfo texDescriptorPosition = vkinit::descriptor_image_create_info(
			_offscreenSampler, _deferredTextures[0].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);	// Position
		VkDescriptorImageInfo texDescriptorNormal = vkinit::descriptor_image_create_info(
			_offscreenSampler, _deferredTextures[1].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);	// Normal
		VkDescriptorImageInfo texDescriptorAlbedo = vkinit::descriptor_image_create_info(
			_offscreenSampler, _deferredTextures[2].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);	// Albedo

		int nLights = _scene->_lights.size();
		if (!lightBuffer._buffer)
			VulkanEngine::engine->create_buffer(sizeof(uboLight) * nLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, lightBuffer);
		if (!_debugBuffer._buffer)
			VulkanEngine::engine->create_buffer(sizeof(uint32_t), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _debugBuffer);

		VkDescriptorBufferInfo lightBufferDesc;
		lightBufferDesc.buffer = lightBuffer._buffer;
		lightBufferDesc.offset = 0;
		lightBufferDesc.range  = sizeof(uboLight) * nLights;

		VkDescriptorBufferInfo debugDesc;
		debugDesc.buffer = _debugBuffer._buffer;
		debugDesc.offset = 0;
		debugDesc.range = sizeof(uint32_t);

		VkWriteDescriptorSet positionWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorPosition, 0);
		VkWriteDescriptorSet normalWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorNormal, 1);
		VkWriteDescriptorSet albedoWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _frames[i].deferredDescriptorSet, &texDescriptorAlbedo, 2);
		VkWriteDescriptorSet lightBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i].deferredDescriptorSet, &lightBufferDesc, 4);
		VkWriteDescriptorSet debugWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _frames[i].deferredDescriptorSet, &debugDesc, 5);

		std::vector<VkWriteDescriptorSet> writes = {
			positionWrite,
			normalWrite,
			albedoWrite,
			lightBufferWrite,
			debugWrite
		};

		vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
	}

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _deferredSetLayout, nullptr);
		});
}

void Renderer::init_forward_pipeline()
{
	VulkanEngine* engine = VulkanEngine::engine;
	VkShaderModule vertexShader;
	if (!engine->load_shader_module(vkutil::findFile("basic.vert.spv", searchPaths, true).c_str() , &vertexShader)) {
		std::cout << "Could not load forward vertex shader!" << std::endl;
	}
	VkShaderModule fragmentShader;
	if (!engine->load_shader_module(vkutil::findFile("forward.frag.spv", searchPaths, true).c_str(), &fragmentShader)) {
		std::cout << "Could not load fragment vertex shader!" << std::endl;
	}

	PipelineBuilder pipBuilder;
	pipBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader));
	pipBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));

	VkDescriptorSetLayout offscreenSetLayouts[] = { _offscreenDescriptorSetLayout, _textureDescriptorSetLayout };

	VkPushConstantRange push_constant;
	push_constant.offset		= 0;
	push_constant.size			= sizeof(int);
	push_constant.stageFlags	= VK_SHADER_STAGE_FRAGMENT_BIT;

	VkPushConstantRange test_constant;
	test_constant.offset		= sizeof(int);
	test_constant.size			= sizeof(material_matrix);
	test_constant.stageFlags	= VK_SHADER_STAGE_VERTEX_BIT;

	VkPushConstantRange matrix_constant;
	matrix_constant.offset = 0;
	matrix_constant.size = sizeof(glm::mat4);
	matrix_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkPushConstantRange constants[] = { push_constant, test_constant };

	VkPipelineLayoutCreateInfo offscreenPipelineLayoutInfo = vkinit::pipeline_layout_create_info();
	offscreenPipelineLayoutInfo.setLayoutCount			= 2;
	offscreenPipelineLayoutInfo.pSetLayouts				= offscreenSetLayouts;
	offscreenPipelineLayoutInfo.pushConstantRangeCount	= 1;
	offscreenPipelineLayoutInfo.pPushConstantRanges		= &matrix_constant;

	VK_CHECK(vkCreatePipelineLayout(*device, &offscreenPipelineLayoutInfo, nullptr, &_forwardPipelineLayout));

	VertexInputDescription vertexDescription = Vertex::get_vertex_description();
	pipBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
	pipBuilder._vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
	pipBuilder._vertexInputInfo.pVertexAttributeDescriptions	= vertexDescription.attributes.data();
	pipBuilder._vertexInputInfo.vertexBindingDescriptionCount	= vertexDescription.bindings.size();
	pipBuilder._vertexInputInfo.pVertexBindingDescriptions		= vertexDescription.bindings.data();

	pipBuilder._pipelineLayout = _forwardPipelineLayout;

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

	pipBuilder._colorBlendStateInfo = vkinit::color_blend_state_create_info(1, &vkinit::color_blend_attachment_state(0xf, VK_FALSE));
	pipBuilder._multisampling		= vkinit::multisample_state_create_info();

	_forwardPipeline = pipBuilder.build_pipeline(*device, _forwardRenderPass);

	vkDestroyShaderModule(*device, vertexShader, nullptr);
	vkDestroyShaderModule(*device, fragmentShader, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipelineLayout(*device, _forwardPipelineLayout, nullptr);
		vkDestroyPipeline(*device, _forwardPipeline, nullptr);
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
	matrix_constant.size								= sizeof(glm::mat4);
	matrix_constant.stageFlags							= VK_SHADER_STAGE_VERTEX_BIT;

	VkPushConstantRange material_constant;
	material_constant.offset							= sizeof(glm::mat4);
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

	std::array<VkPipelineColorBlendAttachmentState, 3> blendAttachmentStates = {
		vkinit::color_blend_attachment_state(
			VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, VK_FALSE),
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

	pipBuilder._colorBlendStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	pipBuilder._colorBlendStateInfo.attachmentCount = 1;
	pipBuilder._colorBlendStateInfo.pAttachments = &att; //vkinit::color_blend_state_create_info(1, &vkinit::color_blend_attachment_state(0xf, VK_FALSE));

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

	VK_CHECK(vkBeginCommandBuffer(_offscreenComandBuffer, &cmdBufInfo));

	VkDeviceSize offset = { 0 };

	std::array<VkClearValue, 4> clearValues;
	clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[1].color = { 0.0f, 0.0f, 0.0f, 1.0f };
	clearValues[2].color = { 1.0f, 0.0f, 0.0f, 1.0f };
	clearValues[3].depthStencil = { 1.0f, 0 };

	VkRenderPassBeginInfo renderPassBeginInfo = {};
	renderPassBeginInfo.sType						= VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassBeginInfo.renderPass					= _offscreenRenderPass;
	renderPassBeginInfo.framebuffer					= _offscreenFramebuffer;
	renderPassBeginInfo.renderArea.extent.width		= VulkanEngine::engine->_window->getWidth();
	renderPassBeginInfo.renderArea.extent.height	= VulkanEngine::engine->_window->getHeight();
	renderPassBeginInfo.clearValueCount				= static_cast<uint32_t>(clearValues.size());
	renderPassBeginInfo.pClearValues				= clearValues.data();

	vkCmdBeginRenderPass(_offscreenComandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

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

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), get_current_frame()._mainCommandBuffer);

	vkCmdEndRenderPass(get_current_frame()._mainCommandBuffer);
	VK_CHECK(vkEndCommandBuffer(get_current_frame()._mainCommandBuffer));
}

void Renderer::create_storage_image()
{
	VkExtent3D extent			= { VulkanEngine::engine->_window->getWidth(), VulkanEngine::engine->_window->getHeight(), 1 };
	VkImageCreateInfo imageInfo = vkinit::image_create_info(VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, extent);
	imageInfo.initialLayout		= VK_IMAGE_LAYOUT_UNDEFINED;

	VmaAllocationCreateInfo allocInfo{};
	allocInfo.usage			= VMA_MEMORY_USAGE_GPU_ONLY;
	allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(VulkanEngine::engine->_allocator, &imageInfo, &allocInfo, 
		&_rtImage.image._image, &_rtImage.image._allocation, nullptr);

	VkImageViewCreateInfo imageViewInfo = vkinit::image_view_create_info(VK_FORMAT_B8G8R8A8_UNORM, _rtImage.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
	VK_CHECK(vkCreateImageView(*device, &imageViewInfo, nullptr, &_rtImage.imageView));

	VkBufferCreateInfo bufferInfo = vkinit::buffer_create_info(sizeof(RTCameraData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
	allocInfo.requiredFlags = 0;

	VK_CHECK(vmaCreateBuffer(VulkanEngine::engine->_allocator, &bufferInfo, &allocInfo, &VulkanEngine::engine->rtCameraBuffer._buffer, &VulkanEngine::engine->rtCameraBuffer._allocation, nullptr));

	glm::mat4 view = _scene->_camera->getView();
	glm::mat4 projection = glm::perspective(glm::radians(70.0f), 1700.f / 900.f, 0.1f, 200.0f);
	projection[1][1] *= -1;

	RTCameraData camera;
	camera.invProj = glm::inverse(projection);
	camera.invView = glm::inverse(view);

	void* data;
	vmaMapMemory(VulkanEngine::engine->_allocator, VulkanEngine::engine->rtCameraBuffer._allocation, &data);
	memcpy(data, &camera, sizeof(RTCameraData));
	vmaUnmapMemory(VulkanEngine::engine->_allocator, VulkanEngine::engine->rtCameraBuffer._allocation);
	
	VulkanEngine::engine->immediate_submit([&](VkCommandBuffer cmd) {
		VkImageMemoryBarrier imageMemoryBarrier{};
		imageMemoryBarrier.sType			= VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		imageMemoryBarrier.image			= _rtImage.image._image;
		imageMemoryBarrier.oldLayout		= VK_IMAGE_LAYOUT_UNDEFINED;
		imageMemoryBarrier.newLayout		= VK_IMAGE_LAYOUT_GENERAL;
		imageMemoryBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

		vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
	});

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vmaDestroyImage(VulkanEngine::engine->_allocator, _rtImage.image._image, _rtImage.image._allocation);
		vkDestroyImageView(*device, _rtImage.imageView, nullptr);
		vmaDestroyBuffer(VulkanEngine::engine->_allocator, VulkanEngine::engine->rtCameraBuffer._buffer, VulkanEngine::engine->rtCameraBuffer._allocation);
		});
	
}

void Renderer::recreate_renderer()
{
	init_render_pass();
	init_forward_render_pass();
	init_offscreen_render_pass();
	init_forward_pipeline();
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

			vertexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(obj->prefab->_mesh->_vertexBuffer._buffer);
			indexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(obj->prefab->_mesh->_indexBuffer._buffer);

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

	buildTlas(_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
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
		asBuildGeoInfos[i].type						= VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		asBuildGeoInfos[i].flags					= flags;
		asBuildGeoInfos[i].geometryCount			= 1;
		asBuildGeoInfos[i].pGeometries				= &_blas[i].asGeometry;
		asBuildGeoInfos[i].mode						= VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
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

	AllocatedBuffer scratchBuffer;
	VulkanEngine::engine->create_buffer(maxScratch, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY, scratchBuffer, false);
	VkDeviceOrHostAddressConstKHR scratchBufferDeviceAddress{};
	scratchBufferDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(scratchBuffer._buffer);

	// Once the scratch buffer is created we finally end inflating the asBuildGeosInfo struct and can proceed to submit the creation command
	for (uint32_t i = 0; i < blasSize; i++) {

		asBuildGeoInfos[i].scratchData.deviceAddress = scratchBufferDeviceAddress.deviceAddress;

		std::vector<VkAccelerationStructureBuildRangeInfoKHR*> asBuildStructureRangeInfo = { &_blas[i].asBuildRangeInfo };

		VulkanEngine::engine->immediate_submit([=](VkCommandBuffer cmd) {
			vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildGeoInfos[i], asBuildStructureRangeInfo.data());
		});
	}

	// Finally we can free the scratch buffer
	vmaDestroyBuffer(VulkanEngine::engine->_allocator, scratchBuffer._buffer, scratchBuffer._allocation);
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
	instanceDataDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(_instanceBuffer._buffer);

	// Create a stucture that holds a device pointer to the uploaded instances.
	VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
	instancesData.sType					= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	instancesData.arrayOfPointers		= VK_FALSE;
	instancesData.data					= instanceDataDeviceAddress;

	// Put the above structure to the structure geometry
	VkAccelerationStructureGeometryKHR asGeometry = vkinit::acceleration_structure_geometry_khr();
	asGeometry.geometryType				= VK_GEOMETRY_TYPE_INSTANCES_KHR;
	asGeometry.flags					= VK_GEOMETRY_OPAQUE_BIT_KHR;
	asGeometry.geometry.instances		= instancesData;

	// Find sizes
	VkAccelerationStructureBuildGeometryInfoKHR asBuildGeometryInfo = vkinit::acceleration_structure_build_geometry_info();
	asBuildGeometryInfo.type			= VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	asBuildGeometryInfo.mode			= update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	asBuildGeometryInfo.flags			= flags;
	asBuildGeometryInfo.geometryCount	= 1;
	asBuildGeometryInfo.pGeometries		= &asGeometry;
	asBuildGeometryInfo.srcAccelerationStructure = update ? _topLevelAS.handle : VK_NULL_HANDLE;

	uint32_t									primitiveCount = static_cast<uint32_t>(instances.size());
	VkAccelerationStructureBuildSizesInfoKHR	asBuildSizesInfo = vkinit::acceleration_structure_build_sizes_info();
	vkGetAccelerationStructureBuildSizesKHR(
		VulkanEngine::engine->_device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&asBuildGeometryInfo,
		&primitiveCount,
		&asBuildSizesInfo
	);

	if (!update)
	{
		create_acceleration_structure(_topLevelAS, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, asBuildSizesInfo);
	}

	AllocatedBuffer scratchBuffer;
	VulkanEngine::engine->create_buffer(asBuildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
		VMA_MEMORY_USAGE_GPU_ONLY, scratchBuffer, false);
	VkDeviceOrHostAddressConstKHR scratchBufferDeviceAddress{};
	scratchBufferDeviceAddress.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(scratchBuffer._buffer);
	asBuildGeometryInfo.dstAccelerationStructure	= _topLevelAS.handle;
	asBuildGeometryInfo.scratchData.deviceAddress	= scratchBufferDeviceAddress.deviceAddress;

	VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};
	asBuildRangeInfo.primitiveCount		= static_cast<uint32_t>(instances.size());
	asBuildRangeInfo.primitiveOffset	= 0;
	asBuildRangeInfo.firstVertex		= 0;
	asBuildRangeInfo.transformOffset	= 0;

	const VkAccelerationStructureBuildRangeInfoKHR* pAsBuildRangeInfo = &asBuildRangeInfo;
	
	VkCommandPoolCreateInfo poolInfo = vkinit::command_pool_create_info(VulkanEngine::engine->_graphicsQueueFamily);

	VkCommandPool pool;
	vkCreateCommandPool(VulkanEngine::engine->_device, &poolInfo, nullptr, &pool);
	VkCommandBufferAllocateInfo alloc = vkinit::command_buffer_allocate_info(pool, 1);

	VkCommandBuffer cmd;
	vkAllocateCommandBuffers(VulkanEngine::engine->_device, &alloc, &cmd);

	VkCommandBufferBeginInfo beginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	vkBeginCommandBuffer(cmd, &beginInfo);

	vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildGeometryInfo, &pAsBuildRangeInfo);

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit = vkinit::submit_info(&cmd);
	vkQueueSubmit(VulkanEngine::engine->_graphicsQueue, 1, &submit, nullptr);

	vkQueueWaitIdle(VulkanEngine::engine->_graphicsQueue);

	vkDestroyCommandPool(VulkanEngine::engine->_device, pool, nullptr);
	
	vmaDestroyBuffer(VulkanEngine::engine->_allocator, scratchBuffer._buffer, scratchBuffer._allocation);
}

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

void Renderer::create_rt_descriptors()
{
	std::vector<VkDescriptorPoolSize> poolSize = {
		{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
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

	const unsigned int nInstances	= _scene->_entities.size();
	const unsigned int nLights		= _scene->_lights.size();
	const unsigned int nMaterials	= Material::_materials.size();
	const unsigned int nTextures	= Texture::_textures.size();

	if (!lightBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(uboLight) * nLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, lightBuffer);
	if(!_matBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(GPUMaterial) * nMaterials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _matBuffer);

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
	VkDescriptorSetLayoutBinding skyboxBufferBinding				= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_MISS_BIT_KHR, 10, 1);

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
		skyboxBufferBinding
		});

	// Allocate Descriptor
	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
	descriptorSetLayoutCreateInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorSetLayoutCreateInfo.bindingCount	= static_cast<uint32_t>(bindings.size());
	descriptorSetLayoutCreateInfo.pBindings		= bindings.data();
	VK_CHECK(vkCreateDescriptorSetLayout(*device, &descriptorSetLayoutCreateInfo, nullptr, &_rtDescriptorSetLayout));

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = vkinit::descriptor_set_allocate_info(_rtDescriptorPool, &_rtDescriptorSetLayout, 1);
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
	VkDescriptorImageInfo storageImageDescriptor{};
	storageImageDescriptor.imageView			= _rtImage.imageView;
	storageImageDescriptor.imageLayout			= VK_IMAGE_LAYOUT_GENERAL;

	// Binding = 2 Camera 
	VkDescriptorBufferInfo _rtDescriptorBufferInfo{};
	_rtDescriptorBufferInfo.buffer				= VulkanEngine::engine->rtCameraBuffer._buffer;
	_rtDescriptorBufferInfo.offset				= 0;
	_rtDescriptorBufferInfo.range				= sizeof(RTCameraData);


	std::vector<VkDescriptorBufferInfo> vertexDescInfo;
	std::vector<VkDescriptorBufferInfo> indexDescInfo;
	std::vector<glm::mat4> matricesVector;
	std::vector<glm::vec4> idVector;
	for (Object* obj : _scene->_entities)
	{
		// Binding = 3 Vertex buffer
		std::vector<Vertex> vertices	= obj->prefab->_mesh->_vertices;
		std::vector<uint32_t> indices	= obj->prefab->_mesh->_indices;
		size_t vertexBufferSize			= sizeof(rtVertexAttribute) * vertices.size();
		size_t indexBufferSize			= sizeof(uint32_t) * indices.size();
		AllocatedBuffer vBuffer;
		VulkanEngine::engine->create_buffer(vertexBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, vBuffer);

		std::vector<rtVertexAttribute> vAttr;
		vAttr.reserve(vertices.size());
		for (Vertex& v : vertices) {
			vAttr.push_back({ {v.position.x, v.position.y, v.position.z, 1}, {v.normal.x, v.normal.y, v.normal.z, 1}, {v.color.x, v.color.y, v.color.z, 1}, {v.uv.x, v.uv.y, 1, 1} });
		}

		void* vdata;
		vmaMapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation, &vdata);
		memcpy(vdata, vAttr.data(), vertexBufferSize);
		vmaUnmapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation);

		VkDescriptorBufferInfo vertexBufferDescriptor{};
		vertexBufferDescriptor.buffer	= vBuffer._buffer;
		vertexBufferDescriptor.offset	= 0;
		vertexBufferDescriptor.range	= vertexBufferSize;
		vertexDescInfo.push_back(vertexBufferDescriptor);

		// Binding = 4 Index buffer
		VkDescriptorBufferInfo indexBufferDescriptor{};
		indexBufferDescriptor.buffer	= obj->prefab->_mesh->_indexBuffer._buffer;
		indexBufferDescriptor.offset	= 0;
		indexBufferDescriptor.range		= indexBufferSize;
		indexDescInfo.push_back(indexBufferDescriptor);

		for (Node* root : obj->prefab->_root)
		{
			root->fill_matrix_buffer(matricesVector, obj->m_matrix);
			root->fill_index_buffer(idVector);
		}
	}

	// Binding = 5 Matrix buffer
	AllocatedBuffer matricesBuffer;
	VulkanEngine::engine->create_buffer(sizeof(glm::mat4) * matricesVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, matricesBuffer);

	void* matricesData;
	vmaMapMemory(VulkanEngine::engine->_allocator, matricesBuffer._allocation, &matricesData);
	memcpy(matricesData, matricesVector.data(), sizeof(glm::mat4) * matricesVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, matricesBuffer._allocation);
	
	VkDescriptorBufferInfo matrixDescInfo;
	matrixDescInfo.buffer	= matricesBuffer._buffer;
	matrixDescInfo.offset	= 0;
	matrixDescInfo.range	= sizeof(glm::mat4) * matricesVector.size();

	// Binding = 6 lights
	VkDescriptorBufferInfo lightBufferInfo;
	lightBufferInfo.buffer		= lightBuffer._buffer;
	lightBufferInfo.offset		= 0;
	lightBufferInfo.range		= sizeof(uboLight) * nLights;

	// Binding = 7 ID buffer
	AllocatedBuffer idBuffer;
	VulkanEngine::engine->create_buffer(sizeof(glm::vec4) * idVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, idBuffer);

	void* idData;
	vmaMapMemory(VulkanEngine::engine->_allocator, idBuffer._allocation, &idData);
	memcpy(idData, idVector.data(), sizeof(glm::vec4) * idVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, idBuffer._allocation);

	VkDescriptorBufferInfo idDescInfo;
	idDescInfo.buffer	= idBuffer._buffer;
	idDescInfo.offset	= 0;
	idDescInfo.range	= sizeof(glm::vec4) * idVector.size();

	// Binding = 8 Materials
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

	VkDescriptorBufferInfo materialBufferInfo;
	materialBufferInfo.buffer	= _matBuffer._buffer;
	materialBufferInfo.offset	= 0;
	materialBufferInfo.range	= sizeof(GPUMaterial) * nMaterials;
	
	// Binding = 9 Textures
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

	VkDescriptorImageInfo skyboxBufferInfo = {};
	skyboxBufferInfo.sampler		= sampler;
	skyboxBufferInfo.imageView		= Texture::GET("woods.jpg")->imageView;
	skyboxBufferInfo.imageLayout	= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

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
	VkWriteDescriptorSet skyboxBufferWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _rtDescriptorSet, &skyboxBufferInfo, 10);

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
		skyboxBufferWrite
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

	// Ray generation group
	VkShaderModule rayGenModule, hraygenModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("raygen.rgen.spv", searchPaths, true).c_str(), &rayGenModule, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("hybridRaygen.rgen.spv", searchPaths, true).c_str(), &hraygenModule, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
		VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
		shaderGroup.sType				= VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
		shaderGroup.type				= VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
		shaderGroup.generalShader		= 0;
		shaderGroup.closestHitShader	= VK_SHADER_UNUSED_KHR;
		shaderGroup.anyHitShader		= VK_SHADER_UNUSED_KHR;
		shaderGroup.intersectionShader	= VK_SHADER_UNUSED_KHR;
		shaderGroups.push_back(shaderGroup);
		shaderGroup.generalShader = 0;
		hybridShaderGroups.push_back(shaderGroup);
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
	VkShaderModule shadowModule, hshadowModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadow.rmiss.spv", searchPaths, true).c_str(), &shadowModule, VK_SHADER_STAGE_MISS_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("shadow.rmiss.spv", searchPaths, true).c_str(), &hshadowModule, VK_SHADER_STAGE_MISS_BIT_KHR));
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

	// Hit group
	VkShaderModule hitModule, hhitModule;
	{
		shaderStages.push_back(engine->load_shader_stage(vkutil::findFile("closesthit.rchit.spv", searchPaths, true).c_str(), &hitModule, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
		hybridShaderStages.push_back(engine->load_shader_stage(vkutil::findFile("hybridHit.rchit.spv", searchPaths, true).c_str(), &hhitModule, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
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
	rtPipelineCreateInfo.maxPipelineRayRecursionDepth	= 10;
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
	hybridPipelineInfo.maxPipelineRayRecursionDepth = 2;
	hybridPipelineInfo.layout						= _hybridPipelineLayout;

	VK_CHECK(vkCreateRayTracingPipelinesKHR(*device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &hybridPipelineInfo, nullptr, &_hybridPipeline));

	vkDestroyShaderModule(*device, rayGenModule, nullptr);
	vkDestroyShaderModule(*device, hitModule, nullptr);
	vkDestroyShaderModule(*device, missModule, nullptr);
	vkDestroyShaderModule(*device, shadowModule, nullptr);
	vkDestroyShaderModule(*device, hraygenModule, nullptr);
	vkDestroyShaderModule(*device, hmissModule, nullptr);
	vkDestroyShaderModule(*device, hshadowModule, nullptr);
	vkDestroyShaderModule(*device, hhitModule, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyPipeline(*device, _rtPipeline, nullptr);
		vkDestroyPipeline(*device, _hybridPipeline, nullptr);
		vkDestroyPipelineLayout(*device, _rtPipelineLayout, nullptr);
		vkDestroyPipelineLayout(*device, _hybridPipelineLayout, nullptr);
		});
}

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

}

void Renderer::build_raytracing_command_buffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
	
	VkCommandBuffer& cmd = _rtCommandBuffer;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBufInfo));

	VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
	raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(raygenShaderBindingTable._buffer);
	raygenShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	raygenShaderSbtEntry.size			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	
	VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
	missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(missShaderBindingTable._buffer);
	missShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize * 2;
	
	VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
	hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(hitShaderBindingTable._buffer);
	hitShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	hitShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;

	VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};

	uint32_t width = VulkanEngine::engine->_window->getWidth(), height = VulkanEngine::engine->_window->getHeight();

	VkImage& image = VulkanEngine::engine->_swapchainImages[VulkanEngine::engine->_indexSwapchainImage];

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _rtPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _rtPipelineLayout, 0, 1, &_rtDescriptorSet, 0, nullptr);
	
	vkCmdTraceRaysKHR(
		cmd,
		&raygenShaderSbtEntry,
		&missShaderSbtEntry,
		&hitShaderSbtEntry,
		&callableShaderSbtEntry,
		width,
		height,
		1
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
			//vkDestroyImageView(*device, VulkanEngine::engine->_swapchainImageViews[i], nullptr);
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
		vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0),
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

		VkDescriptorImageInfo postDescriptor = vkinit::descriptor_image_create_info(
			_offscreenSampler, _rtImage.imageView, VK_IMAGE_LAYOUT_GENERAL);	// final image from rtx

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

	// binding = 0 TLAS
	// binding = 1 Storage image
	// binding = 2 Camera buffer
	// binding = 3 Position Gbuffer
	// binding = 4 Normal Gbuffer
	// binding = 5 Albedo Gbuffer
	// binding = 6 Lights buffer
	// binding = 7 Vertices buffer
	// binding = 8 Indices buffer
	// binding = 9 Textures buffer
	// binding = 10 Matrices buffer
	// binding = 11 Materials buffer
	// binding = 12 Scene indices

	VkDescriptorSetLayoutBinding TLASBinding			= 
		vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 0);			// TLAS
	VkDescriptorSetLayoutBinding storageImageBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 1);			// storage image
	VkDescriptorSetLayoutBinding cameraBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 2);			// Camera buffer
	VkDescriptorSetLayoutBinding positionImageBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 3);	// Position	image
	VkDescriptorSetLayoutBinding normalImageBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 4);	// Normals	image
	VkDescriptorSetLayoutBinding albedoImageBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_RAYGEN_BIT_KHR, 5);	// Albedo	image
	VkDescriptorSetLayoutBinding lightsBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 6);	// Lights
	VkDescriptorSetLayoutBinding vertexBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 7, nInstances);	// Vertices
	VkDescriptorSetLayoutBinding indexBufferBinding		= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 8, nInstances);	// Indices
	VkDescriptorSetLayoutBinding texturesBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR, 9, nTextures); // Textures buffer
	VkDescriptorSetLayoutBinding skyboxBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_MISS_BIT_KHR, 10, 1);
	VkDescriptorSetLayoutBinding materialBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 11);	// Materials buffer
	VkDescriptorSetLayoutBinding matIdxBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 12); // Scene indices
	VkDescriptorSetLayoutBinding matrixBufferBinding	= vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, 13);	// Matrices

	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings =
	{
		TLASBinding,
		storageImageBinding,
		cameraBufferBinding,
		positionImageBinding,
		normalImageBinding,
		albedoImageBinding,
		lightsBufferBinding,
		vertexBufferBinding,
		indexBufferBinding,
		texturesBufferBinding,
		matrixBufferBinding,
		materialBufferBinding,
		matIdxBufferBinding,
		skyboxBufferBinding
	};

	VkDescriptorSetLayoutCreateInfo setInfo = {};
	setInfo.sType			= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.pNext			= nullptr;
	setInfo.bindingCount	= static_cast<uint32_t>(setLayoutBindings.size());
	setInfo.pBindings		= setLayoutBindings.data();

	VK_CHECK(vkCreateDescriptorSetLayout(*device, &setInfo, nullptr, &_hybridDescSetLayout));

	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType					= VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.pNext					= nullptr;
	allocInfo.descriptorPool		= _descriptorPool;
	allocInfo.descriptorSetCount	= 1;
	allocInfo.pSetLayouts			= &_hybridDescSetLayout;

	vkAllocateDescriptorSets(*device, &allocInfo, &_hybridDescSet);

	// TLAS write
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAccelerationStructureInfo{};
	descriptorAccelerationStructureInfo.sType						= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	descriptorAccelerationStructureInfo.accelerationStructureCount	= 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures		= &_topLevelAS.handle;

	VkWriteDescriptorSet accelerationStructureWrite{};
	accelerationStructureWrite.sType			= VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	accelerationStructureWrite.pNext			= &descriptorAccelerationStructureInfo;
	accelerationStructureWrite.dstSet			= _hybridDescSet;
	accelerationStructureWrite.dstBinding		= 0;
	accelerationStructureWrite.descriptorCount	= 1;
	accelerationStructureWrite.descriptorType	= VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

	// Camera write
	VkDescriptorBufferInfo cameraBufferInfo{};
	cameraBufferInfo.buffer = VulkanEngine::engine->rtCameraBuffer._buffer;
	cameraBufferInfo.offset = 0;
	cameraBufferInfo.range = sizeof(RTCameraData);

	// Output image write
	VkDescriptorImageInfo storageImageDescriptor{};
	storageImageDescriptor.imageView	= _rtImage.imageView;
	storageImageDescriptor.imageLayout	= VK_IMAGE_LAYOUT_GENERAL;

	// Input deferred images write
	VkDescriptorImageInfo texDescriptorPosition = vkinit::descriptor_image_create_info(
		_offscreenSampler, _deferredTextures[0].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);	// Position
	VkDescriptorImageInfo texDescriptorNormal = vkinit::descriptor_image_create_info(
		_offscreenSampler, _deferredTextures[1].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);	// Normal
	VkDescriptorImageInfo texDescriptorAlbedo = vkinit::descriptor_image_create_info(
		_offscreenSampler, _deferredTextures[2].imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);	// Albedo

	// lights write
	int nLights = _scene->_lights.size();
	if(!lightBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(uboLight) * nLights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, lightBuffer);

	VkDescriptorBufferInfo lightDescBuffer;
	lightDescBuffer.buffer = lightBuffer._buffer;
	lightDescBuffer.offset = 0;
	lightDescBuffer.range  = sizeof(uboLight) * nLights;

	std::vector<VkDescriptorBufferInfo> vertexDescInfo;
	std::vector<VkDescriptorBufferInfo> indexDescInfo;
	std::vector<glm::mat4> matricesVector;
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
			vAttr.push_back({ {v.position.x, v.position.y, v.position.z, 1}, {v.normal.x, v.normal.y, v.normal.z, 1}, {v.color.x, v.color.y, v.color.z, 1}, {v.uv.x, v.uv.y, 1, 1} });
		}

		void* vdata;
		vmaMapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation, &vdata);
		memcpy(vdata, vAttr.data(), bufferSize);
		vmaUnmapMemory(VulkanEngine::engine->_allocator, vBuffer._allocation);

		VkDescriptorBufferInfo vertexBufferDescriptor{};
		vertexBufferDescriptor.buffer	= vBuffer._buffer;
		vertexBufferDescriptor.offset	= 0;
		vertexBufferDescriptor.range	= bufferSize;
		vertexDescInfo.push_back(vertexBufferDescriptor);

		VkDescriptorBufferInfo indexBufferDescriptor{};
		indexBufferDescriptor.buffer	= obj->prefab->_mesh->_indexBuffer._buffer;
		indexBufferDescriptor.offset	= 0;
		indexBufferDescriptor.range		= sizeof(uint32_t) * obj->prefab->_mesh->_indices.size();
		indexDescInfo.push_back(indexBufferDescriptor);

		for (Node* root : obj->prefab->_root)
		{
			root->fill_matrix_buffer(matricesVector, obj->m_matrix);
			root->fill_index_buffer(idVector);
		}
	}

	// Binding = 5 Matrix buffer
	AllocatedBuffer matricesBuffer;
	VulkanEngine::engine->create_buffer(sizeof(glm::mat4) * matricesVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, matricesBuffer);

	void* matricesData;
	vmaMapMemory(VulkanEngine::engine->_allocator, matricesBuffer._allocation, &matricesData);
	memcpy(matricesData, matricesVector.data(), sizeof(glm::mat4) * matricesVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, matricesBuffer._allocation);

	VkDescriptorBufferInfo matrixDescInfo;
	matrixDescInfo.buffer = matricesBuffer._buffer;
	matrixDescInfo.offset = 0;
	matrixDescInfo.range = sizeof(glm::mat4) * matricesVector.size();

	AllocatedBuffer idBuffer;
	VulkanEngine::engine->create_buffer(sizeof(glm::vec4) * idVector.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, idBuffer);

	void* idData;
	vmaMapMemory(VulkanEngine::engine->_allocator, idBuffer._allocation, &idData);
	memcpy(idData, idVector.data(), sizeof(glm::vec4) * idVector.size());
	vmaUnmapMemory(VulkanEngine::engine->_allocator, idBuffer._allocation);

	VkDescriptorBufferInfo idDescInfo;
	idDescInfo.buffer = idBuffer._buffer;
	idDescInfo.offset = 0;
	idDescInfo.range = sizeof(glm::vec4) * idVector.size();

	if (!_matBuffer._buffer)
		VulkanEngine::engine->create_buffer(sizeof(GPUMaterial) * nMaterials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, _matBuffer);

	std::vector<GPUMaterial> materials;
	for (Material* it : Material::_materials)
	{
		GPUMaterial mat = it->materialToShader();
		materials.push_back(mat);
	}

	void* matData;
	vmaMapMemory(VulkanEngine::engine->_allocator, _matBuffer._allocation, &matData);
	memcpy(matData, materials.data(), sizeof(GPUMaterial) * nMaterials);
	vmaUnmapMemory(VulkanEngine::engine->_allocator, _matBuffer._allocation);

	VkDescriptorBufferInfo materialBufferInfo;
	materialBufferInfo.buffer	= _matBuffer._buffer;
	materialBufferInfo.offset	= 0;
	materialBufferInfo.range	= sizeof(GPUMaterial) * nMaterials;

	// Textures info
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
		VkDescriptorImageInfo imageBufferInfo = {};
		imageBufferInfo.sampler		= sampler;
		imageBufferInfo.imageView	= texture.second->imageView;
		imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		imageInfos.push_back(imageBufferInfo);
	}

	VkDescriptorImageInfo skyboxBufferInfo = {};
	skyboxBufferInfo.sampler		= sampler;
	skyboxBufferInfo.imageView		= Texture::GET("woods.jpg")->imageView;
	skyboxBufferInfo.imageLayout	= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkWriteDescriptorSet storageImageWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, _hybridDescSet, &storageImageDescriptor, 1);
	VkWriteDescriptorSet cameraWrite			= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, _hybridDescSet, &cameraBufferInfo, 2);
	VkWriteDescriptorSet positionImageWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, &texDescriptorPosition, 3);
	VkWriteDescriptorSet normalImageWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, &texDescriptorNormal, 4);
	VkWriteDescriptorSet albedoImageWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, &texDescriptorAlbedo, 5);
	VkWriteDescriptorSet lightWrite				= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &lightDescBuffer, 6);
	VkWriteDescriptorSet vertexBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, vertexDescInfo.data(), 7, nInstances);
	VkWriteDescriptorSet indexBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, indexDescInfo.data(), 8, nInstances);
	VkWriteDescriptorSet texturesBufferWrite	= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, imageInfos.data(), 9, nTextures);
	VkWriteDescriptorSet skyboxBufferWrite		= vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, _hybridDescSet, &skyboxBufferInfo, 10);
	VkWriteDescriptorSet materialBufferWrite	= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &materialBufferInfo, 11);
	VkWriteDescriptorSet matIdxBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &idDescInfo, 12);
	VkWriteDescriptorSet matrixBufferWrite		= vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _hybridDescSet, &matrixDescInfo, 13);

	std::vector<VkWriteDescriptorSet> writes = {
		accelerationStructureWrite,	// 0 TLAS
		storageImageWrite,
		cameraWrite, 
		positionImageWrite,
		normalImageWrite,
		albedoImageWrite,
		lightWrite,
		vertexBufferWrite,
		indexBufferWrite,
		texturesBufferWrite,
		matrixBufferWrite,
		materialBufferWrite,
		matIdxBufferWrite,
		skyboxBufferWrite
	};

	vkUpdateDescriptorSets(*device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vkDestroyDescriptorSetLayout(*device, _hybridDescSetLayout, nullptr);
		vkDestroySampler(*device, sampler, nullptr);
		});
}

void Renderer::build_hybrid_command_buffers()
{
	VkCommandBufferBeginInfo cmdBufInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT);

	VkImageSubresourceRange subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	VkCommandBuffer& cmd = _hybridCommandBuffer;

	VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBufInfo));

	VkStridedDeviceAddressRegionKHR raygenShaderSbtEntry{};
	raygenShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(raygenSBT._buffer);
	raygenShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	raygenShaderSbtEntry.size			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;

	VkStridedDeviceAddressRegionKHR missShaderSbtEntry{};
	missShaderSbtEntry.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(missSBT._buffer);
	missShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	missShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize * 2;

	VkStridedDeviceAddressRegionKHR hitShaderSbtEntry{};
	hitShaderSbtEntry.deviceAddress		= VulkanEngine::engine->getBufferDeviceAddress(hitSBT._buffer);
	hitShaderSbtEntry.stride			= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;
	hitShaderSbtEntry.size				= VulkanEngine::engine->_rtProperties.shaderGroupHandleSize;

	VkStridedDeviceAddressRegionKHR callableShaderSbtEntry{};

	uint32_t width = VulkanEngine::engine->_window->getWidth(), height = VulkanEngine::engine->_window->getHeight();

	VkImage& image = VulkanEngine::engine->_swapchainImages[VulkanEngine::engine->_indexSwapchainImage];

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _hybridPipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, _hybridPipelineLayout, 0, 1, &_hybridDescSet, 0, nullptr);

	vkCmdTraceRaysKHR(
		cmd,
		&raygenShaderSbtEntry,
		&missShaderSbtEntry,
		&hitShaderSbtEntry,
		&callableShaderSbtEntry,
		width,
		height,
		1
	);

	VK_CHECK(vkEndCommandBuffer(cmd));
}