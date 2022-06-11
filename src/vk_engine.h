#pragma once

#include <vma/vk_mem_alloc.h>

#include <imgui/imgui.h>
#include <imgui/ImGuizmo.h>
#include <imgui/imgui_impl_sdl.h>
#include <imgui/imgui_impl_vulkan.h>

#include "renderer.h"
#include "scene.h"

class Window;

#define VK_CHECK(x)												\
	do															\
	{															\
		VkResult err = x;										\
		if (err)												\
		{														\
			std::cout << "Vulkan EROR: " << err << std::endl;	\
			abort();											\
		}														\
	} while (0);												\

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)();
		}
		deletors.clear();
	}
};

enum renderMode{
	DEFERRED,
	RAYTRACING,
	HYBRID
};

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 prevView;
	glm::mat4 prevProj;
	glm::vec3 pos;
	float pad0;
	glm::vec4 nearFarFrame;
};

struct RTCameraData
{
	glm::mat4 invView;
	glm::mat4 invProj;
	glm::vec4 frame;
	//alignas(16) float frame;
};

struct GPUSceneData {
	glm::vec4 fogColor;
	glm::vec4 fogDistances;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct GPUObjectData {
	glm::mat4 modelMatrix;
};

struct uboLight {
	glm::vec4	position;	// w used for maxDistance
	glm::vec4	color;		// w used for intensity;
	alignas(16) float	radius;		// the radius of sphere light for soft shadows purpose
};

struct UploadContext {
	VkFence			_uploadFence;
	VkCommandPool	_commandPool;
};

//TODO solve problem when creating the commands. We may want 2 FRAME_OVERLAP but it has to take
//into account the number of swapchain images
//constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:

	static VulkanEngine* engine;

	VulkanEngine();

	Renderer*	renderer;
	renderMode	_mode;

	bool		_isInitialized{ false };
	int			_frameNumber{ 0 };
	bool		_bQuit{ false };
	uint32_t	_indexSwapchainImage{ 0 };

	bool _skyboxFollow{ true };

	Window *_window;
	Scene* _scene;

	DeletionQueue _mainDeletionQueue;

	void* deviceCreatepNextChain = nullptr;

	// Device stuff
	VkInstance							_instance;
	VkDebugUtilsMessengerEXT			_debug_messenger;
	VkPhysicalDevice					_gpu;
	VkDevice							_device;
	VkSurfaceKHR						_surface;
	VkPhysicalDeviceProperties			_gpuProperties;
	VkPhysicalDeviceMemoryProperties	_memoryProperties;

	// Swapchain stuff
	VkSwapchainKHR						_swapchain;
	VkFormat							_swapchainImageFormat;
	std::vector<VkImage>				_swapchainImages;
	std::vector<VkImageView>			_swapchainImageViews;

	VkImageView							_depthImageView;
	AllocatedImage						_depthImage;
	VkFormat							_depthFormat;

	// Textures used as attachments from the first pass
	VkQueue								_graphicsQueue;
	uint32_t							_graphicsQueueFamily;
	UploadContext						_uploadContext;

	// Set 0 is a Global set - updated once per frame
	//AllocatedBuffer						_cameraBuffer;	// Buffer to hold all information from camera to the shader
	GPUSceneData						_sceneParameters;
	AllocatedBuffer						_sceneBuffer;	// Buffer to hold all information involving the scene

	// Set 1 is a per pass set - updated per pass
	// Buffer to hold all matrices information from the scene
	AllocatedBuffer						_objectBuffer;

	// Allocator
	VmaAllocator						_allocator;

	Texture _skyboxTexture;

	bool mouse_locked;
	int  _denoise_frame{ 0 };
	bool _denoise{ true };
	uint32_t debugTarget = 0;
	int _samples = 1;

	VkPhysicalDeviceDescriptorIndexingFeaturesEXT		enabledIndexingFeatures{};

	// vkRay
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR		_rtProperties;
	VkPhysicalDeviceAccelerationStructureFeaturesKHR	_asFeatures;

	VkPhysicalDeviceBufferDeviceAddressFeatures			enabledBufferDeviceAddressFeatures{};
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR		enabledRayTracingPipelineFeatures{};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR	enabledAccelerationStructureFeatures{};



	PFN_vkGetBufferDeviceAddressKHR						vkGetBufferDeviceAddressKHR;

	VkCommandPool	_commandPool;

	VkCommandPool	_commandPool2;

	AllocatedBuffer transformBuffer;
	   
	// Main functions
	void init();

	void cleanup();

	void run();

	void update(const float dt);

	void immediate_submit(std::function<void(VkCommandBuffer)>&& function);

	VkCommandBuffer beginSingleTimeCommands();

	void endSingleTimeCommands(VkCommandBuffer commandBuffer);

	void create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage, AllocatedBuffer &buffer, bool destroy = true);

	size_t pad_uniform_buffer_size(size_t originalSize);

	void create_attachment(VkFormat format, VkImageUsageFlagBits usage, Texture* texture);

	// Loads a shader module from a SPIR-V file
	bool load_shader_module(
		const char* filePath, 
		VkShaderModule* outShaderModule);

	VkPipelineShaderStageCreateInfo load_shader_stage(const char* filePath, VkShaderModule* outShaderModule, VkShaderStageFlagBits stage);

	uint32_t getBufferDeviceAddress(VkBuffer buffer);

	void recreate_swapchain();

	void updateFrame();

	void resetFrame();

	uint32_t getMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties, VkBool32* memTypeFound);

private:

	void init_vulkan();

	void init_swapchain();

	void clean_swapchain();

	void init_ray_tracing();

	void init_upload_commands();

	void init_imgui();

	VkCommandBuffer create_command_buffer(VkCommandBufferLevel level, bool begin);

	// VKRay features passed to the logical device as a pNext pointer
	void get_enabled_features();

	void updateCameraMatrices();

};

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo>	_shaderStages;
	VkPipelineVertexInputStateCreateInfo			_vertexInputInfo{};
	VkPipelineInputAssemblyStateCreateInfo			_inputAssembly;
	VkViewport										_viewport;
	VkRect2D										_scissor;
	VkPipelineRasterizationStateCreateInfo			_rasterizer;
	VkPipelineColorBlendStateCreateInfo				_colorBlendStateInfo;
	VkPipelineMultisampleStateCreateInfo			_multisampling;
	VkPipelineLayout								_pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo			_depthStencil;

	VkPipeline build_pipeline(VkDevice device, VkRenderPass pass);
};