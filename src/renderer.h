#pragma once
#include <vk_types.h>
#include <glm/glm/glm.hpp>

#include "scene.h"
#include "vk_textures.h"

struct FrameData
{
	VkSemaphore		_renderSemaphore;
	VkSemaphore		_presentSemaphore;
	VkFence			_renderFence;

	VkCommandPool	_commandPool;
	VkCommandBuffer _mainCommandBuffer;

	VkDescriptorSet deferredDescriptorSet;
	VkDescriptorSet postDescriptorSet;
	VkDescriptorSet deferredLightDescriptorSet;
	AllocatedBuffer _lightBuffer;
};

struct pushConstants {
	glm::vec4 data;
	glm::mat4 render_matrix;
};

struct Surfel
{
	glm::vec3 position;
	float padding0;
	glm::vec3 normal;
	float padding1;
	glm::vec3 color;
	float radius;
};

struct SurfelData
{
	glm::vec3 mean;
	float pad0;

	glm::vec3 shortMean;
	float vbbr;

	glm::vec3 variance;
	float inconsistency;

	glm::vec3 hitpos;
	float padding0;

	glm::vec3 hitnormal;
	float padding1;

	glm::vec3 hitenergy;
	float padding2;

	glm::vec3 traceresult;
	float padding3;
};


static const unsigned int SURFEL_STATS_OFFSET_COUNT = 0;
static const unsigned int SURFEL_INDIRECT_NUMTHREADS = 32;
static const glm::uvec3 SURFEL_GRID_DIMENSIONS = glm::uvec3(128, 64, 128); // (64, 32, 64)   (128, 64, 128)
static const unsigned int SURFEL_TABLE_SIZE = SURFEL_GRID_DIMENSIONS.x * SURFEL_GRID_DIMENSIONS.y * SURFEL_GRID_DIMENSIONS.z;
static const unsigned int SURFEL_CAPACITY = 100000;
static const float SURFEL_TARGET_COVERAGE = 0.5;
const float SURFEL_MAX_RADIUS = 1;


struct AccelerationStructure {
	VkAccelerationStructureKHR	handle;
	uint64_t					deviceAddress = 0;
	AllocatedBuffer	buffer;
};

struct RayTracingScratchBuffer {

	uint64_t					deviceAddress = 0;
	AllocatedBuffer				buffer;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class Renderer {

public:

	Renderer(Scene* scene);

	// Auxiliar pointer to engine variables
	VkDevice*		device;
	VkSwapchainKHR* swapchain;
	int*			frameNumber;
	Entity*			gizmoEntity;
	Scene*			_scene;

	FrameData		_frames[FRAME_OVERLAP];
	pushConstants	_constants;

	Texture blueNoise;


	// RASTERIZER VARIABLES -----------------------
	VkRenderPass				_forwardRenderPass;
	VkCommandPool				_commandPool;
	VkCommandPool				_resetCommandPool;
	VkDescriptorPool			_descriptorPool;

	// Forward stuff
	VkPipelineLayout			_forwardPipelineLayout;
	VkPipeline					_forwardPipeline;

	// DEFERRED STUFF
	// Onscreen stuff
	VkRenderPass				_renderPass;
	std::vector<VkFramebuffer>	_framebuffers;
	VkDescriptorSetLayout		_deferredSetLayout;
	VkPipelineLayout			_finalPipelineLayout;
	VkPipeline					_finalPipeline;

	std::vector<Texture>		_deferredTextures;

	// Offscreen stuff
	VkFramebuffer				_offscreenFramebuffer;
	VkRenderPass				_offscreenRenderPass;
	VkDescriptorSetLayout		_offscreenDescriptorSetLayout;
	VkDescriptorSet				_offscreenDescriptorSet;
	VkDescriptorSetLayout		_objectDescriptorSetLayout;
	VkDescriptorSet				_objectDescriptorSet;
	VkDescriptorSetLayout		_textureDescriptorSetLayout;
	VkDescriptorSet				_textureDescriptorSet;
	VkCommandBuffer				_offscreenComandBuffer;
	VkSampler					_offscreenSampler;
	VkSemaphore					_offscreenSemaphore;
	VkPipelineLayout			_offscreenPipelineLayout;
	VkPipeline					_offscreenPipeline;

	AllocatedBuffer				_cameraBuffer;
	AllocatedBuffer				_cameraPositionBuffer;

	// Skybox pass
	VkDescriptorSetLayout		_skyboxDescriptorSetLayout;
	VkDescriptorSet				_skyboxDescriptorSet;
	VkPipeline					_skyboxPipeline;
	VkPipelineLayout			_skyboxPipelineLayout;
	AllocatedBuffer				_skyboxBuffer;

	// RAYTRACING VARIABLES ------------------------
	VkDescriptorPool			_rtDescriptorPool;
	VkDescriptorSetLayout		_rtDescriptorSetLayout;
	VkDescriptorSet				_rtDescriptorSet;
	Texture						_rtImage;
	VkPipeline					_rtPipeline;
	VkPipelineLayout			_rtPipelineLayout;
	VkCommandBuffer				_rtCommandBuffer;
	VkSemaphore					_rtSemaphore;

	std::vector<AccelerationStructure>	_bottomLevelAS;
	AccelerationStructure				_topLevelAS;

	std::vector<BlasInput>		_blas;
	std::vector<TlasInstance>	_tlas;
	AllocatedBuffer				_lightBuffer;
	AllocatedBuffer				_debugBuffer;
	AllocatedBuffer				_matBuffer;
	AllocatedBuffer				_instanceBuffer;
	AllocatedBuffer				_rtCameraBuffer;
	AllocatedBuffer				_matricesBuffer;
	AllocatedBuffer				_idBuffer;
	AllocatedBuffer				_shadowSamplesBuffer;
	AllocatedBuffer				_frameCountBuffer;

	AllocatedBuffer				raygenShaderBindingTable;
	AllocatedBuffer				missShaderBindingTable;
	AllocatedBuffer				hitShaderBindingTable;



	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};

	PFN_vkCreateAccelerationStructureKHR				vkCreateAccelerationStructureKHR;
	PFN_vkGetAccelerationStructureBuildSizesKHR			vkGetAccelerationStructureBuildSizesKHR;
	PFN_vkGetAccelerationStructureDeviceAddressKHR		vkGetAccelerationStructureDeviceAddressKHR;
	PFN_vkBuildAccelerationStructuresKHR				vkBuildAccelerationStructuresKHR;
	PFN_vkCmdBuildAccelerationStructuresKHR				vkCmdBuildAccelerationStructuresKHR;
	PFN_vkGetRayTracingShaderGroupHandlesKHR			vkGetRayTracingShaderGroupHandlesKHR;
	PFN_vkCreateRayTracingPipelinesKHR					vkCreateRayTracingPipelinesKHR;
	PFN_vkCmdTraceRaysKHR								vkCmdTraceRaysKHR;
	PFN_vkDestroyAccelerationStructureKHR				vkDestroyAccelerationStructureKHR;



	// POST VARIABLES ------------------------
	VkPipeline					_postPipeline;
	VkPipelineLayout			_postPipelineLayout;
	VkDescriptorSet				_postDescSet;
	VkDescriptorSetLayout		_postDescSetLayout;
	VkRenderPass				_postRenderPass;
	std::vector<VkFramebuffer>	_postFramebuffers;

	// HYBRID VARIABLES -----------------------
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> hybridShaderGroups{};
	VkPipeline					_hybridPipeline;
	VkPipelineLayout			_hybridPipelineLayout;
	VkDescriptorSet				_hybridDescSet;
	VkDescriptorSetLayout		_hybridDescSetLayout;
	VkCommandBuffer				_hybridCommandBuffer;

	AllocatedBuffer				raygenSBT;
	AllocatedBuffer				missSBT;
	AllocatedBuffer				hitSBT;

	// SHADOW VARIABLES ----------------------
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shadowShaderGroups{};
	VkDescriptorPool			_shadowDescPool;
	VkDescriptorSet				_shadowDescSet;
	VkDescriptorSetLayout		_shadowDescSetLayout;
	//Texture						_shadowImage;
	VkPipeline					_shadowPipeline;
	VkPipelineLayout			_shadowPipelineLayout;
	VkCommandBuffer				_shadowCommandBuffer;
	VkSemaphore					_shadowSemaphore;
	std::vector<Texture>		_shadowImages;

	AllocatedBuffer				sraygenSBT;
	AllocatedBuffer				smissSBT;
	AllocatedBuffer				shitSBT;

	// SHADOW POST VARIABLES -----------------
	VkPipeline					_sPostPipeline;
	VkPipelineLayout			_sPostPipelineLayout;
	VkRenderPass				_sPostRenderPass;
	VkDescriptorPool			_sPostDescPool;
	VkDescriptorSet				_sPostDescSet;
	VkDescriptorSetLayout		_sPostDescSetLayout;
	std::vector<Texture>		_denoisedImages;
	VkCommandBuffer				_denoiseCommandBuffer;
	VkSemaphore					_denoiseSemaphore;
	AllocatedBuffer				_denoiseFrameBuffer;





	VkCommandBuffer				_SurfelPositionCmd;
	VkSemaphore					_SurfelPositionSemaphore;
	VkDescriptorPool			_SurfelPositionDescPool;
	VkDescriptorSet				_SurfelPositionDescSet;
	VkDescriptorSetLayout		_SurfelPositionDescSetLayout;
	VkPipeline					_SurfelPositionPipeline;
	VkPipelineLayout			_SurfelPositionPipelineLayout;
	VkSampler					_SurfelPositionNormalSampler;


	VkCommandBuffer				_PrepareIndirectCmdBuffer;
	VkSemaphore					_PrepareIndirectSemaphore;
	VkDescriptorPool			_PrepareIndirectDescPool;
	VkDescriptorSet				_PrepareIndirectDescSet;
	VkDescriptorSetLayout		_PrepareIndirectDescSetLayout;
	VkPipeline					_PrepareIndirectPipeline;
	VkPipelineLayout			_PrepareIndirectPipelineLayout;


	VkCommandBuffer				_GridResetCmdBuffer;
	VkSemaphore					_GridResetSemaphore;
	VkDescriptorPool			_GridResetDescPool;
	VkDescriptorSet				_GridResetDescSet;
	VkDescriptorSetLayout		_GridResetDescSetLayout;
	VkPipeline					_GridResetPipeline;
	VkPipelineLayout			_GridResetPipelineLayout;


	VkCommandBuffer				_UpdateSurfelsCmdBuffer;
	VkSemaphore					_UpdateSurfelsSemaphore;
	VkDescriptorPool			_UpdateSurfelsDescPool;
	VkDescriptorSet				_UpdateSurfelsDescSet;
	VkDescriptorSetLayout		_UpdateSurfelsDescSetLayout;
	VkPipeline					_UpdateSurfelsPipeline;
	VkPipelineLayout			_UpdateSurfelsPipelineLayout;

	VkCommandBuffer				_GridOffsetCmdBuffer;
	VkSemaphore					_GridOffsetSemaphore;
	VkDescriptorPool			_GridOffsetDescPool;
	VkDescriptorSet				_GridOffsetDescSet;
	VkDescriptorSetLayout		_GridOffsetDescSetLayout;
	VkPipeline					_GridOffsetPipeline;
	VkPipelineLayout			_GridOffsetPipelineLayout;

	VkCommandBuffer				_SurfelBinningCmdBuffer;
	VkSemaphore					_SurfelBinningSemaphore;
	VkDescriptorPool			_SurfelBinningDescPool;
	VkDescriptorSet				_SurfelBinningDescSet;
	VkDescriptorSetLayout		_SurfelBinningDescSetLayout;
	VkPipeline					_SurfelBinningPipeline;
	VkPipelineLayout			_SurfelBinningPipelineLayout;


	std::vector<VkRayTracingShaderGroupCreateInfoKHR> surfelShaderGroups{};
	
	VkDescriptorPool			_SurfelRTXDescPool;
	VkPipeline					_SurfelRTXPipeline;
	VkPipelineLayout			_SurfelRTXPipelineLayout;
	VkDescriptorSet				_SurfelRTXDescSet;
	VkDescriptorSetLayout		_SurfelRTXDescSetLayout;
	VkCommandBuffer				_SurfelRTXCommandBuffer;

	AllocatedBuffer				_SurfelRTXraygenSBT;
	AllocatedBuffer				_SurfelRTXmissSBT;
	AllocatedBuffer				_SurfelRTXhitSBT;

	VkCommandBuffer				_SurfelShadeCmdBuffer;
	VkSemaphore					_SurfelShadeSemaphore;
	VkDescriptorPool			_SurfelShadeDescPool;
	VkDescriptorSet				_SurfelShadeDescSet;
	VkDescriptorSetLayout		_SurfelShadeDescSetLayout;
	VkPipeline					_SurfelShadePipeline;
	VkPipelineLayout			_SurfelShadePipelineLayout;


	////Surfel GI
	AllocatedBuffer				_SurfelPositionBuffer;
	AllocatedBuffer				_SurfelBuffer;
	AllocatedBuffer				_SurfelDataBuffer;
	AllocatedBuffer				_SurfelStatsBuffer;
	AllocatedBuffer				_SurfelGridBuffer;
	AllocatedBuffer				_SurfelCellBuffer;
	Texture						_result;
	Texture						_debugGI;

	int shaderIf;

	void rasterize();

	void render();

	//void raytrace();

	//void rasterize_hybrid();

	void render_gui();

	void init_commands();

	void init_render_pass();

	void init_forward_render_pass();

	void init_offscreen_render_pass();

	FrameData& get_current_frame();

	void create_storage_image();

	void recreate_renderer();

	void buildTlas(const std::vector<TlasInstance>& input, VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, bool update = false);

private:

	void init_framebuffers();

	void init_offscreen_framebuffers();

	void init_sync_structures();

	void init_descriptors();

	void init_deferred_descriptors();

	//void init_gisurfels_descriptors();

	//void init_forward_pipeline();

	void init_deferred_pipelines();

	void build_forward_command_buffer();

	void build_previous_command_buffer();
	
	void build_deferred_command_buffer();

	void load_data_to_gpu();

	// VKRay

	void create_bottom_acceleration_structure();

	void create_top_acceleration_structure();

	void create_acceleration_structure(AccelerationStructure& accelerationStructure, 
		VkAccelerationStructureTypeKHR type, 
		VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo);

	void buildBlas(const std::vector<BlasInput>& input, VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	VkAccelerationStructureInstanceKHR object_to_instance(const TlasInstance& instance);

	void create_shadow_descriptors();

	void create_rt_descriptors();

	void create_shader_binding_table();

	void init_raytracing_pipeline();

	void init_compute_pipeline();

	//void build_raytracing_command_buffers();

	void build_shadow_command_buffer();

	void build_compute_command_buffer();

	void create_SurfelGi_resources();
	
	void surfel_position();

	void prepare_indirect();
	
	void grid_reset();

	void update_surfels();

	void grid_offset();

	void surfel_binning();

	void surfel_ray_tracing();

	void surfel_shade();


	//void transitionBufferLayout(VkBuffer buffer, size_t size, VkAccessFlagBits oldLayout, VkAccessFlagBits newLayout, VkCommandBuffer cmd);



	void create_surfel_position_descriptors();

	void init_surfel_position_pipeline();

	void build_surfel_position_command_buffer();

	void create_prepare_indirect_descriptors();

	void init_prepare_indirect_pipeline();

	void build_prepare_indirect_buffer();

	void create_grid_reset_descriptors();

	void init_grid_reset_pipeline();

	void build_grid_reset_buffer();

	void create_update_surfels_descriptors();

	void init_update_surfels_pipeline();

	void build_update_surfels_buffer();

	void create_grid_offset_descriptors();

	void init_grid_offset_pipeline();

	void build_grid_offset_buffer();

	void create_surfel_binning_descriptors();

	void init_surfel_binning_pipeline();

	void build_surfel_binning_buffer();


	void create_surfel_rtx_descriptors();

	void create_surfel_rtx_pipeline();

	void create_surfel_rtx_SBT();

	void create_surfel_rtx_cmd_buffer();


	void create_surfel_shade_descriptors();

	void init_surfel_shade_pipeline();

	void build_surfel_shade_buffer();

	// POST
	void create_post_renderPass();

	void create_post_framebuffers();

	void create_post_pipeline();

	void create_post_descriptor();

	void build_post_command_buffers();

	// HYBRID
	void create_hybrid_descriptors();

	uint32_t alignedSize(uint32_t value, uint32_t alignment);

	//void build_hybrid_command_buffers();
};