#pragma once

#include <vk_types.h>
#include <vk_textures.h>
#include "material.h"

struct VertexInputDescription{
	std::vector<VkVertexInputBindingDescription> bindings;
	std::vector<VkVertexInputAttributeDescription> attributes;

	VkPipelineVertexInputStateCreateFlags flags = 0;
};

struct rtVertexAttribute
{
	glm::vec4 position;
	glm::vec4 normal;
	glm::vec4 color;
	glm::vec4 uv;
	glm::vec4 tangent;
};

struct Vertex
{
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 uv;
	glm::vec4 tangent;

	static VertexInputDescription get_vertex_description();

	bool operator==(const Vertex& other) const {
		return position == other.position && color == other.color && uv == other.uv;
	}
};

struct BlasInput {
	VkAccelerationStructureGeometryKHR			asGeometry;
	VkAccelerationStructureBuildRangeInfoKHR	asBuildRangeInfo;
	uint32_t									nTriangles;
};

struct TlasInstance {
	uint32_t					blasId{ 0 };		// Index of the BLAS
	uint32_t					instanceId{ 0 };	// Instance index
	uint32_t					hitGroupId{ 0 };	// Hit group index in the SBT
	uint32_t					mask{ 0xFF };		// Visibility mask
	VkGeometryInstanceFlagsKHR	flags{ VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR };
	glm::mat4					transform{ glm::mat4(1) };	// Identity model matrix
};

struct material_matrix {
	glm::mat4 matrix;
	int material;
};

struct Primitive
{
	uint32_t firstIndex{ 0 };
	uint32_t indexCount{ 0 };
	uint32_t firstVertex{ 0 };
	uint32_t vertexCount{ 0 };
	int32_t	materialID;
	int32_t	instanceID;
	int32_t	transformID;

};

struct Mesh
{
	static std::unordered_map<std::string, Mesh*> _loadedMeshes;
	std::vector<Vertex>		_vertices;
	std::vector<uint32_t>	_indices;
	
	AllocatedBuffer			_vertexBuffer;
	AllocatedBuffer			_indexBuffer;

	static Mesh* GET(const char* filename);

	static Mesh* get_quad();
	static Mesh* get_triangle();
	static Mesh* get_cube();

	void upload();
	BlasInput mesh_to_geometry();

private:

	bool load_from_obj(const char* filename);
	void create_vertex_buffer();
	void create_index_buffer();
};

class Node
{
public:
	std::string				_name;
	Node*					_parent;
	std::vector<Node*>		_children;

	std::vector<Primitive*>	_primitives;
	glm::mat4				_matrix;
	glm::mat4				_global_matrix;

	Node() { _matrix = glm::mat4(1); }

	void addChild(Node* child);
	glm::mat4 getGlobalMatrix(bool fast = false);

	void node_to_geometry(
		std::vector<BlasInput>& blasVector,
		const VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress, 
		const VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress);
	void node_to_instance(
		std::vector<TlasInstance>& instances, 
		int& index, 
		glm::mat4 model);

	unsigned int get_number_nodes();
	void fill_matrix_buffer(std::vector<glm::mat4>& buffer, const glm::mat4 model);
	void fill_index_buffer(std::vector<glm::vec4>& index_buffer);
	void addMaterial(Material* mat);
};

namespace tinygltf {
	class Node;
	class Model;
};

class Prefab
{
public:
	static std::unordered_map<std::string, Prefab*> _prefabsMap;

	std::string			_name;
	std::vector<Node*>	_root;
	Mesh*				_mesh = NULL;

	static Prefab* GET(std::string filename, const bool invertNormals = false);
	static Prefab* GET(std::string name, Mesh* mesh);
	void draw(VkCommandBuffer& cmd, VkPipelineLayout pipelineLayout, glm::mat4& model);
	BlasInput primitive_to_geometry(const Primitive& prim);

private:

	void loadNode(const tinygltf::Model& tmodel, const tinygltf::Node& tnode, Node* parent, const bool invertNormals = false);
	int loadMaterial(const tinygltf::Model& tmodel, const int index);
	void loadTextures(const tinygltf::Model&, const int index);
	void drawNode(VkCommandBuffer& cmd, VkPipelineLayout pipelineLayout, Node& node, glm::mat4& model);
	void createOBJprefab(Mesh* mesh = NULL);
	glm::mat4 get_local_matrix(const tinygltf::Node& tnode);
};