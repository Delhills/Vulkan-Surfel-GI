
#include "vk_mesh.h"
#include "vk_initializers.h"
#include "vk_engine.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

extern std::vector<std::string> searchPaths;
std::unordered_map<std::string, Mesh*> Mesh::_loadedMeshes;
std::unordered_map<std::string, Prefab*> Prefab::_prefabsMap;

VertexInputDescription Vertex::get_vertex_description()
{
	VertexInputDescription description;

	VkVertexInputBindingDescription mainBinding = {};
	mainBinding.binding			= 0;
	mainBinding.stride			= sizeof(Vertex);
	mainBinding.inputRate		= VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	// Position will be stored at locaiton 0
	VkVertexInputAttributeDescription positionAttribute = {};
	positionAttribute.binding	= 0;
	positionAttribute.location	= 0;
	positionAttribute.format	= VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset	= offsetof(Vertex, position);

	// Normal will be stores at location 1
	VkVertexInputAttributeDescription normalAttribute{};
	normalAttribute.binding		= 0;
	normalAttribute.location	= 1;
	normalAttribute.format		= VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset		= offsetof(Vertex, normal);

	// Color will be stored at location 2
	VkVertexInputAttributeDescription colorAttribute = {};
	colorAttribute.binding		= 0;
	colorAttribute.location		= 2;
	colorAttribute.format		= VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset		= offsetof(Vertex, color);

	// UV will be stored at location 3
	VkVertexInputAttributeDescription uvAttribute = {};
	uvAttribute.binding			= 0;
	uvAttribute.location		= 3;
	uvAttribute.format			= VK_FORMAT_R32G32_SFLOAT;
	uvAttribute.offset			= offsetof(Vertex, uv);

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(colorAttribute);
	description.attributes.push_back(uvAttribute);

	return description;
}

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((((hash<glm::vec3>()(vertex.position) ^
				(hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.uv) << 1);
		}
	};
}

Mesh* Mesh::GET(const char* filename)
{
	Mesh* mesh = new Mesh();
	std::string s = filename;
	std::string name = VulkanEngine::engine->findFile(s, searchPaths, true);
	
	if(!_loadedMeshes[name])
	{
		mesh->load_from_obj(name.c_str());
		_loadedMeshes[name] = mesh;
	}
	else
	{
		mesh = _loadedMeshes[name];
	}
	return mesh;
}

bool Mesh::load_from_obj(const char* filename)
{
	tinyobj::attrib_t attrib;

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn, err;

	tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, nullptr);

	if (!warn.empty())
		std::cout << "WARN: " << warn << std::endl;

	if (!err.empty()) {
		std::cout << "ERR: " << err << std::endl;
		return false;
	}

	std::unordered_map<Vertex, uint32_t> uniqueVertices{};

	for (const auto& shape : shapes)
	{
		for (const auto& index : shape.mesh.indices)
		{
			Vertex vertex{};

			vertex.position = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.normal = {
				attrib.normals[3 * index.normal_index + 0],
				attrib.normals[3 * index.normal_index + 1],
				attrib.normals[3 * index.normal_index + 2]
			};

			vertex.color = { 1.0f, 1.0f, 1.0f };

			if (attrib.texcoords.size() > 0)
			{
				vertex.uv = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};
			}


			if (uniqueVertices.count(vertex) == 0) {
				uniqueVertices[vertex] = static_cast<uint32_t>(_vertices.size());
				_vertices.push_back(vertex);
			}

			_indices.push_back(uniqueVertices[vertex]);
		}
	}

	// upload mesh
	upload();

	return true;
}

Mesh* Mesh::get_quad()
{

	if (!_loadedMeshes["quad"])
	{
		Mesh* mesh = new Mesh();

		mesh->_vertices.push_back({ {  1.0f,  1.0f, 0.0f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f} });
		mesh->_vertices.push_back({ { -1.0f,  1.0f, 0.0f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f} });
		mesh->_vertices.push_back({ { -1.0f, -1.0f, 0.0f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 0.0f} });
		mesh->_vertices.push_back({ {  1.0f, -1.0f, 0.0f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f} });

		mesh->_indices = {0, 1, 2, 2, 3, 0};

		mesh->upload();

		_loadedMeshes["quad"] = mesh;

		return mesh;
	}

	return _loadedMeshes["quad"];
}

Mesh* Mesh::get_triangle()
{
	if (!_loadedMeshes["triangle"])
	{
		Mesh* mesh = new Mesh();
		
		mesh->_vertices.clear();

		mesh->_vertices.push_back({ {  1.f,  1.f, 0.f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f} });
		mesh->_vertices.push_back({ { -1.f,  1.f, 0.f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f} });
		mesh->_vertices.push_back({ {  0.f, -1.f, 0.f }, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 0.0f} });

		mesh->_indices = { 0, 1, 2 };

		mesh->upload();
		_loadedMeshes["triangle"] = mesh;

		return mesh;
	}
	
	return _loadedMeshes["triangle"];
}

Mesh* Mesh::get_cube()
{
	if (!_loadedMeshes["cube"])
	{
		Mesh* mesh = new Mesh();

		mesh->_vertices.clear();

		mesh->_vertices.push_back({ { 1.0,  1.0, -1.0}, {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0,  1.0, -1.0}, {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0, -1.0, -1.0}, {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ { 1.0, -1.0, -1.0}, {0.0, 0.0, -1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });

		mesh->_vertices.push_back({ { 1.0,  1.0,  1.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0,  1.0,  1.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0, -1.0,  1.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ { 1.0, -1.0,  1.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
	
		mesh->_vertices.push_back({ { 1.0,  1.0,  1.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0,  1.0,  1.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0,  1.0, -1.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ { 1.0,  1.0, -1.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
	
		mesh->_vertices.push_back({ { 1.0, -1.0,  1.0}, {0.0, -1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0, -1.0,  1.0}, {0.0, -1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0, -1.0, -1.0}, {0.0, -1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ { 1.0, -1.0, -1.0}, {0.0, -1.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
	
		mesh->_vertices.push_back({ {-1.0,  1.0,  1.0}, {-1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0, -1.0,  1.0}, {-1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0, -1.0, -1.0}, {-1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {-1.0,  1.0, -1.0}, {-1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
	
		mesh->_vertices.push_back({ {1.0,  1.0,  1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {1.0, -1.0,  1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {1.0, -1.0, -1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });
		mesh->_vertices.push_back({ {1.0,  1.0, -1.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {1.0, 1.0} });

		mesh->_indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20 
		};

		// upload mesh
		mesh->upload();
		_loadedMeshes["cube"] = mesh;

		return mesh;
	}
	
	return _loadedMeshes["cube"];
}

void Mesh::create_vertex_buffer()
{
	const size_t bufferSize = _vertices.size() * sizeof(Vertex);

	VkBufferCreateInfo stagingBufferInfo = vkinit::buffer_create_info(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	VmaAllocationCreateInfo vmaAllocInfo = {};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBuffer stagingBuffer;

	VK_CHECK(vmaCreateBuffer(VulkanEngine::engine->_allocator, 
		&stagingBufferInfo, &vmaAllocInfo, 
		&stagingBuffer._buffer, 
		&stagingBuffer._allocation, 
		nullptr));

	// Copy Vertex data
	void* data;
	vmaMapMemory(VulkanEngine::engine->_allocator, stagingBuffer._allocation, &data);
	memcpy(data, _vertices.data(), _vertices.size() * sizeof(Vertex));
	vmaUnmapMemory(VulkanEngine::engine->_allocator, stagingBuffer._allocation);

	VkBufferCreateInfo vertexBufferInfo = vkinit::buffer_create_info(bufferSize,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

	vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VK_CHECK(vmaCreateBuffer(VulkanEngine::engine->_allocator, &vertexBufferInfo, &vmaAllocInfo,
		&_vertexBuffer._buffer,
		&_vertexBuffer._allocation,
		nullptr));

	// Copy vertex data
	VulkanEngine::engine->immediate_submit([=](VkCommandBuffer cmd) {
		VkBufferCopy copy;
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = bufferSize;
		vkCmdCopyBuffer(cmd, stagingBuffer._buffer, _vertexBuffer._buffer, 1, &copy);
		});

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vmaDestroyBuffer(VulkanEngine::engine->_allocator, this->_vertexBuffer._buffer, this->_vertexBuffer._allocation);
		});

	vmaDestroyBuffer(VulkanEngine::engine->_allocator, stagingBuffer._buffer, stagingBuffer._allocation);

}

void Mesh::create_index_buffer()
{
	const size_t bufferSize = _indices.size() * sizeof(uint32_t);
	VkBufferCreateInfo stagingBufferInfo = vkinit::buffer_create_info(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

	VmaAllocationCreateInfo vmaAllocInfo = {};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBuffer stagingBuffer;

	VK_CHECK(vmaCreateBuffer(VulkanEngine::engine->_allocator, &stagingBufferInfo, &vmaAllocInfo,
		&stagingBuffer._buffer,
		&stagingBuffer._allocation,
		nullptr));

	void* data;
	vmaMapMemory(VulkanEngine::engine->_allocator, stagingBuffer._allocation, &data);
	memcpy(data, _indices.data(), _indices.size() * sizeof(uint32_t));
	vmaUnmapMemory(VulkanEngine::engine->_allocator, stagingBuffer._allocation);

	vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	VkBufferCreateInfo indexBufferInfo = vkinit::buffer_create_info(bufferSize,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	VK_CHECK(vmaCreateBuffer(VulkanEngine::engine->_allocator, &indexBufferInfo, &vmaAllocInfo,
		&_indexBuffer._buffer,
		&_indexBuffer._allocation,
		nullptr));

	// Copy index data
	VulkanEngine::engine->immediate_submit([=](VkCommandBuffer cmd) {
		VkBufferCopy copy;
		copy.dstOffset = 0;
		copy.srcOffset = 0;
		copy.size = bufferSize;
		vkCmdCopyBuffer(cmd, stagingBuffer._buffer, _indexBuffer._buffer, 1, &copy);
		});

	VulkanEngine::engine->_mainDeletionQueue.push_function([=]() {
		vmaDestroyBuffer(VulkanEngine::engine->_allocator, this->_indexBuffer._buffer, this->_indexBuffer._allocation);
		});
	
	vmaDestroyBuffer(VulkanEngine::engine->_allocator, stagingBuffer._buffer, stagingBuffer._allocation);
}

void Mesh::upload()
{
	create_vertex_buffer();
	create_index_buffer();
}

BlasInput Mesh::mesh_to_geometry()
{
	VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
	VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

	vertexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(_vertexBuffer._buffer);
	indexBufferDeviceAddress.deviceAddress	= VulkanEngine::engine->getBufferDeviceAddress(_indexBuffer._buffer);

	const uint32_t nTriangles = _indices.size() / 3;

	// Set the triangles geometry
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
	triangles.sType			= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	triangles.pNext			= nullptr;
	triangles.vertexFormat	= VK_FORMAT_R32G32B32_SFLOAT;
	triangles.vertexData	= vertexBufferDeviceAddress;
	triangles.vertexStride	= sizeof(Vertex);
	triangles.maxVertex		= static_cast<uint32_t>(_vertices.size());
	triangles.indexData		= indexBufferDeviceAddress;
	triangles.indexType		= VK_INDEX_TYPE_UINT32;

	VkAccelerationStructureGeometryKHR asGeometry = vkinit::acceleration_structure_geometry_khr();
	asGeometry.flags				= VK_GEOMETRY_OPAQUE_BIT_KHR;
	asGeometry.geometryType			= VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	asGeometry.geometry.triangles	= triangles;

	VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};
	asBuildRangeInfo.primitiveCount		= nTriangles;
	asBuildRangeInfo.primitiveOffset	= 0;
	asBuildRangeInfo.firstVertex		= 0;
	asBuildRangeInfo.transformOffset	= 0;

	// Store all info in the BlasInput structure to be returned
	BlasInput input;
	input.asBuildRangeInfo	= asBuildRangeInfo;
	input.asGeometry		= asGeometry;
	input.nTriangles		= nTriangles;

	return input;
}

void Node::addChild(Node* child)
{
	assert(child->_parent == NULL);
	_children.push_back(child);
	child->_parent = this;
}

glm::mat4 Node::getGlobalMatrix(bool fast)
{
	if (_parent)
		_global_matrix = (fast ? _parent->_matrix : _parent->getGlobalMatrix()) * _matrix;
	else
		_global_matrix = _matrix;
	return _global_matrix;
}

std::vector<BlasInput> Node::node_to_geometry(
	const VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress,
	const VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress)
{
	std::vector<BlasInput> blasVector;
	//blasVector.reserve(_primitives.size());
	for (Primitive& p : _primitives)
	{
		const uint32_t nTriangles = p.indexCount / 3;

		// Set the triangles geometry
		VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
		triangles.sType						= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		triangles.pNext						= nullptr;
		triangles.vertexFormat				= VK_FORMAT_R32G32B32_SFLOAT;
		triangles.vertexData				= vertexBufferDeviceAddress;
		triangles.vertexStride				= sizeof(Vertex);
		triangles.maxVertex					= static_cast<uint32_t>(p.vertexCount);
		triangles.indexData					= indexBufferDeviceAddress;
		triangles.indexType					= VK_INDEX_TYPE_UINT32;

		VkAccelerationStructureGeometryKHR asGeometry = vkinit::acceleration_structure_geometry_khr();
		asGeometry.flags					= VK_GEOMETRY_OPAQUE_BIT_KHR;
		asGeometry.geometryType				= VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		asGeometry.geometry.triangles		= triangles;

		VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};
		asBuildRangeInfo.firstVertex		= p.firstVertex;
		asBuildRangeInfo.primitiveCount		= nTriangles;
		asBuildRangeInfo.primitiveOffset	= p.firstIndex * sizeof(uint32_t);
		asBuildRangeInfo.transformOffset	= 0;

		// Store all info in the BlasInput structure to be returned
		BlasInput input;
		input.asBuildRangeInfo	= asBuildRangeInfo;
		input.asGeometry		= asGeometry;
		input.nTriangles		= nTriangles;

		blasVector.emplace_back(input);
	}
	if (!_children.empty())
	{
		for (Node* child : _children)
		{
			std::vector<BlasInput> childBlas = child->node_to_geometry(vertexBufferDeviceAddress, indexBufferDeviceAddress);
			blasVector.insert(blasVector.end(), childBlas.begin(), childBlas.end());
		}
		//return blasVector;
	}
	return blasVector;
}

std::vector<TlasInstance> Node::node_to_instance(int& index, glm::mat4 model)
{
	std::vector<TlasInstance> instances;
	glm::mat4 matrix = model * getGlobalMatrix(false);
	if (!_primitives.empty())
	{
		TlasInstance instance{};
		instance.transform	= matrix;
		instance.instanceId = index;
		instance.blasId		= index;
		instances.emplace_back(instance);
		index++;
	}
	for (Node* child : _children)
	{
		std::vector<TlasInstance> childInstances = child->node_to_instance(index, model);
		instances.insert(instances.begin(), childInstances.begin(), childInstances.end());
	}
	return instances;
}

BlasInput Prefab::primitive_to_geometry(const Primitive& p)
{
	VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
	VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

	vertexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(_mesh->_vertexBuffer._buffer);
	indexBufferDeviceAddress.deviceAddress = VulkanEngine::engine->getBufferDeviceAddress(_mesh->_indexBuffer._buffer);

	const uint32_t nTriangles = p.indexCount / 3;

	// Set the triangles geometry
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
	triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	triangles.pNext = nullptr;
	triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	triangles.vertexData = vertexBufferDeviceAddress;
	triangles.vertexStride = sizeof(Vertex);
	triangles.maxVertex = static_cast<uint32_t>(p.vertexCount);
	triangles.indexData = indexBufferDeviceAddress;
	triangles.indexType = VK_INDEX_TYPE_UINT32;

	VkAccelerationStructureGeometryKHR asGeometry = vkinit::acceleration_structure_geometry_khr();
	asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	asGeometry.geometry.triangles = triangles;

	VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};
	asBuildRangeInfo.firstVertex = p.firstVertex;
	asBuildRangeInfo.primitiveCount = nTriangles;
	asBuildRangeInfo.primitiveOffset = p.firstIndex * sizeof(uint32_t);
	asBuildRangeInfo.transformOffset = 0;

	// Store all info in the BlasInput structure to be returned
	BlasInput input;
	input.asBuildRangeInfo = asBuildRangeInfo;
	input.asGeometry = asGeometry;
	input.nTriangles = nTriangles;

	return input;
}

void Prefab::drawNode(VkCommandBuffer& cmd, VkPipelineLayout pipelineLayout, Node node, glm::mat4& model)
{
	if (node._primitives.size() > 0)
	{
		glm::mat4 node_matrix = model * node.getGlobalMatrix(false);

		for (Primitive& prim : node._primitives)
		{
			GPUMaterial mat;
			mat.diffuseColor = _materials[prim.materialIndex].diffuseColor;
			mat.textures = glm::vec4(_materials[prim.materialIndex].diffuseTexture, _materials[prim.materialIndex].normalTexture, _materials[prim.materialIndex].emissiveTexture, _materials[prim.materialIndex].metallicRoughnessTexture);
			mat.shadingMetallicRoughness = glm::vec4(_materials[prim.materialIndex].shadingModel, _materials[prim.materialIndex].metallicFactor, _materials[prim.materialIndex].roughnessFactor, 0);

			if (prim.indexCount > 0) 
			{

				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &node_matrix);
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(GPUMaterial), &mat);
				vkCmdDrawIndexed(cmd, prim.indexCount, 1, prim.firstIndex, 0, 0);
			}
		}
	}

	for(auto& child : node._children)
		drawNode(cmd, pipelineLayout, *child, model);
}

void Prefab::drawGltfNode(VkCommandBuffer& cmd, VkPipelineLayout pipelineLayout, gltfNode node, glm::mat4& model)
{
	Primitive& prim = _primitives[node.primMesh];

	GPUMaterial mat;
	mat.diffuseColor = _materials[prim.materialIndex].diffuseColor;
	mat.textures = glm::vec4(_materials[prim.materialIndex].diffuseTexture, _materials[prim.materialIndex].normalTexture, _materials[prim.materialIndex].emissiveTexture, _materials[prim.materialIndex].metallicRoughnessTexture);
	mat.shadingMetallicRoughness = glm::vec4(_materials[prim.materialIndex].shadingModel, _materials[prim.materialIndex].metallicFactor, _materials[prim.materialIndex].roughnessFactor, 0);

	model = model * node.matrix;

	vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &model);
	vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(GPUMaterial), &mat);
	vkCmdDrawIndexed(cmd, prim.indexCount, 1, prim.firstIndex, 0, 0);
}

glm::mat4 Prefab::get_local_matrix(const tinygltf::Node& inputNode)
{
	glm::mat4 matrix = glm::mat4(1);
	if (inputNode.translation.size() == 3)
		matrix = glm::translate(matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
	if (inputNode.rotation.size() == 4)
	{
		glm::quat q = glm::make_quat(inputNode.rotation.data());
		matrix *= glm::mat4(q);
	}
	if (inputNode.scale.size() == 3)
	{
		matrix = glm::scale(matrix, glm::vec3(glm::make_vec3(inputNode.scale.data())));
	}
	if (inputNode.matrix.size() == 16)
	{
		matrix = glm::make_mat4x4(inputNode.matrix.data());
	}
	return matrix;
}

void Prefab::draw(VkCommandBuffer& cmd, VkPipelineLayout pipelineLayout, glm::mat4& model)
{
	VkDeviceSize offset{ 0 };
	vkCmdBindVertexBuffers(cmd, 0, 1, &_mesh->_vertexBuffer._buffer, &offset);
	vkCmdBindIndexBuffer(cmd, _mesh->_indexBuffer._buffer, offset, VK_INDEX_TYPE_UINT32);
	if (_nodes.empty())
	{
		GPUMaterial mat;
		mat.diffuseColor				= _materials[0].diffuseColor;
		mat.textures					= glm::vec4(_materials[0].diffuseTexture, _materials[0].normalTexture, _materials[0].emissiveTexture, _materials[0].metallicRoughnessTexture);
		mat.shadingMetallicRoughness	= glm::vec4(_materials[0].shadingModel, _materials[0].metallicFactor, _materials[0].roughnessFactor, 0);

		vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &model);
		vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(GPUMaterial), &mat);
		vkCmdDrawIndexed(cmd, static_cast<uint32_t>(_mesh->_indices.size()), 1, 0, 0, 0);
	}
	for (Node* node : _nodes)
	{
		drawNode(cmd, pipelineLayout, *node, model);
		//drawGltfNode(cmd, pipelineLayout, node, model);
	}
}

void Prefab::loadNode(const tinygltf::Model& tmodel, const tinygltf::Node& tnode, Node* parent, const bool invertNormals)
{
	// Init node and compute its local matrix
	Node* node = new Node();
	node->_matrix = get_local_matrix(tnode);

	// Load node's children
	if (tnode.children.size() > 0)
	{
		for (size_t i = 0; i < tnode.children.size(); i++)
		{
			loadNode(tmodel, tmodel.nodes[tnode.children[i]], node);
		}
	}

	// If node contains mesh data, load vertices and indices from the buffers
	if (tnode.mesh > -1)
	{
		const tinygltf::Mesh mesh = tmodel.meshes[tnode.mesh];
		// Iterate through all primitives in mesh
		for (size_t i = 0; i < mesh.primitives.size(); i++)
		{
			const tinygltf::Primitive& tprimitive = mesh.primitives[i];
			uint32_t firstIndex = static_cast<uint32_t>(_mesh->_indices.size());
			uint32_t firstVertex = static_cast<uint32_t>(_mesh->_vertices.size());
			uint32_t indexCount = 0;
			uint32_t vertexCount = 0;

			//Vertices
			{
				const float* positionBuffer = nullptr;
				const float* normalsBuffer = nullptr;
				const float* texCoordsBuffer = nullptr;

				// Get buffer data for vertex position
				if (tprimitive.attributes.find("POSITION") != tprimitive.attributes.end()) {
					const tinygltf::Accessor& accessor = tmodel.accessors[tprimitive.attributes.find("POSITION")->second];
					const tinygltf::BufferView& view = tmodel.bufferViews[accessor.bufferView];
					positionBuffer = reinterpret_cast<const float*>(&(tmodel.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					vertexCount = accessor.count;
				}
				// Get buffer data for vertex normals
				if (tprimitive.attributes.find("NORMAL") != tprimitive.attributes.end()) {
					const tinygltf::Accessor& accessor = tmodel.accessors[tprimitive.attributes.find("NORMAL")->second];
					const tinygltf::BufferView& view = tmodel.bufferViews[accessor.bufferView];
					normalsBuffer = reinterpret_cast<const float*>(&(tmodel.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}
				// Get buffer data for vertex texture coordinates
				// glTF supports multiple sets, we only load the first one
				if (tprimitive.attributes.find("TEXCOORD_0") != tprimitive.attributes.end()) {
					const tinygltf::Accessor& accessor = tmodel.accessors[tprimitive.attributes.find("TEXCOORD_0")->second];
					const tinygltf::BufferView& view = tmodel.bufferViews[accessor.bufferView];
					texCoordsBuffer = reinterpret_cast<const float*>(&(tmodel.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}

				// Append data to model's vertex buffer
				for (size_t v = 0; v < vertexCount; v++) 
				{
					glm::vec3 normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
					Vertex vert{};
					vert.position	= glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
					vert.normal		= invertNormals ? normal * glm::vec3(-1) : normal;
					vert.uv			= texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
					vert.color		= glm::vec3(1.0f);
					_mesh->_vertices.push_back(vert);
				}
			}
			//Indices
			{
				const tinygltf::Accessor& accessor		= tmodel.accessors[tprimitive.indices];
				const tinygltf::BufferView& bufferView	= tmodel.bufferViews[accessor.bufferView];
				const tinygltf::Buffer& buffer			= tmodel.buffers[bufferView.buffer];

				indexCount += static_cast<uint32_t>(accessor.count);

				// glTF supports different component types of indices
				switch (accessor.componentType) {
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
					uint32_t* buf = new uint32_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(uint32_t));
					for (size_t index = 0; index < accessor.count; index++) {
						_mesh->_indices.push_back(buf[index] + firstVertex);
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
					uint16_t* buf = new uint16_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(uint16_t));
					for (size_t index = 0; index < accessor.count; index++) {
						_mesh->_indices.push_back(buf[index] + firstVertex);
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
					uint8_t* buf = new uint8_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset], accessor.count * sizeof(uint8_t));
					for (size_t index = 0; index < accessor.count; index++) {
						_mesh->_indices.push_back(buf[index] + firstVertex);
					}
					break;
				}
				default:
					std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
					return;
				}
			}

			// Load the primitive information
			Primitive prim{};
			prim.firstIndex		= firstIndex;
			prim.indexCount		= indexCount;
			prim.firstVertex	= firstVertex;
			prim.vertexCount	= vertexCount;
			prim.materialIndex	= std::max(0, tprimitive.material);
			node->_primitives.push_back(prim);
		}
	}

	// Add the node to the parent's children list if it has a parent
	if (parent)
	{
		node->_parent = parent;
		parent->_children.push_back(node);
	}
	else
	{
		_nodes.push_back(node);
	}
}

void Prefab::importNode(const tinygltf::Model& inputModel)
{

	uint32_t nVert{ 0 };
	uint32_t nIndex{ 0 };
	uint32_t meshCount{ 0 };
	uint32_t primCount{ 0 };
	_mesh = new Mesh();
	for (const auto& mesh : inputModel.meshes)
	{
		std::vector<uint32_t> vprim;
		for (const auto& primitive : mesh.primitives)
		{
			if (primitive.mode != 4) // if not triangle
				continue;
			const auto& posAccessor = inputModel.accessors[primitive.attributes.find("POSITION")->second];
			nVert += static_cast<uint32_t>(posAccessor.count);
			const auto& indexAccessor = inputModel.accessors[primitive.indices];
			nIndex += static_cast<uint32_t>(indexAccessor.count);
			vprim.emplace_back(primCount++);
		}
		_meshToPrimMeshes[meshCount++] = std::move(vprim);
	}

	_mesh->_vertices.reserve(nVert * sizeof(Vertex));
	_mesh->_indices.reserve(nIndex);

	processMesh(inputModel);

	for (auto nodeIdx : inputModel.scenes[0].nodes)
	{
		processNode(inputModel, nodeIdx, glm::mat4(1));
	}

	_meshToPrimMeshes.clear();
}

void Prefab::importMaterials(const tinygltf::Model& tmodel)
{
	_materials.reserve(tmodel.materials.size());

	for (auto& tmat : tmodel.materials)
	{
		GltfMaterial mat;
		mat.emissiveTexture				= tmat.emissiveTexture.index;
		mat.normalTexture				= tmat.normalTexture.index;
		
		auto& tpbr = tmat.pbrMetallicRoughness;
		mat.diffuseColor				= glm::vec4(tpbr.baseColorFactor[0], tpbr.baseColorFactor[1], tpbr.baseColorFactor[2], tpbr.baseColorFactor[3]);
		mat.metallicFactor				= tpbr.metallicFactor;
		mat.roughnessFactor				= tpbr.roughnessFactor;
		mat.diffuseTexture				= tpbr.baseColorTexture.index;
		mat.metallicRoughnessTexture	= tpbr.metallicRoughnessTexture.index;

		_materials.emplace_back(mat);
	}
	
	// default material
	if (_materials.empty())
	{
		GltfMaterial mat;
		mat.metallicFactor = 0;
		_materials.emplace_back(mat);
	}
}

void Prefab::importTextures(const tinygltf::Model& tmodel)
{
	for (auto& mat : _materials)
	{
		if (mat.diffuseTexture > -1)
		{
			Texture::GET(tmodel.images[mat.diffuseTexture].uri.c_str());
			mat.diffuseTexture = Texture::get_id(tmodel.images[mat.diffuseTexture].uri.c_str());
		}
		if (mat.normalTexture > -1)
		{
			Texture::GET(tmodel.images[mat.normalTexture].uri.c_str());
			mat.normalTexture = Texture::get_id(tmodel.images[mat.normalTexture].uri.c_str());
		}
		if (mat.emissiveTexture > -1)
		{
			Texture::GET(tmodel.images[mat.emissiveTexture].uri.c_str());
			mat.emissiveTexture = Texture::get_id(tmodel.images[mat.emissiveTexture].uri.c_str());
		}
		if (mat.metallicRoughnessTexture > -1)
		{
			Texture::GET(tmodel.images[mat.metallicRoughnessTexture].uri.c_str());
			mat.metallicRoughnessTexture = Texture::get_id(tmodel.images[mat.metallicRoughnessTexture].uri.c_str());
		}
	}
}

void Prefab::processNode(const tinygltf::Model& tmodel, int& nodeIdx, const glm::mat4 parent_matrix)
{
	const auto& tnode = tmodel.nodes[nodeIdx];

	glm::mat4 matrix = get_local_matrix(tnode);
	glm::mat4 worldMatrix = parent_matrix * matrix;

	if (tnode.mesh > -1)
	{
		const auto& meshes = _meshToPrimMeshes[tnode.mesh];
		for (const auto& mesh : meshes)
		{
			gltfNode node;
			node.matrix		= worldMatrix;
			node.primMesh	= mesh;
			_gltfNodes.emplace_back(node);
		}
	}

	for (auto child : tnode.children)
	{
		processNode(tmodel, child, worldMatrix);
	}
}

void Prefab::processMesh(const tinygltf::Model& input)
{
	for (const auto& mesh : input.meshes)
	{
		for (size_t i = 0; i < mesh.primitives.size(); i++)
		{
			const tinygltf::Primitive& glTFPrimitive	= mesh.primitives[i];
			uint32_t                   firstIndex		= static_cast<uint32_t>(_mesh->_indices.size());
			uint32_t                   vertexStart		= static_cast<uint32_t>(_mesh->_vertices.size());
			uint32_t                   indexCount		= 0;
			uint32_t				   vCount			= 0;
			// Vertices
			{
				const float* positionBuffer = nullptr;
				const float* normalsBuffer = nullptr;
				const float* texCoordsBuffer = nullptr;
				size_t       vertexCount = 0;

				// Get buffer data for vertex normals
				if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor& accessor =
						input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
					const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
					positionBuffer = reinterpret_cast<const float*>(
						&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					vertexCount = accessor.count;
					vCount = vertexCount;
				}
				// Get buffer data for vertex normals
				if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor& accessor =
						input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
					const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
					normalsBuffer = reinterpret_cast<const float*>(
						&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}
				// Get buffer data for vertex texture coordinates
				// glTF supports multiple sets, we only load the first one
				if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor& accessor =
						input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
					const tinygltf::BufferView& view = input.bufferViews[accessor.bufferView];
					texCoordsBuffer = reinterpret_cast<const float*>(
						&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}

				// Append data to model's vertex buffer
				for (size_t v = 0; v < vertexCount; v++)
				{
					Vertex vert{};
					vert.position = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
					vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
					vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
					vert.color = glm::vec3(1.0f);
					_mesh->_vertices.push_back(vert);
				}
			}
			// Indices
			{
				const tinygltf::Accessor& accessor = input.accessors[glTFPrimitive.indices];
				const tinygltf::BufferView& bufferView = input.bufferViews[accessor.bufferView];
				const tinygltf::Buffer& buffer = input.buffers[bufferView.buffer];

				indexCount = static_cast<uint32_t>(accessor.count);

				// glTF supports different component types of indices
				switch (accessor.componentType)
				{
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
					uint32_t* buf = new uint32_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset],
						accessor.count * sizeof(uint32_t));
					for (size_t index = 0; index < accessor.count; index++)
					{
						_mesh->_indices.push_back(buf[index] + vertexStart);
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
					uint16_t* buf = new uint16_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset],
						accessor.count * sizeof(uint16_t));
					for (size_t index = 0; index < accessor.count; index++)
					{
						_mesh->_indices.push_back(buf[index] + vertexStart);
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
					uint8_t* buf = new uint8_t[accessor.count];
					memcpy(buf, &buffer.data[accessor.byteOffset + bufferView.byteOffset],
						accessor.count * sizeof(uint8_t));
					for (size_t index = 0; index < accessor.count; index++)
					{
						_mesh->_indices.push_back(buf[index] + vertexStart);
					}
					break;
				}
				default:
					std::cerr << "Index component type " << accessor.componentType << " not supported!"
						<< std::endl;
					return;
				}
			}
			Primitive primitive{};
			primitive.firstIndex	= firstIndex;
			primitive.indexCount	= indexCount;
			primitive.firstVertex	= vertexStart;
			primitive.vertexCount	= vCount;
			primitive.materialIndex = std::max(0, glTFPrimitive.material);
			_primitives.push_back(primitive);
		}
	}
}

Prefab* Prefab::GET(const std::string filename, bool invertNormals)
{
	std::string name = VulkanEngine::engine->findFile(filename, searchPaths, true);
	if (!_prefabsMap[name])
	{
		Prefab* prefab = new Prefab();
		if (filename.find("obj") != std::string::npos) 
		{
			prefab->_mesh = Mesh::GET(name.c_str());
			prefab->_materials.push_back(GltfMaterial());
			_prefabsMap[name] = prefab;
			return prefab;
		}
		else if (filename.find(".gltf") != std::string::npos)
		{
			std::cout << "Loading gltf... " << filename << std::endl;

			tinygltf::Model		gltfModel;
			tinygltf::TinyGLTF	gltfContext;
			std::string			warn, err;
			bool				fileLoaded{ false };

			fileLoaded = gltfContext.LoadASCIIFromFile(&gltfModel, &err, &warn, VulkanEngine::engine->findFile(filename, searchPaths, true));

			if (!err.empty())
				throw std::runtime_error(err.c_str());

			if (fileLoaded)
			{
				// TODO: import materials
				const tinygltf::Scene& scene = gltfModel.scenes[0];
				prefab->_mesh = new Mesh();

				//prefab->importNode(gltfModel);
				for(const int node : scene.nodes)
					prefab->loadNode(gltfModel, gltfModel.nodes[node], nullptr, invertNormals);
				prefab->importMaterials(gltfModel);
				prefab->importTextures(gltfModel);

				prefab->_mesh->upload();

				_prefabsMap[name] = prefab;
				return prefab;
			}
		}
	}

	return _prefabsMap[name];
}