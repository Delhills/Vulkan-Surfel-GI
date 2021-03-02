
#include "vk_mesh.h"
#include "vk_initializers.h"
#include "vk_engine.h"
#include "vk_utils.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

extern std::vector<std::string> searchPaths;
std::unordered_map<std::string, Mesh*> Mesh::_loadedMeshes;
std::unordered_map<std::string, Prefab*> Prefab::_prefabsMap;
std::vector<Material*> Material::_materials;

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
	std::string s = filename;
	std::string name = vkutil::findFile(s, searchPaths, true);
	
	if(!_loadedMeshes[name])
	{
		Mesh* mesh = new Mesh();
		mesh->load_from_obj(name.c_str());
		_loadedMeshes[name] = mesh;
		return mesh;
	}
	else
	{
		return _loadedMeshes[name];
	}
	//return mesh;
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

void Node::node_to_geometry(
	std::vector<BlasInput>& blasVector,
	const VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress,
	const VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress)
{
	for (Primitive* p : _primitives)
	{
		const uint32_t nTriangles = p->indexCount / 3;

		// Set the triangles geometry
		VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
		triangles.sType						= VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
		triangles.pNext						= nullptr;
		triangles.vertexFormat				= VK_FORMAT_R32G32B32_SFLOAT;
		triangles.vertexData				= vertexBufferDeviceAddress;
		triangles.vertexStride				= sizeof(Vertex);
		triangles.maxVertex					= static_cast<uint32_t>(p->vertexCount);
		triangles.indexData					= indexBufferDeviceAddress;
		triangles.indexType					= VK_INDEX_TYPE_UINT32;

		VkAccelerationStructureGeometryKHR asGeometry = vkinit::acceleration_structure_geometry_khr();
		asGeometry.flags					= VK_GEOMETRY_OPAQUE_BIT_KHR;
		asGeometry.geometryType				= VK_GEOMETRY_TYPE_TRIANGLES_KHR;
		asGeometry.geometry.triangles		= triangles;

		VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};
		asBuildRangeInfo.firstVertex		= 0;// p->firstVertex;
		asBuildRangeInfo.primitiveCount		= nTriangles;
		asBuildRangeInfo.primitiveOffset	= p->firstIndex * sizeof(uint32_t);
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
			child->node_to_geometry(blasVector, vertexBufferDeviceAddress, indexBufferDeviceAddress);
		}
	}
}

void Node::node_to_instance(std::vector<TlasInstance>& instances, int& index, glm::mat4 model)
{
	if (!_primitives.empty())
	{
		glm::mat4 matrix = model * getGlobalMatrix(false);
		for (auto& prim : _primitives)
		{
			TlasInstance instance{};
			instance.transform	= matrix;
			instance.instanceId	= index;
			instance.blasId		= index;
			instances.emplace_back(instance);
			prim->instanceID	= index;
			index++;
		}
	}
	for (Node* child : _children)
	{
		child->node_to_instance(instances, index, model);
	}
}

unsigned int Node::get_number_nodes()
{
	unsigned int count = 0;

	if (!_primitives.empty())
		count++;
	for (Node* child : _children)
		count += child->get_number_nodes();

	return count;
}

void Node::fill_matrix_buffer(std::vector<glm::mat4>& buffer, const glm::mat4 model)
{
	if (!_primitives.empty())
	{
		glm::mat4 matrix = model * getGlobalMatrix();
		for (auto& prim : _primitives)
		{
			prim->transformID = buffer.size();
		}
		buffer.push_back(matrix);
	}
	for (Node* n : _children)
		n->fill_matrix_buffer(buffer, model);
}

void Node::fill_index_buffer(std::vector<glm::vec4>& buffer)
{
	if (!_primitives.empty())
	{
		for (const auto& prim : _primitives)
		{
			glm::vec4 aux = glm::vec4(prim->instanceID, prim->materialID, prim->transformID, prim->firstIndex);
			buffer.push_back(aux);
		}
	}
	for (Node* n : _children)
		n->fill_index_buffer(buffer);
}

void Node::addMaterial(Material* mat)
{
	if (Material::exists(mat))
	{
		_primitives[0]->materialID = Material::getIndex(mat);
	}
	else
	{
		Material::_materials.push_back(mat);
		_primitives[0]->materialID = Material::_materials.size() - 1;
	}
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
	input.asBuildRangeInfo				= asBuildRangeInfo;
	input.asGeometry					= asGeometry;
	input.nTriangles					= nTriangles;

	return input;
}

void Prefab::drawNode(VkCommandBuffer& cmd, VkPipelineLayout pipelineLayout, Node& node, glm::mat4& model)
{
	if (node._primitives.size() > 0)
	{
		glm::mat4 node_matrix = model * node.getGlobalMatrix(false);

		for (Primitive* prim : node._primitives)
		{
			GPUMaterial mat = Material::_materials[prim->materialID]->materialToShader();

			if (prim->indexCount > 0) 
			{
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4), &node_matrix);
				vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(glm::mat4), sizeof(GPUMaterial), &mat);
				vkCmdDrawIndexed(cmd, prim->indexCount, 1, prim->firstIndex, 0, 0);
			}
		}
	}

	for(auto& child : node._children)
		drawNode(cmd, pipelineLayout, *child, model);
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
	if (!_root.empty())
	{
		for(auto& root : _root)
			drawNode(cmd, pipelineLayout, *root, model);
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
			Primitive* prim = new Primitive();
			prim->firstIndex	= firstIndex;
			prim->indexCount	= indexCount;
			prim->firstVertex	= firstVertex;
			prim->vertexCount	= vertexCount;
			prim->materialID	= loadMaterial(tmodel, tprimitive.material);
			loadTextures(tmodel, prim->materialID);
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
		_root.push_back(node);
	}
}

int Prefab::loadMaterial(const tinygltf::Model& tmodel, const int index)
{
	Material* mat = new Material();
	if (index > -1)
	{
		auto& tmat = tmodel.materials[index];
		
		auto& tpbr = tmat.pbrMetallicRoughness;
		mat->diffuseColor				= glm::vec4(tpbr.baseColorFactor[0], tpbr.baseColorFactor[1], tpbr.baseColorFactor[2], tpbr.baseColorFactor[3]);
		mat->metallicFactor				= tpbr.metallicFactor;
		mat->roughnessFactor			= tpbr.roughnessFactor;
		mat->diffuseTexture				= tpbr.baseColorTexture.index;
		mat->metallicRoughnessTexture	= tpbr.metallicRoughnessTexture.index;
		mat->normalTexture				= tmat.normalTexture.index;
	}

	if (Material::exists(mat))
	{
		return Material::getIndex(mat);
	}
	else
	{
		Material::_materials.push_back(mat);
		return Material::_materials.size() - 1;
	}
}

void Prefab::loadTextures(const tinygltf::Model& tmodel, const int index)
{
	Material* mat = Material::_materials[index];

	if (mat->diffuseTexture > -1)
	{
		Texture::GET(tmodel.images[mat->diffuseTexture].uri.c_str());
		mat->diffuseTexture = Texture::get_id(tmodel.images[mat->diffuseTexture].uri.c_str());
	}
	if (mat->normalTexture > -1)
	{
		Texture::GET(tmodel.images[mat->normalTexture].uri.c_str());
		mat->normalTexture = Texture::get_id(tmodel.images[mat->normalTexture].uri.c_str());
	}
	if (mat->emissiveTexture > -1)
	{
		Texture::GET(tmodel.images[mat->emissiveTexture].uri.c_str());
		mat->emissiveTexture = Texture::get_id(tmodel.images[mat->emissiveTexture].uri.c_str());
	}
	if (mat->metallicRoughnessTexture > -1)
	{
		Texture::GET(tmodel.images[mat->metallicRoughnessTexture].uri.c_str());
		mat->metallicRoughnessTexture = Texture::get_id(tmodel.images[mat->metallicRoughnessTexture].uri.c_str());
	}
}

void Prefab::createOBJprefab(Mesh* mesh)
{
	Node* node = new Node();
	_mesh = mesh;
	Primitive* p = new Primitive();
	p->indexCount		= _mesh ? _mesh->_indices.size() : 0;
	p->vertexCount		= _mesh ? _mesh->_vertices.size() : 0;
	p->materialID	= Material::setDefaultMaterial();
	node->_primitives.push_back(p);
	_root.push_back(node);
}

Prefab* Prefab::GET(const std::string filename, bool invertNormals)
{
	std::string name = vkutil::findFile(filename, searchPaths, true);
	if (!_prefabsMap[name])
	{
		Prefab* prefab = new Prefab();
		if (filename.find(".obj") != std::string::npos) 
		{
			prefab->createOBJprefab(Mesh::GET(name.c_str()));
			_prefabsMap[name] = prefab;

			return prefab;
		}
		else
		{
			bool binary;
			if (filename.find(".gltf") != std::string::npos)
				binary = false;
			else if (filename.find(".glb") != std::string::npos)
				binary = true;
			else {
				std::cout << "No valid filename" << std::endl;
				return nullptr;
			}

			std::cout << "Loading gltf... " << filename << std::endl;

			tinygltf::Model		gltfModel;
			tinygltf::TinyGLTF	gltfContext;
			std::string			warn, err;
			bool				fileLoaded{ false };

			fileLoaded = binary ? gltfContext.LoadBinaryFromFile(&gltfModel, &err, &warn, vkutil::findFile(filename, searchPaths, true))
				: gltfContext.LoadASCIIFromFile(&gltfModel, &err, &warn, vkutil::findFile(filename, searchPaths, true));

			if (!err.empty())
				throw std::runtime_error(err.c_str());

			if (fileLoaded)
			{
				// TODO: import materials
				const tinygltf::Scene& scene = gltfModel.scenes[0];
				prefab->_mesh = new Mesh();

				for (const int node : scene.nodes)
				{
					prefab->loadNode(gltfModel, gltfModel.nodes[node], nullptr, invertNormals);
				}

				prefab->_mesh->upload();

				_prefabsMap[name] = prefab;
				return prefab;
			}
		}
	}

	return _prefabsMap[name];
}

Prefab* Prefab::GET(const std::string name, Mesh* mesh)
{
	if (!Prefab::_prefabsMap[name])
	{
		Prefab* prefab = new Prefab();
		prefab->createOBJprefab(mesh);
		return prefab;
	}

	return Prefab::_prefabsMap[name];
}