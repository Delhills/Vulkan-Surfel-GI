#pragma once

#include "vk_types.h"

struct GPUMaterial
{
	glm::vec4 diffuseColor;
	glm::vec4 textures;
	glm::vec4 shadingMetallicRoughness;
};

class Material
{
public:
	static std::vector<Material*> _materials;

	int shadingModel{ 0 }; // 0: metallic-roughnes, 1: specular-glossines

	// PBR Metallic-Roughness
	glm::vec4 diffuseColor = glm::vec4(1);
	float metallicFactor{ 1.f };
	float roughnessFactor{ 1.f };
	float ior{ 1.f };
	float uvFactor{ 1.f };

	int diffuseTexture{ -1 };
	int metallicRoughnessTexture{ -1 };
	int emissiveTexture{ -1 };
	int normalTexture{ -1 };

	static int setDefaultMaterial();
	static bool exists(Material* material);
	static int getIndex(Material* material);
	GPUMaterial materialToShader();
	
	bool operator== (const Material& m);
};
