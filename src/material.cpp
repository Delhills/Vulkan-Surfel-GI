#include "material.h"
#include "vk_utils.h"

int Material::setDefaultMaterial()
{
	Material* mat = new Material();

	if (Material::exists(mat))
	{
		return getIndex(mat);
	}
	else
	{
		Material::_materials.push_back(mat);
		return Material::_materials.size() - 1;
	}
}

GPUMaterial Material::materialToShader()
{
	GPUMaterial mat;
	mat.diffuseColor				= glm::vec4(diffuseColor[0], diffuseColor[1], diffuseColor[2], ior);
	mat.textures					= glm::vec4(diffuseTexture, normalTexture, emissiveTexture, metallicRoughnessTexture);
	mat.shadingMetallicRoughness	= glm::vec4(shadingModel, metallicFactor, roughnessFactor, uvFactor/*Material::getIndex(this)*/);
	return mat;
}

bool Material::exists(Material* material)
{
	for (Material* m : _materials)
	{
		if (*m == *material)
			return true;
	}
	
	return false;
}

int Material::getIndex(Material* material)
{
	int count = 0;
	for (Material* m : _materials)
	{
		if (*m == *material)
			return count;
		count++;
	}
	return -1;
}

inline bool Material::operator==(const Material& m)
{
	return shadingModel == m.shadingModel &&
		diffuseColor				== m.diffuseColor &&
		diffuseTexture				== m.diffuseTexture &&
		normalTexture				== m.normalTexture &&
		emissiveTexture				== m.emissiveTexture &&
		metallicRoughnessTexture	== m.metallicRoughnessTexture &&
		ior							== m.ior &&
		metallicFactor				== m.metallicFactor &&
		roughnessFactor				== m.roughnessFactor;
}
