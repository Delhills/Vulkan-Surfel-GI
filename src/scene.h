#pragma once

#include "vk_types.h"
#include "camera.h"
#include "entity.h"

class Scene
{
public:
	std::vector<Object*>		_entities;
	std::vector<Light*>			_lights;

	std::vector<glm::mat4> _matricesVector;

	Camera* _camera;

	unsigned int get_drawable_nodes_size();
	void create_scene(int i);
private:
	void default_scene();
	void cornell_scene();
	void big_cornell_scene();
};