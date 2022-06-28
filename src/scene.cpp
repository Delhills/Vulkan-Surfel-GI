
#include "scene.h"

unsigned int Scene::get_drawable_nodes_size()
{
	unsigned int count = 0;
	for (Object* o : _entities)
	{
		for (Node* root : o->prefab->_root) {
			count += root->get_number_nodes();
		}
	}

	return count;
}

bool equals(int* a) { return *a > 1; }

void Scene::create_scene(int i)
{
	switch (i)
	{
	case 0:
		default_scene();
		break;
	case 1:
		cornell_scene();
		break;
	case 2:
		big_cornell_scene();
		break;
	default:
		break;
	}
}

void Scene::default_scene()
{
	// Create camera
	_camera = new Camera(glm::vec3(0, 5, 10));
	// Create lights
	// -------------
	Light* light = new Light();
	light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(10, 12, -5));
	light->color = glm::vec3{ 1.0f, 0.8f, 0.5f };
	light->intensity = 3.0f;
	light->radius = 0.1f;

	Light* light2 = new Light();
	light2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-10, 15, -5));
	light2->color = glm::vec3{ 0.5f, 1.0f, 1.0f };
	light2->intensity = 3.0f;
	light2->radius = 0.2f;

	Light* light3 = new Light();
	light3->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 15, 5));
	light3->intensity = 3.f;
	light3->radius = 0.09f;

	_lights.push_back(light);
	_lights.push_back(light2);
	_lights.push_back(light3);

	// Create own Materials
	// --------------------
	Material* m_mirror = new Material();
	m_mirror->shadingModel = 3;
	m_mirror->metallicFactor = 1.f;
	Material* m_glass = new Material();
	m_glass->diffuseColor = glm::vec4{ 0.7f, 0.7f, 1.0f, 1 };
	m_glass->shadingModel = 4;
	m_glass->ior = 1.125;// 1.2f;
	m_glass->metallicFactor = 1.f;
	Material* m_gold = new Material();
	m_gold->diffuseColor = glm::vec4{ 1.0, 0.71, 0.29, 1.0 };
	m_gold->metallicFactor = 0.5f;
	m_gold->roughnessFactor = 0.1f;
	Material* m_red = new Material();
	m_red->diffuseColor = glm::vec4{ 1.0, 0.0, 0.0, 1.0 };
	m_red->metallicFactor = 0.0;
	m_red->roughnessFactor = 0.2;
	Material* m_floor = new Material();
	m_floor->metallicFactor = 0.1f;

	// Create prefabs
	// --------------
	//Prefab* p_sphere	= Prefab::GET("sphere.obj");
	Prefab* p_quad = Prefab::GET("quad", Mesh::get_quad());
	p_quad->_root[0]->addMaterial(m_floor);
	Prefab* p_mirror = Prefab::GET("cube", Mesh::get_cube());
	p_mirror->_root[0]->addMaterial(m_mirror);
	Prefab* p_sphere_mirror = Prefab::GET("sphere.obj");
	p_sphere_mirror->_root[0]->addMaterial(m_mirror);
	Prefab* p_glass_sphere = Prefab::GET("Glass Sphere", Mesh::GET("sphere.obj"));
	p_glass_sphere->_root[0]->addMaterial(m_glass);
	//Prefab* p_gold_sphere = Prefab::GET("goldSphere", Mesh::get_cube());
	//p_gold_sphere->_root[0]->addMaterial(m_gold);
	Prefab* p_red_sphere = Prefab::GET("Red Sphere", Mesh::GET("sphere.obj"));
	p_red_sphere->_root[0]->addMaterial(m_mirror);
	Prefab* p_helmet = Prefab::GET("DamagedHelmet.gltf");
	Prefab* p_lucy = Prefab::GET("lucy", Mesh::GET("lucy.obj"));
	p_lucy->_root[0]->addMaterial(m_gold);

	// Create entities
	// ---------------
	Object* sphere = new Object();
	sphere->prefab = p_red_sphere;
	sphere->m_matrix = glm::translate(glm::mat4(1), glm::vec3(5, 1, -5));
	sphere->material = Material::_materials[p_red_sphere->_root[0]->_primitives[0]->materialID];

	//Object* sphere2 = new Object();
	//sphere2->prefab = p_gold_sphere;
	//sphere2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-5, 1, -5));
	//sphere2->material = Material::_materials[p_gold_sphere->_root[0]->_primitives[0]->materialID];

	Object* sphere3 = new Object();
	sphere3->prefab = p_glass_sphere;
	sphere3->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 1, -5));
	sphere3->material = Material::_materials[p_glass_sphere->_root[0]->_primitives[0]->materialID];

	Object* floor = new Object();
	floor->prefab = p_quad;
	floor->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0, -5)) *
		glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(1, 0, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(15));
	floor->material = Material::_materials[p_quad->_root[0]->_primitives[0]->materialID];

	Object* mirror = new Object();
	mirror->prefab = p_mirror;
	mirror->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 4, -10)) *
		glm::scale(glm::mat4(1), glm::vec3(4, 4, 1));
	mirror->material = Material::_materials[p_mirror->_root[0]->_primitives[0]->materialID];

	Object* cube = new Object();
	cube->prefab = p_glass_sphere;
	cube->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0, -5));

	Object* helmet = new Object();
	helmet->prefab = p_helmet;
	helmet->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-5, 1, -5));
	helmet->material = Material::_materials[p_helmet->_root[0]->_primitives[0]->materialID];

	Object* lucy = new Object();
	lucy->prefab = p_lucy;
	lucy->m_matrix = glm::scale(glm::mat4(1), glm::vec3(0.01));
	lucy->material = Material::_materials[p_lucy->_root[0]->_primitives[0]->materialID];

	Object* lucy2 = new Object();
	lucy2->prefab = p_lucy;
	lucy2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-10, 0, 0)) * glm::scale(glm::mat4(1), glm::vec3(0.01));
	lucy2->material = Material::_materials[p_lucy->_root[0]->_primitives[0]->materialID];

	_entities.push_back(floor);
	_entities.push_back(sphere);
	//_entities.push_back(sphere2);
	_entities.push_back(sphere3);
	_entities.push_back(mirror);
	_entities.push_back(lucy);
	//_entities.push_back(lucy2);
	_entities.push_back(helmet);
	//_entities.push_back(cube);
}

void Scene::cornell_scene()
{
	_camera = new Camera(glm::vec3(0, 5, 20));

	// Create lights
	// -------------
	Light* light = new Light();
	light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 9.0, 0.0));
	//light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0.5, 0));
	light->color = glm::vec3{ 1.0f, 1.0f, 1.0f };
	//light->color = glm::vec3{ 0.78f, 0.78f, 0.78f };
	light->intensity = 4.0f;
	light->radius = 10.1f;
	light->maxDistance = 60.1f;

	_lights.push_back(light);

	Light* light2 = new Light();
	light2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(3.0, 9.6, 0.0));
	//light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0.5, 0));
	light2->color = glm::vec3{ 1.0f, 0.0f, 0.0f };
	//light->color = glm::vec3{ 0.78f, 0.78f, 0.78f };
	light2->intensity = 3.0f;
	light2->radius = 10.1f;
	light2->maxDistance = 60.1f;

	//_lights.push_back(light2);

	// Create own Materials
	// --------------------
	Material* m_white = new Material();
	m_white->diffuseColor = glm::vec4(0.725,
		0.71,
		0.68,
		1.0);
	m_white->metallicFactor = 0.0;


	//m_floor->metallicFactor = 0.1f;
	Material* m_red = new Material();
	m_red->diffuseColor = glm::vec4(0.63,
		0.065,
		0.05,
		1);
	m_red->metallicFactor = 0.0;


	Material* m_green = new Material();
	m_green->diffuseColor = glm::vec4(0.14,
		0.45,
		0.091,
		1.0);
	m_green->metallicFactor = 0.0;

	//Material* m_mirror = new Material();
	//m_mirror->diffuseColor = glm::vec4{ 0.725, 0.71, 0.68, 1 };
	//
	Material* m_glass = new Material();
	m_glass->diffuseColor = glm::vec4{ 0.725, 0.71, 0.68, 1 };
	m_glass->metallicFactor = 0.0;

	// Create prefabs
	// --------------
	Prefab* p_quad = Prefab::GET("quad", Mesh::get_quad());
	p_quad->_root[0]->addMaterial(m_white);
	Prefab* p_white_quad = Prefab::GET("white_quad", Mesh::GET("planonormal.obj"));
	p_white_quad->_root[0]->addMaterial(m_white);
	Prefab* p_white_quad2 = Prefab::GET("white_quad", Mesh::GET("planonormal.obj"));
	p_white_quad2->_root[0]->addMaterial(m_white);
	Prefab* p_ceil = Prefab::GET("white_quad", Mesh::GET("untitled.obj"));
	p_ceil->_root[0]->addMaterial(m_white);
	//Prefab* p_red_quad = Prefab::GET("red_quad", Mesh::get_cube());
	Prefab* p_red_quad = Prefab::GET("red_quad", Mesh::GET("planonormal.obj"));
	p_red_quad->_root[0]->addMaterial(m_red);
	Prefab* p_green_quad = Prefab::GET("quad", Mesh::GET("untitled.obj"));
	p_green_quad->_root[0]->addMaterial(m_green);
	Prefab* p_helmet = Prefab::GET("DamagedHelmet.gltf");
	Prefab* p_mirror_sphere = Prefab::GET("mirror_sphere", Mesh::GET("cube.obj"));
	p_mirror_sphere->_root[0]->addMaterial(m_white);
	Prefab* p_glass_sphere = Prefab::GET("glass_sphere", Mesh::GET("cube.obj"));
	p_glass_sphere->_root[0]->addMaterial(m_white);

	// Create entities
	// ---------------
	Object* floor = new Object();
	floor->prefab = p_white_quad;
	floor->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0, 0.0)); /*
		glm::rotate(glm::mat4(1), glm::radians(180.0f), glm::vec3(0, 0, 1)); */
	floor->material = Material::_materials[p_white_quad->_root[0]->_primitives[0]->materialID];

	// Back wall
	Object* wall1 = new Object();
	wall1->prefab = p_white_quad2;
	wall1->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 5.0, -5.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1, 0, 0));// *
	glm::scale(glm::mat4(1), glm::vec3(0.5, 0.5, 0.5));
	wall1->material = Material::_materials[p_white_quad2->_root[0]->_primitives[0]->materialID];

	// Left wall
	Object* wall2 = new Object();
	wall2->prefab = p_red_quad;
	wall2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-5.0, 5, 0.0)) *
		glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(0, 0, 1));
	wall2->material = Material::_materials[p_red_quad->_root[0]->_primitives[0]->materialID];

	// Right wall
	Object* wall3 = new Object();
	wall3->prefab = p_green_quad;
	wall3->m_matrix = glm::translate(glm::mat4(1), glm::vec3(5.0, 5, 0.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 0, 1));
	wall3->material = Material::_materials[p_green_quad->_root[0]->_primitives[0]->materialID];

	// Cieling
	Object* wall4 = new Object();
	wall4->prefab = p_ceil;
	wall4->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 10.0, 0.0)) *
		glm::rotate(glm::mat4(1), glm::radians(180.0f), glm::vec3(0, 0, 1));
	wall4->material = Material::_materials[p_ceil->_root[0]->_primitives[0]->materialID];

	Object* glass_sphere = new Object();
	glass_sphere->prefab = p_glass_sphere;
	glass_sphere->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-2.0f, 3.01f, -1.5)) *
		glm::rotate(glm::mat4(1), glm::radians(20.0f), glm::vec3(0, 1, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(3, 6, 3));
	glass_sphere->material = Material::_materials[p_glass_sphere->_root[0]->_primitives[0]->materialID];

	Object* planoraro = new Object();
	planoraro->prefab = p_white_quad;
	planoraro->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-3, 0.1, -0.5)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1, 0, 0)) *
		glm::rotate(glm::mat4(1), glm::radians(110.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(0.2, 0.2, 0.2));
	planoraro->material = Material::_materials[p_white_quad->_root[0]->_primitives[0]->materialID];


	Object* mirror_sphere = new Object();
	mirror_sphere->prefab = p_mirror_sphere;
	mirror_sphere->m_matrix = glm::translate(glm::mat4(1), glm::vec3(1.7f, 1.51f, 1.5f)) *
		glm::rotate(glm::mat4(1), glm::radians(-17.5f), glm::vec3(0, 1, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(3, 3, 3));
	mirror_sphere->material = Material::_materials[p_mirror_sphere->_root[0]->_primitives[0]->materialID];

	Object* helmet = new Object();
	helmet->prefab = p_helmet;
	helmet->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 2.5, -5));
	helmet->material = Material::_materials[p_helmet->_root[0]->_primitives[0]->materialID];

	_entities.push_back(floor);
	_entities.push_back(wall1);
	_entities.push_back(wall2);
	_entities.push_back(wall3);
	_entities.push_back(wall4);
	_entities.push_back(glass_sphere);
	_entities.push_back(mirror_sphere);
	///_entities.push_back(planoraro);
}


void Scene::big_cornell_scene()
{
	_camera = new Camera(glm::vec3(0, 5, 20));

	// Create lights
	// -------------
	Light* light = new Light();
	light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 15, -25.0));
	//light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 9, -10.0));
	//light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0.5, 0));
	//light->color = glm::vec3{ 0.0f, 0.0f, 1.0f };
	light->color = glm::vec3{ 0.78f, 0.78f, 0.78f };
	light->intensity = 4.0f;
	light->radius = 10.1f;
	light->maxDistance = 60.1f;

	Light* light1 = new Light();
	light1->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-12.5, 9.5, 40.0));
	//light1->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 9.5, 40.0));
	//light->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0.5, 0));
	//light->color = glm::vec3{ 0.0f, 0.0f, 1.0f };
	light1->color = glm::vec3{ 0.78f, 0.78f, 0.78f };
	light1->intensity = 2.0f;
	light1->radius = 10.1f;
	light1->maxDistance = 60.1f;

	_lights.push_back(light);
	_lights.push_back(light1);

	// Create own Materials
	// --------------------
	Material* m_white = new Material();
	m_white->diffuseColor = glm::vec4(0.725,
		0.71,
		0.68,
		1.0);
	m_white->metallicFactor = 0.0;


	Material* m_blue = new Material();
	m_blue->diffuseColor = glm::vec4(0.05,
		0.11,
		0.78,
		1.0);
	m_blue->metallicFactor = 0.0;


	//m_floor->metallicFactor = 0.1f;
	Material* m_red = new Material();
	m_red->diffuseColor = glm::vec4(0.63,
		0.065,
		0.05,
		1);
	m_red->metallicFactor = 0.0;


	Material* m_green = new Material();
	m_green->diffuseColor = glm::vec4(0.14,
		0.45,
		0.091,
		1.0);
	m_green->metallicFactor = 0.0;

	//Material* m_mirror = new Material();
	//m_mirror->diffuseColor = glm::vec4{ 0.725, 0.71, 0.68, 1 };
	//
	Material* m_glass = new Material();
	m_glass->diffuseColor = glm::vec4{ 0.725, 0.71, 0.68, 1 };
	m_glass->metallicFactor = 0.0;

	// Create prefabs
	// --------------
	Prefab* p_quad = Prefab::GET("quad", Mesh::get_quad());
	p_quad->_root[0]->addMaterial(m_white);
	Prefab* p_white_quad = Prefab::GET("white_quad", Mesh::GET("planonormal.obj"));
	p_white_quad->_root[0]->addMaterial(m_white);
	Prefab* p_white_quad2 = Prefab::GET("white_quad", Mesh::GET("planonormal.obj"));
	p_white_quad2->_root[0]->addMaterial(m_white);

	Prefab* p_ceil = Prefab::GET("white_quad", Mesh::GET("untitled.obj"));
	p_ceil->_root[0]->addMaterial(m_white);
	//Prefab* p_red_quad = Prefab::GET("red_quad", Mesh::get_cube());
	Prefab* p_red_quad = Prefab::GET("red_quad", Mesh::GET("planonormal.obj"));
	p_red_quad->_root[0]->addMaterial(m_red);
	Prefab* p_green_quad = Prefab::GET("quad", Mesh::GET("planonormal.obj"));
	p_green_quad->_root[0]->addMaterial(m_green);
	Prefab* p_helmet = Prefab::GET("DamagedHelmet.gltf");
	Prefab* p_mirror_sphere = Prefab::GET("mirror_sphere", Mesh::GET("cube.obj"));
	p_mirror_sphere->_root[0]->addMaterial(m_white);
	Prefab* p_glass_sphere = Prefab::GET("glass_sphere", Mesh::GET("cube.obj"));
	p_glass_sphere->_root[0]->addMaterial(m_white);

	Prefab* p_blue_quad = Prefab::GET("quad", Mesh::GET("planonormal.obj"));
	p_blue_quad->_root[0]->addMaterial(m_blue);

	Prefab* p_blue_quad2 = Prefab::GET("quad", Mesh::GET("untitled.obj"));
	p_blue_quad2->_root[0]->addMaterial(m_blue);

	Prefab* p_room_box = Prefab::GET("room_box", Mesh::GET("cube.obj"));
	p_room_box->_root[0]->addMaterial(m_white);

	// Create entities
	// ---------------
	Object* floor = new Object();
	floor->prefab = p_white_quad;
	floor->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0, 0.0)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 5));
	floor->material = Material::_materials[p_white_quad->_root[0]->_primitives[0]->materialID];

	Object* floor2 = new Object();
	floor2->prefab = p_white_quad;
	floor2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 0, 40.0)) *
		glm::scale(glm::mat4(1), glm::vec3(5, 1, 3));
	floor2->material = Material::_materials[p_white_quad->_root[0]->_primitives[0]->materialID];

	// Back wall
	Object* wall1 = new Object();
	wall1->prefab = p_white_quad2;
	wall1->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 5.0, -15.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1, 0, 0));// *
	//glm::scale(glm::mat4(1), glm::vec3(0.5, 0.5, 0.5));
	wall1->material = Material::_materials[p_white_quad2->_root[0]->_primitives[0]->materialID];

	// Left wall
	Object* wall2 = new Object();
	wall2->prefab = p_red_quad;
	wall2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-5.0, 5, 0.0)) *
		glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 5));
	wall2->material = Material::_materials[p_red_quad->_root[0]->_primitives[0]->materialID];

	// Right wall
	Object* wall3 = new Object();
	wall3->prefab = p_green_quad;
	wall3->m_matrix = glm::translate(glm::mat4(1), glm::vec3(5.0, 5, 0.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 5));
	wall3->material = Material::_materials[p_green_quad->_root[0]->_primitives[0]->materialID];

	// Cieling
	Object* ceiling = new Object();
	ceiling->prefab = p_ceil;
	ceiling->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 10.0, 10.0)) *
		glm::rotate(glm::mat4(1), glm::radians(180.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 3));
	ceiling->material = Material::_materials[p_ceil->_root[0]->_primitives[0]->materialID];


	// Cieling 2
	Object* ceiling_2 = new Object();
	ceiling_2->prefab = p_ceil;
	ceiling_2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 10.0, -20)) *
		glm::rotate(glm::mat4(1), glm::radians(180.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 1));
	ceiling_2->material = Material::_materials[p_ceil->_root[0]->_primitives[0]->materialID];


	// Cieling 3
	Object* ceiling_3 = new Object();
	ceiling_3->prefab = p_ceil;
	ceiling_3->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 10.0, 40.0)) *
		glm::rotate(glm::mat4(1), glm::radians(180.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(5, 1, 3));
	ceiling_3->material = Material::_materials[p_ceil->_root[0]->_primitives[0]->materialID];

	Object* glass_sphere = new Object();
	glass_sphere->prefab = p_glass_sphere;
	glass_sphere->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-1.5f, 3.1f, -1.5)) *
		glm::rotate(glm::mat4(1), glm::radians(20.0f), glm::vec3(0, 1, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(3, 6, 3));
	glass_sphere->material = Material::_materials[p_glass_sphere->_root[0]->_primitives[0]->materialID];

	Object* mirror_sphere = new Object();
	mirror_sphere->prefab = p_mirror_sphere;
	mirror_sphere->m_matrix = glm::translate(glm::mat4(1), glm::vec3(1.5f, 1.6f, 1.5f)) *
		glm::rotate(glm::mat4(1), glm::radians(-17.5f), glm::vec3(0, 1, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(3, 3, 3));
	mirror_sphere->material = Material::_materials[p_mirror_sphere->_root[0]->_primitives[0]->materialID];

	Object* helmet = new Object();
	helmet->prefab = p_helmet;
	helmet->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0, 2.5, -5));
	helmet->material = Material::_materials[p_helmet->_root[0]->_primitives[0]->materialID];


	Object* roomwall1 = new Object();
	roomwall1->prefab = p_blue_quad;
	roomwall1->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-15.0, 5, 25.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1, 0, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(2, 1, 1));
	roomwall1->material = Material::_materials[p_blue_quad->_root[0]->_primitives[0]->materialID];


	Object* roomwall2 = new Object();
	roomwall2->prefab = p_blue_quad;
	roomwall2->m_matrix = glm::translate(glm::mat4(1), glm::vec3(15.0, 5, 25.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1, 0, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(2, 1, 1));
	roomwall2->material = Material::_materials[p_blue_quad->_root[0]->_primitives[0]->materialID];


	Object* roomwall3 = new Object();
	roomwall3->prefab = p_blue_quad2;
	roomwall3->m_matrix = glm::translate(glm::mat4(1), glm::vec3(0.0, 5, 55.0)) *
		glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(1, 0, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(5, 1, 1));
	roomwall3->material = Material::_materials[p_blue_quad2->_root[0]->_primitives[0]->materialID];


	Object* roomwall4 = new Object();
	roomwall4->prefab = p_red_quad;
	roomwall4->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-25.0, 5, 40.0)) *
		glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 3));
	roomwall4->material = Material::_materials[p_red_quad->_root[0]->_primitives[0]->materialID];


	Object* roomwall5 = new Object();
	roomwall5->prefab = p_green_quad;
	roomwall5->m_matrix = glm::translate(glm::mat4(1), glm::vec3(25.0, 5, 40.0)) *
		glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0, 0, 1)) *
		glm::scale(glm::mat4(1), glm::vec3(1, 1, 3));
	roomwall5->material = Material::_materials[p_green_quad->_root[0]->_primitives[0]->materialID];

	Object* roombox = new Object();
	roombox->prefab = p_room_box;
	roombox->m_matrix = glm::translate(glm::mat4(1), glm::vec3(-20.0f, 2.01f, 30.0f)) *
		glm::rotate(glm::mat4(1), glm::radians(-45.0f), glm::vec3(0, 1, 0)) *
		glm::scale(glm::mat4(1), glm::vec3(4, 4, 4));
	roombox->material = Material::_materials[p_room_box->_root[0]->_primitives[0]->materialID];

	_entities.push_back(floor);
	_entities.push_back(floor2);
	_entities.push_back(wall1);
	_entities.push_back(wall2);
	_entities.push_back(wall3);
	_entities.push_back(ceiling);
	_entities.push_back(ceiling_2);
	_entities.push_back(ceiling_3);
	_entities.push_back(roomwall1);
	_entities.push_back(roomwall2);
	_entities.push_back(roomwall3);
	_entities.push_back(roomwall4);
	_entities.push_back(roomwall5);
	_entities.push_back(glass_sphere);
	_entities.push_back(mirror_sphere);
	_entities.push_back(roombox);
}