#pragma once

#include <glm/glm/vec3.hpp>
#include <glm/glm/mat4x4.hpp>
#include <glm/glm/gtc/matrix_transform.hpp>

enum Camera_Movement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 0.1f;
const float SENSITIVITY = 0.1f;

class Camera
{
public:

	Camera(glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0, 1.0f, 0.0f), float fov = 40.0f, float yaw = YAW, float pitch = PITCH);

	glm::vec3 _position;
	glm::vec3 _direction;
	glm::vec3 _up;
	glm::vec3 _right;

	float _yaw;
	float _pitch;
	float _speed;
	float _sensitivity;
	float _fov;

	void processKeyboard(Camera_Movement direction, const float dt);
	void rotate(float xoffset, float yoffset, bool constrainPitch = true);

	glm::mat4 getView();
	glm::mat4 getProjection(const float ratio);

private:
	void updateCameraVectors();
};