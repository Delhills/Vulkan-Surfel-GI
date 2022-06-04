
#include "camera.h"

Camera::Camera(glm::vec3 position, glm::vec3 up, float fov, float yaw, float pitch) : _direction(glm::vec3(0, 0, -1)), _speed(SPEED), _sensitivity(SENSITIVITY)
{
	_position	= position;
	_up			= up;
	_yaw		= yaw;
	_pitch		= pitch;
	_fov		= fov;
	updateCameraVectors();
}

void Camera::processKeyboard(Camera_Movement direction, const float dt)
{
	float movementSpeed = _speed * dt;
	if (direction == FORWARD)
		_position += _direction * movementSpeed;
	if (direction == BACKWARD)
		_position -= _direction * movementSpeed;
	if (direction == RIGHT)
		_position += _right * movementSpeed;
	if (direction == LEFT)
		_position -= _right * movementSpeed;
	if (direction == UP)
		_position += _up * movementSpeed;
	if (direction == DOWN)
		_position -= _up * movementSpeed;
}

void Camera::rotate(float xoffset, float yoffset, bool constrainPitch)
{
	xoffset *= _sensitivity;
	yoffset *= _sensitivity;

	_yaw -= xoffset;
	_pitch += yoffset;

	if (constrainPitch) {
		if (_pitch > 89.0f)
			_pitch = 89.0f;
		if (_pitch < -89.0f)
			_pitch = -89.9f;
	}

	updateCameraVectors();
}

glm::mat4 Camera::getView()
{
	return glm::lookAt(_position, _position + _direction, glm::vec3(0, 1, 0));
}

glm::mat4 Camera::getProjection(const float ratio)
{
	// TODO
	return glm::perspective(glm::radians(_fov), ratio, 0.1f, 1000.0f);
}


void Camera::updateCameraVectors()
{
	glm::vec3 front;
	front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
	front.y = sin(glm::radians(_pitch));
	front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));

	_direction	= glm::normalize(front);
	_right		= glm::normalize(glm::cross(_direction, glm::vec3(0, 1, 0)));
	_up			= glm::normalize(glm::cross(_right, _direction));
}