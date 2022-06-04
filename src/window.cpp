#include "window.h"
#include "vk_engine.h"

Window::Window() : 
	_handle(NULL), 
	_width(0), _height(0),
	_minimized(false), _fullscreen(false), _resized(false), 
	_mouse_delta(glm::vec2(0)), _mouse_position(glm::vec2(0)), _mouse_locked(false)
{
}

bool Window::init(const char* name, const int w, const int h)
{
	_width = w;
	_height = h;

	SDL_Init(SDL_INIT_VIDEO);

	SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

	_handle = SDL_CreateWindow(
		name,
		SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_width,
		_height,
		window_flags
	);

	return _handle != NULL;
}

void Window::handleEvent(SDL_Event& e, const float dt)
{
	if (e.type == SDL_WINDOWEVENT) {
		switch (e.window.event)
		{
		case SDL_WINDOWEVENT_RESIZED:
			VulkanEngine::engine->recreate_swapchain();
			break;
		case SDL_WINDOWEVENT_MINIMIZED:
			_minimized = true;
			VulkanEngine::engine->recreate_swapchain();
			break;
		case SDL_WINDOWEVENT_MAXIMIZED:
			_minimized = false;
			break;
		default:
			break;
		}
	}
	else if (e.type == SDL_KEYDOWN)
	{
		if (e.key.keysym.sym == SDLK_w) {
			VulkanEngine::engine->_scene->_camera->processKeyboard(FORWARD, dt);
		}
		if (e.key.keysym.sym == SDLK_a) {
			VulkanEngine::engine->_scene->_camera->processKeyboard(LEFT, dt);
		}
		if (e.key.keysym.sym == SDLK_s) {
			VulkanEngine::engine->_scene->_camera->processKeyboard(BACKWARD, dt);
		}
		if (e.key.keysym.sym == SDLK_d) {
			VulkanEngine::engine->_scene->_camera->processKeyboard(RIGHT, dt);
		}
		if (e.key.keysym.sym == SDLK_LSHIFT) {
			VulkanEngine::engine->_scene->_camera->processKeyboard(DOWN, dt);
		}
		if (e.key.keysym.sym == SDLK_SPACE) {
			VulkanEngine::engine->_scene->_camera->processKeyboard(UP, dt);
		}
		if (e.key.keysym.sym == SDLK_1) {
			VulkanEngine::engine->_mode = DEFERRED;
		}
		if (e.key.keysym.sym == SDLK_2) {
			VulkanEngine::engine->_mode = RAYTRACING;
		}
		if (e.key.keysym.sym == SDLK_3) {
			VulkanEngine::engine->_mode = HYBRID;
		}
		if (e.key.keysym.sym == SDLK_4) {
			VulkanEngine::engine->_skyboxFollow = !VulkanEngine::engine->_skyboxFollow;
			std::cout << VulkanEngine::engine->_skyboxFollow << std::endl;
		}
		if (e.key.keysym.sym == SDLK_ESCAPE) VulkanEngine::engine->_bQuit = true;
	}
	if (e.type == SDL_MOUSEBUTTONDOWN) {
		if (e.button.button == SDL_BUTTON_MIDDLE)
			_mouse_locked = !_mouse_locked;
	}
	if (e.type == SDL_MOUSEMOTION) {
		if (_mouse_locked)
			VulkanEngine::engine->_scene->_camera->rotate(_mouse_delta.x, _mouse_delta.y);
	}
}

int Window::getWidth() { return _width; }
int Window::getHeight() { return _height; }
void Window::setWidth(const int w) { _width = w; }
void Window::setHeight(const int h) { _height = h; }
bool Window::isMinimized() { return _minimized; }
bool Window::isFullscreen() { return _fullscreen; }

void Window::center_mouse()
{
	int center_x = (int)glm::floor(_width * 0.5);
	int center_y = (int)glm::floor(_height * 0.5);

	SDL_WarpMouseInWindow(_handle, center_x, center_y);
	_mouse_position = glm::vec2((float)center_x, (float)center_y);
}

void Window::input_update()
{
	int x, y;
	SDL_GetMouseState(&x, &y);
	_mouse_delta = glm::vec2(_mouse_position.x - x, _mouse_position.y - y);
	_mouse_position = glm::vec2(x, y);

	SDL_ShowCursor(!_mouse_locked);

	//ImGui::SetMouseCursor(_mouse_locked ? ImGuiMouseCursor_None : ImGuiMouseCursor_Arrow);

	if (_mouse_locked)
		center_mouse();
}

void Window::clean()
{
	SDL_DestroyWindow(_handle);
}
