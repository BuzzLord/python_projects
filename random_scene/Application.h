#pragma once
#include "OVR_CAPI_GL.h"
#include "GLUtils.h"

#include "SDL.h"
#include "SDL_syswm.h"

class Application
{
public:
	Application();
	Application(unsigned int seed);
	~Application();

	bool RenderLoop();

protected:

	bool Initialize(unsigned int seed);
	bool Cleanup();
	bool HandleEvents();

	bool CreateCubeMapStuff();

	void SaveImage(Vector3f);

	bool initialized;
	int saveImage;

	TextureBuffer * eyeRenderTexture[2] = { nullptr, nullptr };
	DepthBuffer   * eyeDepthBuffer[2] = { nullptr, nullptr };

	// Generate a cube map for each eye, then turn them into warped 180x180 dome images
	TextureBuffer * cubeRenderTexture[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
	DepthBuffer   * cubeDepthBuffer[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
	TextureBuffer * warpedEyeTexture[2] = { nullptr, nullptr };
	DepthBuffer   * warpedEyeDepth[2] = { nullptr, nullptr };

	ovrMirrorTexture mirrorTexture = nullptr;
	GLuint          fboId = 0;
	GLuint          mirrorFBO = 0;

	unsigned int    randSeed;
	//Scene         * roomScene = nullptr;
	RandomScene         * roomScene = nullptr;
	Vector3f        Pos2;
	float           Yaw;
	float			Pitch;
	float           cubeClock;
	Model         * cubeMapModels[6]; // Front,left,right,up,down,warped
	Model		  * screenspaceQuad;

	long long frameIndex = 0;

	ovrSession session;
	ovrGraphicsLuid luid;

	ovrHmdDesc hmdDesc;
	ovrSizei windowSize;

	SDL_Window *window = nullptr;

	SDL_GLContext context;
};