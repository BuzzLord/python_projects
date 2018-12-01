
#define GLEW_STATIC
#include "GL/glew.h"
#include <IL/il.h>
#include <stdio.h>
#include <string>
#include <sstream>
// Uncomment your platform
#define OVR_OS_WIN32
//#define OVR_OS_MAC
//#define OVR_OS_LINUX
#include "OVR_CAPI_GL.h"
#include "GLUtils.h"

#include "SDL.h"
#include "SDL_syswm.h"

#include "Application.h"


Application::Application(unsigned int seed)
{
	initialized = Initialize(seed);
}

Application::Application()
{
	randSeed = 1234;
	initialized = Initialize(randSeed);
}

Application::~Application()
{
	Cleanup();
}

bool Application::Initialize(unsigned int seed)
{
	int x = SDL_WINDOWPOS_CENTERED;
	int y = SDL_WINDOWPOS_CENTERED;
	Uint32 flags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN;
	randSeed = seed;

	ovrResult result = ovr_Create(&session, &luid);
	if (!OVR_SUCCESS(result))
		return false;

	hmdDesc = ovr_GetHmdDesc(session);
	windowSize = { hmdDesc.Resolution.w / 2, hmdDesc.Resolution.h / 2 };
	window = SDL_CreateWindow("Oculus Rift SDL2 OpenGL Demo", x, y, windowSize.w, windowSize.h, flags);
	context = SDL_GL_CreateContext(window);
	
	glewExperimental = GL_TRUE;
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CW);
	glEnable(GL_CULL_FACE);

	// Make eye render buffers
	for (int eye = 0; eye < 2; ++eye)
	{
		ovrSizei idealTextureSize = ovr_GetFovTextureSize(session, ovrEyeType(eye), hmdDesc.DefaultEyeFov[eye], 1);
		eyeRenderTexture[eye] = new TextureBuffer(session, true, true, idealTextureSize, 1, NULL, 1);
		eyeDepthBuffer[eye] = new DepthBuffer(eyeRenderTexture[eye]->GetSize(), 0);

		if (!eyeRenderTexture[eye]->TextureChain)
		{
			VALIDATE(false, "Failed to create texture.");
		}
	}

	CreateCubeMapStuff();

	ovrMirrorTextureDesc desc;
	memset(&desc, 0, sizeof(desc));
	desc.Width = windowSize.w;
	desc.Height = windowSize.h;
	desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;

	// Create mirror texture and an FBO used to copy mirror texture to back buffer
	result = ovr_CreateMirrorTextureWithOptionsGL(session, &desc, &mirrorTexture);
	if (!OVR_SUCCESS(result))
	{
		VALIDATE(false, "Failed to create mirror texture.");
	}

	// Configure the mirror read buffer
	GLuint texId;
	ovr_GetMirrorTextureBufferGL(session, mirrorTexture, &texId);

	glGenFramebuffers(1, &mirrorFBO);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId, 0);
	glFramebufferRenderbuffer(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

	SDL_GL_SetSwapInterval(0);

	SDL_SysWMinfo info;

	SDL_VERSION(&info.version);

	SDL_GetWindowWMInfo(window, &info);

	saveImage = 0;
	ilInit();
	
	//roomScene = new Scene(false);
	roomScene = new RandomScene(randSeed);

	Pos2 = Vector3f(0.0f, 0.0f, 0.0f);
	Yaw = 3.1415926f;
	Pitch = 0.0f;
	cubeClock = 0.0f;

	// FloorLevel will give tracking poses where the floor height is 0
	ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);
	
	return true;
}

bool Application::CreateCubeMapStuff()
{
	// Make cube map render buffers
	for (int i = 0; i < 5; ++i)
	{
		ovrSizei cubeTexSize;
		cubeTexSize.w = 4096;
		cubeTexSize.h = 4096;

		cubeRenderTexture[i] = new TextureBuffer(session, true, false, cubeTexSize, 4, NULL, 1);
		cubeDepthBuffer[i] = new DepthBuffer(cubeRenderTexture[i]->GetSize(), 0);
	}

	// Make warp textures
	for (int i = 0; i < 2; ++i)
	{
		ovrSizei warpTexSize;
		warpTexSize.w = 2048;
		warpTexSize.h = 2048;

		warpedEyeTexture[i] = new TextureBuffer(session, true, false, warpTexSize, 1, NULL, 1);
		warpedEyeDepth[i] = new DepthBuffer(warpedEyeTexture[i]->GetSize(), 0);
	}

	static const GLchar* VertexShaderSrc =
		"#version 150\n"
		"uniform mat4 matWVP;\n"
		"in      vec4 Position;\n"
		"in      vec4 Color;\n"
		"in      vec2 TexCoord;\n"
		"out     vec2 oTexCoord;\n"
		"out     vec4 oColor;\n"
		"void main()\n"
		"{\n"
		"   gl_Position = (matWVP * Position);\n"
		"   oTexCoord   = TexCoord;\n"
		"   oColor.rgb  = pow(Color.rgb, vec3(2.2));\n"   // convert from sRGB to linear
		"   oColor.a    = Color.a;\n"
		"}\n";

	static const char* FragmentShaderSrc =
		"#version 150\n"
		"uniform sampler2D Texture0;\n"
		"in      vec4      oColor;\n"
		"in      vec2      oTexCoord;\n"
		"out     vec4      FragColor;\n"
		"void main()\n"
		"{\n"
		"   FragColor = oColor * texture2D(Texture0, oTexCoord);\n"
		"}\n";

	static const char* WarpFragmentShaderSrc =
		"#version 150\n\
    uniform sampler2D Texture0;\n\
	uniform sampler2D Texture1;\n\
	uniform sampler2D Texture2;\n\
	uniform sampler2D Texture3;\n\
	uniform sampler2D Texture4;\n\
	in      vec4      oColor;\n\
	in      vec2      oTexCoord;\n\
	out     vec4      FragColor;\n\
	vec2 sampleCube(const vec3 v,out float faceIndex) {\n\
		vec3 vAbs = abs(v);\n\
		float ma;\n\
		vec2 uv;\n\
		if (vAbs.z >= vAbs.x && vAbs.z >= vAbs.y) {\n\
			faceIndex = v.z < 0.0 ? 5.0 : 4.0;\n\
			ma = 0.5 / vAbs.z;\n\
			uv = vec2(v.z < 0.0 ? -v.x : v.x, -v.y);\n\
		} else if (vAbs.y >= vAbs.x) {\n\
			faceIndex = v.y < 0.0 ? 3.0 : 2.0;\n\
			ma = 0.5 / vAbs.y;\n\
			uv = vec2(v.x, v.y < 0.0 ? -v.z : v.z);\n\
		} else {\n\
			faceIndex = v.x < 0.0 ? 1.0 : 0.0;\n\
			ma = 0.5 / vAbs.x;\n\
			uv = vec2(v.x < 0.0 ? v.z : -v.z, -v.y);\n\
		}\n\
		return uv * ma + 0.5;\n\
	}\n\
	vec3 toSphere(const vec2 texCoord) {\n\
		vec2 theta = 1.570796 * ((texCoord) * 2.0 - vec2(1.0));\n\
		float cosphi = cos(theta.y);\n\
		vec3 v = vec3(cosphi * sin(-theta.x), sin(theta.y), cosphi * cos(-theta.x));\n\
		return v;\n\
	}\n\
	vec4 sampleCubeColor(vec3 v) {\n\
		vec4 c;\n\
		float faceIndex;\n\
		vec2 uv = sampleCube(v, faceIndex);\n\
		if (faceIndex == 0.0) {\n\
			c = texture2D(Texture2, uv);\n\
		} else if (faceIndex == 1.0) {\n\
			c = texture2D(Texture1, uv);\n\
		} else if (faceIndex == 2.0) {\n\
			c = texture2D(Texture4, uv);\n\
		} else if (faceIndex == 3.0) {\n\
			c = texture2D(Texture3, uv);\n\
		} else if (faceIndex == 4.0) {\n\
			c = texture2D(Texture0, uv);\n\
		} else {\n\
			c = vec4(0.0);\n\
		}\n\
		return c;\n\
	}\n\
	void main() {\n\
		vec4 AccColor = vec4(0.0);\n\
		AccColor += 0.40*sampleCubeColor(toSphere(oTexCoord));\n\
		// Pixel width for 2048 is 0.000488. Offsets a little under half pixel width.\n\
		AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(0.0002,0.0002)));\n\
		AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(0.0002,-0.0002)));\n\
		AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(-0.0002,0.0002)));\n\
		AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(-0.0002,-0.0002)));\n\
		FragColor = AccColor;\n\
	}";

	GLuint    vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
	GLuint    fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);
	GLuint    wfshader = CreateShader(GL_FRAGMENT_SHADER, WarpFragmentShaderSrc);

	ShaderFill * warp_shader = new ShaderFill(vshader, wfshader, cubeRenderTexture[0]);
	warp_shader->AddTextureBuffer(cubeRenderTexture[1]);
	warp_shader->AddTextureBuffer(cubeRenderTexture[2]);
	warp_shader->AddTextureBuffer(cubeRenderTexture[3]);
	warp_shader->AddTextureBuffer(cubeRenderTexture[4]);
	warp_shader->AddTextureBuffer(cubeRenderTexture[5]);
	warp_shader->destroyTextures = false;
	screenspaceQuad = new Model(Vector3f(0, 0, 0), warp_shader);
	screenspaceQuad->AddOrientedQuad(1.0f, 0.0f, 0.5f, 0.0f, 1.0f, 0.5f, 0xffffffff);
	screenspaceQuad->AllocateBuffers();
	
	ShaderFill * render_texture[6];
	// Front
	render_texture[0] = new ShaderFill(vshader, fshader, cubeRenderTexture[0]);
	render_texture[0]->destroyTextures = false;
	cubeMapModels[0] = new Model(Vector3f(0, 0, 0), render_texture[0]);
	cubeMapModels[0]->AddOrientedQuad(0.1f, 1.0f, 1.0f, -0.1f, 1.2f, 1.0f, 0xffffffff);
	cubeMapModels[0]->AllocateBuffers();
	// left
	render_texture[1] = new ShaderFill(vshader, fshader, cubeRenderTexture[1]);
	render_texture[1]->destroyTextures = false;
	cubeMapModels[1] = new Model(Vector3f(0, 0, 0), render_texture[1]);
	cubeMapModels[1]->AddOrientedQuad(0.3f, 1.0f, 1.0f, 0.1f, 1.2f, 1.0f, 0xffffffff);
	cubeMapModels[1]->AllocateBuffers();
	// right
	render_texture[2] = new ShaderFill(vshader, fshader, cubeRenderTexture[2]);
	render_texture[2]->destroyTextures = false;
	cubeMapModels[2] = new Model(Vector3f(0, 0, 0), render_texture[2]);
	cubeMapModels[2]->AddOrientedQuad(-0.1f, 1.0f, 1.0f, -0.3f, 1.2f, 1.0f, 0xffffffff);
	cubeMapModels[2]->AllocateBuffers();
	// up
	render_texture[3] = new ShaderFill(vshader, fshader, cubeRenderTexture[3]);
	render_texture[3]->destroyTextures = false;
	cubeMapModels[3] = new Model(Vector3f(0, 0, 0), render_texture[3]);
	cubeMapModels[3]->AddOrientedQuad(0.1f, 1.2f, 1.0f, -0.1f, 1.4f, 1.0f, 0xffffffff);
	cubeMapModels[3]->AllocateBuffers();
	// down
	render_texture[4] = new ShaderFill(vshader, fshader, cubeRenderTexture[4]);
	render_texture[4]->destroyTextures = false;
	cubeMapModels[4] = new Model(Vector3f(0, 0, 0), render_texture[4]);
	cubeMapModels[4]->AddOrientedQuad(0.1f, 0.8f, 1.0f, -0.1f, 1.0f, 1.0f, 0xffffffff);
	cubeMapModels[4]->AllocateBuffers();
	// warped
	render_texture[5] = new ShaderFill(vshader, fshader, warpedEyeTexture[0]);
	render_texture[5]->destroyTextures = false;
	cubeMapModels[5] = new Model(Vector3f(0, 0, 0), render_texture[5]);
	cubeMapModels[5]->AddOrientedQuad(-0.1f, 1.2f, 1.0f, -0.3f, 1.4f, 1.0f, 0xffffffff);
	cubeMapModels[5]->AllocateBuffers();

	return true;
}

bool Application::HandleEvents()
{
	SDL_Event event;
	bool running = true;
	
	while (SDL_PollEvent(&event))
	{
		switch (event.type)
		{
		case SDL_QUIT:
			running = false;
			break;
		case SDL_KEYDOWN:
			switch (event.key.keysym.sym)
			{
			case SDLK_ESCAPE:
				running = false;
				break;
			case SDLK_w:
			case SDLK_UP:
				Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(0, 0, -0.05f));
				break;
			case SDLK_s:
			case SDLK_DOWN:
				Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(0, 0, +0.05f));
				break;
			case SDLK_d:
				Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(+0.05f, 0, 0));
				break;
			case SDLK_a:
				Pos2 += Matrix4f::RotationY(Yaw).Transform(Vector3f(-0.05f, 0, 0));
				break;
			case SDLK_RIGHT:
				Yaw -= 0.02f;
				break;
			case SDLK_LEFT:
				Yaw += 0.02f;
				break;
			case SDLK_e:
				Pitch += 0.01f;
				break;
			case SDLK_c:
				Pitch -= 0.01f;
				break;
			default:
				break;
			}
			break;
		case SDL_KEYUP:
			switch (event.key.keysym.sym)
			{
			case SDLK_SPACE:
				saveImage = 2;
				break;
			default:
				break;
			}
		default:
			break;
		}
	}
	return running;
}

bool Application::RenderLoop()
{
	if (!initialized) 
		return false;

	bool running = true;
	ovrResult result;

	while (running)
	{
		running = HandleEvents();

		ovrSessionStatus sessionStatus;
		ovr_GetSessionStatus(session, &sessionStatus);
		if (sessionStatus.ShouldQuit)
		{
			// Because the application is requested to quit, should not request retry
			break;
		}

		if (sessionStatus.ShouldRecenter)
			ovr_RecenterTrackingOrigin(session);

		if (sessionStatus.IsVisible)
		{
			if (!sessionStatus.OverlayPresent) {
				// Pause the application if an overlay is present
				roomScene->Models[0]->Pos = Vector3f(9 * (float)sin(cubeClock), 3, 9 * (float)cos(cubeClock += 0.015f));
			}

			SaveImage(Vector3f(0.03f,0.0f,0.0f));
			SaveImage(Vector3f(-0.03f, 0.0f, 0.0f));
			
			// Call ovr_GetRenderDesc each frame to get the ovrEyeRenderDesc, as the returned values (e.g. HmdToEyePose) may change at runtime.
			ovrEyeRenderDesc eyeRenderDesc[2];
			eyeRenderDesc[0] = ovr_GetRenderDesc(session, ovrEye_Left, hmdDesc.DefaultEyeFov[0]);
			eyeRenderDesc[1] = ovr_GetRenderDesc(session, ovrEye_Right, hmdDesc.DefaultEyeFov[1]);

			// Get eye poses, feeding in correct IPD offset
			ovrPosef EyeRenderPose[2];
			ovrPosef HmdToEyePose[2] = { eyeRenderDesc[0].HmdToEyePose,
				eyeRenderDesc[1].HmdToEyePose };

			double sensorSampleTime;    // sensorSampleTime is fed into the layer later
			ovr_GetEyePoses(session, frameIndex, ovrTrue, HmdToEyePose, EyeRenderPose, &sensorSampleTime);

			// Render Scene to Eye Buffers
			for (int eye = 0; eye < 2; ++eye)
			{
				// Switch to eye render target
				eyeRenderTexture[eye]->SetAndClearRenderSurface(eyeDepthBuffer[eye]);

				// Get view and projection matrices
				Matrix4f rollPitchYaw = Matrix4f::RotationY(Yaw) * Matrix4f::RotationX(Pitch);
				Matrix4f finalRollPitchYaw = rollPitchYaw * Matrix4f(EyeRenderPose[eye].Orientation);
				Vector3f finalUp = finalRollPitchYaw.Transform(Vector3f(0, 1, 0));
				Vector3f finalForward = finalRollPitchYaw.Transform(Vector3f(0, 0, -1));
				Vector3f shiftedEyePos = Pos2 + rollPitchYaw.Transform(EyeRenderPose[eye].Position);

				Matrix4f view = Matrix4f::LookAtRH(shiftedEyePos, shiftedEyePos + finalForward, finalUp);
				Matrix4f proj = ovrMatrix4f_Projection(hmdDesc.DefaultEyeFov[eye], 0.2f, 1000.0f, ovrProjection_None);

				// Draw the scene to the view with all transforms				
				roomScene->Render(view, proj);

				// Draw the cube map textures to quads without taking pos2 or yaw into account.
				Matrix4f interfaceRollPitchYaw = Matrix4f::RotationY(2.6179939f);
				Matrix4f interfaceEyeRollPitchYaw = interfaceRollPitchYaw * Matrix4f(EyeRenderPose[eye].Orientation);
				Vector3f interfaceUp = interfaceEyeRollPitchYaw.Transform(Vector3f(0, 1, 0));
				Vector3f interfaceForward = interfaceEyeRollPitchYaw.Transform(Vector3f(0, 0, -1));
				Vector3f interfaceShiftedEyePos = Vector3f(0.0f,0.0f,0.0f) + interfaceRollPitchYaw.Transform(EyeRenderPose[eye].Position);
				Matrix4f interfaceView = Matrix4f::LookAtRH(interfaceShiftedEyePos, interfaceShiftedEyePos + interfaceForward, interfaceUp);
				for (int i = 0; i < 6; i++) {
					cubeMapModels[i]->Render(interfaceView, proj);
				}
				
				// Avoids an error when calling SetAndClearRenderSurface during next iteration.
				// Without this, during the next while loop iteration SetAndClearRenderSurface
				// would bind a framebuffer with an invalid COLOR_ATTACHMENT0 because the texture ID
				// associated with COLOR_ATTACHMENT0 had been unlocked by calling wglDXUnlockObjectsNV.
				eyeRenderTexture[eye]->UnsetRenderSurface();

				// Commit changes to the textures so they get picked up frame
				eyeRenderTexture[eye]->Commit();
			}

			// Do distortion rendering, Present and flush/sync

			ovrLayerEyeFov ld;
			ld.Header.Type = ovrLayerType_EyeFov;
			ld.Header.Flags = ovrLayerFlag_TextureOriginAtBottomLeft;   // Because OpenGL.

			for (int eye = 0; eye < 2; ++eye)
			{
				ld.ColorTexture[eye] = eyeRenderTexture[eye]->TextureChain;
				ld.Viewport[eye] = Recti(eyeRenderTexture[eye]->GetSize());
				ld.Fov[eye] = hmdDesc.DefaultEyeFov[eye];
				ld.RenderPose[eye] = EyeRenderPose[eye];
				ld.SensorSampleTime = sensorSampleTime;
			}

			ovrLayerHeader* layers = &ld.Header;
			result = ovr_SubmitFrame(session, frameIndex, nullptr, &layers, 1);
			// exit the rendering loop if submit returns an error, will retry on ovrError_DisplayLost
			if (!OVR_SUCCESS(result))
				break;

			frameIndex++;
		}

		// Blit mirror texture to back buffer
		glBindFramebuffer(GL_READ_FRAMEBUFFER, mirrorFBO);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		GLint w = windowSize.w;
		GLint h = windowSize.h;
		glBlitFramebuffer(0, h, w, 0,
			0, 0, w, h,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		SDL_GL_SwapWindow(window);
	}
	return running;
}

void Application::SaveImage(Vector3f position)
{
	// Render the cube maps
	{
		Matrix4f cubemapRotations[5];
		cubemapRotations[0] = Matrix4f::RotationY(0.0f);
		cubemapRotations[1] = Matrix4f::RotationY(1.5708f);
		cubemapRotations[2] = Matrix4f::RotationY(-1.5708f);
		cubemapRotations[3] = Matrix4f::RotationX(1.5708f);
		cubemapRotations[4] = Matrix4f::RotationX(-1.5708f);
		Matrix4f rollPitchYaw = Matrix4f::RotationY(Yaw) * Matrix4f::RotationX(Pitch);
		Vector3f positionOffset = rollPitchYaw.Transform(-position);
		Vector3f finalPos = Pos2 + Vector3f(0.0f, 1.25f, 0.0f) + positionOffset;
		ovrFovPort fov = { 1.0f, 1.0f, 1.0f, 1.0f };

		for (int i = 0; i < 5; i++)
		{
			cubeRenderTexture[i]->SetAndClearRenderSurface(cubeDepthBuffer[i]);

			Matrix4f finalRollPitchYaw = rollPitchYaw * cubemapRotations[i];
			Vector3f finalUp = finalRollPitchYaw.Transform(Vector3f(0, 1, 0));
			Vector3f finalForward = finalRollPitchYaw.Transform(Vector3f(0, 0, -1));

			Matrix4f view = Matrix4f::LookAtRH(finalPos, finalPos + finalForward, finalUp);
			Matrix4f proj = ovrMatrix4f_Projection(fov, 0.2f, 1000.0f, ovrProjection_None);

			roomScene->Render(view, proj);
			cubeRenderTexture[i]->UnsetRenderSurface();
		}
	}

	// Render the warped image
	{
		warpedEyeTexture[0]->SetAndClearRenderSurface(warpedEyeDepth[0]);
		Matrix4f view = Matrix4f::LookAtRH(Vector3f(0.0f, 0.0f, 0.0f), Vector3f(0.0f, 0.0f, -1.0f), Vector3f(0.0f, 1.0f, 0.0f));
		Matrix4f proj = Matrix4f::Ortho2D(1.0f, 1.0f);
		screenspaceQuad->Render(view, proj);
		warpedEyeTexture[0]->UnsetRenderSurface();
	}

	if (saveImage > 0) 
	{
		saveImage--;
		std::string filename;
		std::stringstream strstream;
		strstream << frameIndex << "_";
		int xPos = (int)(position.x * 1000.0f);
		strstream << xPos << "_";
		int yPos = (int)(position.y * 1000.0f);
		strstream << yPos << "_";
		int zPos = (int)(position.z * 1000.0f);
		strstream << zPos << "_";
		strstream << "snapshot.png";
		strstream >> filename;
		//std::string filename;
		//const char* filename = "snapshot.png";

		int width = warpedEyeTexture[0]->texSize.w;
		int height = warpedEyeTexture[0]->texSize.h;
		int components = 4;
		GLubyte *data = (GLubyte*)malloc(components * width * height);
		if (data) {
			warpedEyeTexture[0]->SetRenderSurface();
			glReadBuffer(GL_COLOR_ATTACHMENT0);
			glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
			warpedEyeTexture[0]->UnsetRenderSurface();

			ILuint image;
			ilGenImages(1, &image);
			ilBindImage(image);
			ILboolean ilSuccess = ilTexImage(width, height, 1, components, IL_RGBA, IL_UNSIGNED_BYTE, data);
			if (ilSuccess != IL_TRUE) {
				ILenum error = ilGetError();
				printf("Could not tex image: %d.\n", (int)error);
			}
			else {
				printf("Tex image successful.\n");
				ilEnable(IL_FILE_OVERWRITE);
				ilSuccess = ilSave(IL_PNG, (const wchar_t*)filename.c_str());

				if (ilSuccess != IL_TRUE) {
					ILenum error = ilGetError();
					printf("Could not save image: %d.\n", (int)error);
				}
				else {
					printf("Save successful.\n");
				}
			}

			ilBindImage(0);
			ilDeleteImages(1, &image);
			free(data);
		}
	}
}

bool Application::Cleanup()
{
	if (roomScene) delete roomScene;

	if (mirrorFBO) glDeleteFramebuffers(1, &mirrorFBO);
	if (mirrorTexture) ovr_DestroyMirrorTexture(session, mirrorTexture);
	for (int eye = 0; eye < 2; ++eye)
	{
		delete eyeRenderTexture[eye];
		delete eyeDepthBuffer[eye];
	}

	// Delete cube map render buffers
	for (int i = 0; i < 5; ++i)
	{
		delete cubeRenderTexture[i];
		delete cubeDepthBuffer[i];
	}

	// Delete warp textures
	for (int i = 0; i < 2; ++i)
	{
		delete warpedEyeTexture[i];
		delete warpedEyeDepth[i];
	}

	for (int i = 0; i < 6; ++i)
		delete cubeMapModels[i];

	SDL_GL_DeleteContext(context);

	SDL_DestroyWindow(window);

	ovr_Destroy(session);

	return true;
}


int main(int argc, char *argv[])
{
	SDL_Init(SDL_INIT_VIDEO);

	ovrInitParams initParams = { ovrInit_RequestVersion | ovrInit_FocusAware, OVR_MINOR_VERSION, NULL, 0, 0 };
	ovrResult result = ovr_Initialize(&initParams);

	Application *app = new Application();
	
	app->RenderLoop();

	delete app;

	ovr_Shutdown();

	SDL_Quit();

	return 0;
}
