#pragma once

#include "GL/glew.h"
#include "Extras/OVR_Math.h"
#include "OVR_CAPI_GL.h"
#include <assert.h>

#define MAT_FLOOR		0
#define MAT_CEILING		1
#define MAT_WALLS		2
#define MAT_BEAMS		3
#define MAT_TABLE		4
#define MAT_CHAIRS		5
#define MAT_OBJ1		6
#define MAT_OBJ2		7

using namespace OVR;

#ifndef VALIDATE
#define VALIDATE(x, msg) if (!(x)) { MessageBoxA(NULL, (msg), "OculusRoomTiny", MB_ICONERROR | MB_OK); exit(-1); }
#endif

#ifndef OVR_DEBUG_LOG
#define OVR_DEBUG_LOG(x)
#endif

#ifndef COL4
#define COL4(r,g,b,a) ((unsigned long)(((unsigned long)(a) << 24) + ((unsigned long)(r) << 16) + ((unsigned long)(g) << 8) + ((unsigned long)(b))))
#endif

#ifndef COL3
#define COL3(r,g,b) ((unsigned long)((unsigned long)0xff000000 + ((unsigned long)(r) << 16) + ((unsigned long)(g) << 8) + ((unsigned long)(b))))
#endif

//---------------------------------------------------------------------------------------
struct DepthBuffer
{
	GLuint        texId;

	DepthBuffer(Sizei size, int sampleCount)
	{

		assert(sampleCount <= 1); // The code doesn't currently handle MSAA textures.

		glGenTextures(1, &texId);
		glBindTexture(GL_TEXTURE_2D, texId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		GLenum internalFormat = GL_DEPTH_COMPONENT24;
		GLenum type = GL_UNSIGNED_INT;
		if (GLEW_ARB_depth_buffer_float)
		{
			internalFormat = GL_DEPTH_COMPONENT32F;
			type = GL_FLOAT;
		}

		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, size.w, size.h, 0, GL_DEPTH_COMPONENT, type, NULL);
	}
	~DepthBuffer()
	{
		if (texId)
		{
			glDeleteTextures(1, &texId);
			texId = 0;
		}
	}
};

//--------------------------------------------------------------------------
struct TextureBuffer
{
	ovrSession          Session;
	ovrTextureSwapChain  TextureChain;
	GLuint              texId;
	GLuint              fboId;
	Sizei               texSize;

	TextureBuffer(ovrSession session, bool rendertarget, bool displayableOnHmd, Sizei size, int mipLevels, unsigned char * data, int sampleCount) :
		Session(session),
		TextureChain(nullptr),
		texId(0),
		fboId(0),
		texSize(0, 0)
	{
		assert(sampleCount <= 1); // The code doesn't currently handle MSAA textures.

		texSize = size;

		if (displayableOnHmd)
		{
			// This texture isn't necessarily going to be a rendertarget, but it usually is.
			assert(session); // No HMD? A little odd.
			assert(sampleCount == 1); // ovr_CreateSwapTextureSetD3D11 doesn't support MSAA.

			ovrTextureSwapChainDesc desc = {};
			desc.Type = ovrTexture_2D;
			desc.ArraySize = 1;
			desc.Width = size.w;
			desc.Height = size.h;
			desc.MipLevels = 1;
			desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
			desc.SampleCount = 1;
			desc.StaticImage = ovrFalse;

			ovrResult result = ovr_CreateTextureSwapChainGL(Session, &desc, &TextureChain);

			int length = 0;
			ovr_GetTextureSwapChainLength(session, TextureChain, &length);

			if (OVR_SUCCESS(result))
			{
				for (int i = 0; i < length; ++i)
				{
					GLuint chainTexId;
					ovr_GetTextureSwapChainBufferGL(Session, TextureChain, i, &chainTexId);
					glBindTexture(GL_TEXTURE_2D, chainTexId);

					if (rendertarget)
					{
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
					}
					else
					{
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
					}
				}
			}
		}
		else
		{
			glGenTextures(1, &texId);
			glBindTexture(GL_TEXTURE_2D, texId);

			if (rendertarget)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			}

			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, texSize.w, texSize.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		}

		if (mipLevels > 1)
		{
			glGenerateMipmap(GL_TEXTURE_2D);
		}

		glGenFramebuffers(1, &fboId);
	}

	~TextureBuffer()
	{
		if (TextureChain)
		{
			ovr_DestroyTextureSwapChain(Session, TextureChain);
			TextureChain = nullptr;
		}
		if (texId)
		{
			glDeleteTextures(1, &texId);
			texId = 0;
		}
		if (fboId)
		{
			glDeleteFramebuffers(1, &fboId);
			fboId = 0;
		}
	}

	Sizei GetSize() const
	{
		return texSize;
	}

	void SetAndClearRenderSurface(DepthBuffer* dbuffer)
	{
		GLuint curTexId;
		if (TextureChain)
		{
			int curIndex;
			ovr_GetTextureSwapChainCurrentIndex(Session, TextureChain, &curIndex);
			ovr_GetTextureSwapChainBufferGL(Session, TextureChain, curIndex, &curTexId);
		}
		else
		{
			curTexId = texId;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dbuffer->texId, 0);

		glViewport(0, 0, texSize.w, texSize.h);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_FRAMEBUFFER_SRGB);
	}

	void SetRenderSurface()
	{
		GLuint curTexId;
		if (TextureChain)
		{
			int curIndex;
			ovr_GetTextureSwapChainCurrentIndex(Session, TextureChain, &curIndex);
			ovr_GetTextureSwapChainBufferGL(Session, TextureChain, curIndex, &curTexId);
		}
		else
		{
			curTexId = texId;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
	}

	void UnsetRenderSurface()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	}

	void Commit()
	{
		if (TextureChain)
		{
			ovr_CommitTextureSwapChain(Session, TextureChain);
		}
	}
};

GLuint CreateShader(GLenum type, const GLchar* src)
{
	GLuint shader = glCreateShader(type);

	glShaderSource(shader, 1, &src, NULL);
	glCompileShader(shader);

	GLint r;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &r);
	if (!r)
	{
		GLchar msg[1024];
		glGetShaderInfoLog(shader, sizeof(msg), 0, msg);
		if (msg[0]) {
			OVR_DEBUG_LOG(("Compiling shader failed: %s\n", msg));
		}
		return 0;
	}

	return shader;
}

//------------------------------------------------------------------------------
struct ShaderFill
{
	GLuint            program;
	TextureBuffer   * texture[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
	const char      * textureNames[8] = { "Texture0", "Texture1", "Texture2", "Texture3", "Texture4", "Texture5", "Texture6", "Texture7" };
	int               numTextures = 0;
	bool			  destroyTextures = true;
	
	float			  floatUniforms[8];
	const char      * floatUniformNames[8];
	int				  numFloatUniforms = 0;

	Matrix4f          matrixUniforms[4];
	const char      * matrixUniformNames[4];
	int				  numMatrixUniforms = 0;
	

	ShaderFill(GLuint vertexShader, GLuint pixelShader, TextureBuffer* _texture)
	{
		AddTextureBuffer(_texture);
		
		program = glCreateProgram();

		glAttachShader(program, vertexShader);
		glAttachShader(program, pixelShader);

		glLinkProgram(program);

		glDetachShader(program, vertexShader);
		glDetachShader(program, pixelShader);

		GLint r;
		glGetProgramiv(program, GL_LINK_STATUS, &r);
		if (!r)
		{
			GLchar msg[1024];
			glGetProgramInfoLog(program, sizeof(msg), 0, msg);
			OVR_DEBUG_LOG(("Linking shaders failed: %s\n", msg));
		}
	}

	~ShaderFill()
	{
		if (program)
		{
			glDeleteProgram(program);
			program = 0;
		}
		while (numTextures-- > 0)
		{
			if (destroyTextures)
				delete texture[numTextures];
			texture[numTextures] = nullptr;
		}
	}

	void AddTextureBuffer(TextureBuffer* _texture)
	{
		if (numTextures >= 8) {
			return;
		}
		texture[numTextures] = _texture;
		numTextures++;
	}

	void AddFloatUniform(const char* uniformName, float value) {
		if (numFloatUniforms >= 8) {
			return;
		}
		floatUniformNames[numFloatUniforms] = uniformName;
		floatUniforms[numFloatUniforms] = value;
		numFloatUniforms++;
	}

	void AddMatrixUniform(const char* uniformName, Matrix4f value) {
		if (numMatrixUniforms >= 4) {
			return;
		}
		matrixUniformNames[numMatrixUniforms] = uniformName;
		matrixUniforms[numMatrixUniforms] = value;
		numMatrixUniforms++;
	}
};

//----------------------------------------------------------------
struct VertexBuffer
{
	GLuint    buffer;

	VertexBuffer(void* vertices, size_t size)
	{
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ARRAY_BUFFER, buffer);
		glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);
	}
	~VertexBuffer()
	{
		if (buffer)
		{
			glDeleteBuffers(1, &buffer);
			buffer = 0;
		}
	}
};

//----------------------------------------------------------------
struct IndexBuffer
{
	GLuint    buffer;

	IndexBuffer(void* indices, size_t size)
	{
		glGenBuffers(1, &buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, indices, GL_STATIC_DRAW);
	}
	~IndexBuffer()
	{
		if (buffer)
		{
			glDeleteBuffers(1, &buffer);
			buffer = 0;
		}
	}
};

//---------------------------------------------------------------------------
struct Model
{
	struct Vertex
	{
		Vector3f  Pos;
		unsigned long C;
		float     U, V;
	};

	Vector3f        Pos;
	Quatf           Rot;
	Matrix4f        Mat;
	int             numVertices, numIndices;
	Vertex          Vertices[2000]; // Note fixed maximum
	GLushort        Indices[2000];
	ShaderFill    * Fill;
	VertexBuffer  * vertexBuffer;
	IndexBuffer   * indexBuffer;

	Model(Vector3f pos, ShaderFill * fill) :
		numVertices(0),
		numIndices(0),
		Pos(pos),
		Rot(),
		Mat(),
		Fill(fill),
		vertexBuffer(nullptr),
		indexBuffer(nullptr)
	{}

	Model(Model *model) :
		numVertices(0),
		numIndices(0),
		Pos(Vector3f(0.0f,0.0f,0.0f)),
		Rot(),
		Mat(),
		Fill(0),
		vertexBuffer(nullptr),
		indexBuffer(nullptr)
	{
		if (model != 0) {
			Pos = model->Pos;
			Rot = model->Rot;
			Fill = model->Fill;
			for (int i = 0; i < model->numVertices; i++) {
				AddVertex(model->Vertices[i]);
			}
			for (int i = 0; i < model->numIndices; i++) {
				AddIndex(model->Indices[i]);
			}
		}
	
	}

	~Model()
	{
		FreeBuffers();
	}

	Matrix4f& GetMatrix()
	{
		Mat = Matrix4f(Rot);
		Mat = Matrix4f::Translation(Pos) * Mat;
		return Mat;
	}

	void AddVertex(const Vertex& v) { Vertices[numVertices++] = v; }
	void AddIndex(GLushort a) { Indices[numIndices++] = a; }

	void AllocateBuffers()
	{
		vertexBuffer = new VertexBuffer(&Vertices[0], numVertices * sizeof(Vertices[0]));
		indexBuffer = new IndexBuffer(&Indices[0], numIndices * sizeof(Indices[0]));
	}

	void FreeBuffers()
	{
		delete vertexBuffer; vertexBuffer = nullptr;
		delete indexBuffer; indexBuffer = nullptr;
	}

	void AddSolidColorBox(float x1, float y1, float z1, float x2, float y2, float z2, unsigned long c, bool shading)
	{
		Vector3f Vert[][2] =
		{
			// Top
			Vector3f(x1, y2, z1), Vector3f(z1, x1), Vector3f(x2, y2, z1), Vector3f(z1, x2),
			Vector3f(x2, y2, z2), Vector3f(z2, x2), Vector3f(x1, y2, z2), Vector3f(z2, x1),
			// Bottom
			Vector3f(x1, y1, z1), Vector3f(z1, x1), Vector3f(x2, y1, z1), Vector3f(z1, x2),
			Vector3f(x2, y1, z2), Vector3f(z2, x2), Vector3f(x1, y1, z2), Vector3f(z2, x1),
			// Left
			Vector3f(x1, y1, z2), Vector3f(z2, y1), Vector3f(x1, y1, z1), Vector3f(z1, y1),
			Vector3f(x1, y2, z1), Vector3f(z1, y2), Vector3f(x1, y2, z2), Vector3f(z2, y2),
			// Right
			Vector3f(x2, y1, z2), Vector3f(z2, y1), Vector3f(x2, y1, z1), Vector3f(z1, y1),
			Vector3f(x2, y2, z1), Vector3f(z1, y2), Vector3f(x2, y2, z2), Vector3f(z2, y2),
			// Back
			Vector3f(x1, y1, z1), Vector3f(x1, y1), Vector3f(x2, y1, z1), Vector3f(x2, y1),
			Vector3f(x2, y2, z1), Vector3f(x2, y2), Vector3f(x1, y2, z1), Vector3f(x1, y2),
			// Front
			Vector3f(x1, y1, z2), Vector3f(x1, y1), Vector3f(x2, y1, z2), Vector3f(x2, y1),
			Vector3f(x2, y2, z2), Vector3f(x2, y2), Vector3f(x1, y2, z2), Vector3f(x1, y2)
		};

		GLushort CubeIndices[] =
		{
			0, 1, 3, 3, 1, 2,
			5, 4, 6, 6, 4, 7,
			8, 9, 11, 11, 9, 10,
			13, 12, 14, 14, 12, 15,
			16, 17, 19, 19, 17, 18,
			21, 20, 22, 22, 20, 23
		};

		for (int i = 0; i < sizeof(CubeIndices) / sizeof(CubeIndices[0]); ++i)
			AddIndex(CubeIndices[i] + GLushort(numVertices));

		// Generate a quad for each box face
		for (int v = 0; v < 6 * 4; v++)
		{
			// Make vertices, with some token lighting
			Vertex vvv; vvv.Pos = Vert[v][0]; vvv.U = Vert[v][1].x; vvv.V = Vert[v][1].y;
			if (shading) {
				float dist1 = (vvv.Pos - Vector3f(-2, 4, -2)).Length();
				float dist2 = (vvv.Pos - Vector3f(3, 4, -3)).Length();
				float dist3 = (vvv.Pos - Vector3f(-4, 3, 25)).Length();
				int   bri = rand() % 160;
				float B = ((c >> 16) & 0xff) * (bri + 192.0f * (0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
				float G = ((c >> 8) & 0xff) * (bri + 192.0f * (0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
				float R = ((c >> 0) & 0xff) * (bri + 192.0f * (0.65f + 8 / dist1 + 1 / dist2 + 4 / dist3)) / 255.0f;
				vvv.C = (c & 0xff000000) +
					((R > 255 ? 255 : (unsigned long)(R)) << 16) +
					((G > 255 ? 255 : (unsigned long)(G)) << 8) +
					(B > 255 ? 255 : (unsigned long)(B));
			}
			else {
				vvv.C = c;
			}

			AddVertex(vvv);
		}
	}

	void AddSolidColorBox(float x1, float y1, float z1, float x2, float y2, float z2, unsigned long c)
	{
		AddSolidColorBox(x1, y1, z1, x2, y2, z2, c, true);
	}

	void AddOrientedQuad(float x1, float y1, float z1, float x2, float y2, float z2, unsigned long c)
	{
		Vector3f Vert[][2] =
		{
			Vector3f(x1, y1, z1), Vector3f(0.0f, 0.0f), Vector3f(x2, y1, z2), Vector3f(1.0f, 0.0f),
			Vector3f(x2, y2, z2), Vector3f(1.0f, 1.0f), Vector3f(x1, y2, z1), Vector3f(0.0f, 1.0f)
		};

		GLushort QuadIndices[] =
		{
			//0, 1, 3, 3, 1, 2
			0, 3, 1, 1, 3, 2
		};

		for (int i = 0; i < sizeof(QuadIndices) / sizeof(QuadIndices[0]); ++i)
			AddIndex(QuadIndices[i] + GLushort(numVertices));

		// Generate a quad
		for (int v = 0; v < 4; v++)
		{
			Vertex vvv; vvv.Pos = Vert[v][0]; vvv.U = Vert[v][1].x; vvv.V = Vert[v][1].y;
			vvv.C = 0xffffffff; // White
			AddVertex(vvv);
		}
	}

	void Render(Matrix4f view, Matrix4f proj)
	{
		Matrix4f combined = proj * view * GetMatrix();

		glUseProgram(Fill->program);
		glUniformMatrix4fv(glGetUniformLocation(Fill->program, "matWVP"), 1, GL_TRUE, (GLfloat*)&combined);

		for (int i = 0; i < Fill->numMatrixUniforms; i++) {
			glUniformMatrix4fv(glGetUniformLocation(Fill->program, Fill->matrixUniformNames[i]), 1, GL_TRUE, (GLfloat*)&Fill->matrixUniforms[i]);
		}

		for (int i = 0; i < Fill->numTextures; i++) {
			glUniform1i(glGetUniformLocation(Fill->program, Fill->textureNames[i]), i);

			switch (i) {
				case 0: glActiveTexture(GL_TEXTURE0); break;
				case 1: glActiveTexture(GL_TEXTURE1); break;
				case 2: glActiveTexture(GL_TEXTURE2); break;
				case 3: glActiveTexture(GL_TEXTURE3); break;
				case 4: glActiveTexture(GL_TEXTURE4); break;
				case 5: glActiveTexture(GL_TEXTURE5); break;
				case 6: glActiveTexture(GL_TEXTURE6); break;
				case 7: glActiveTexture(GL_TEXTURE7); break;
			}
			glBindTexture(GL_TEXTURE_2D, Fill->texture[i]->texId);
		}

		for (int i = 0; i < Fill->numFloatUniforms; i++) {
			glUniform1f(glGetUniformLocation(Fill->program, Fill->floatUniformNames[i]), Fill->floatUniforms[i]);
		}

		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer->buffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer->buffer);

		GLuint posLoc = glGetAttribLocation(Fill->program, "Position");
		GLuint colorLoc = glGetAttribLocation(Fill->program, "Color");
		GLuint uvLoc = glGetAttribLocation(Fill->program, "TexCoord");

		glEnableVertexAttribArray(posLoc);
		glEnableVertexAttribArray(colorLoc);
		glEnableVertexAttribArray(uvLoc);

		glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Pos));
		glVertexAttribPointer(colorLoc, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (void*)offsetof(Vertex, C));
		glVertexAttribPointer(uvLoc, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, U));

		glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_SHORT, NULL);

		glDisableVertexAttribArray(posLoc);
		glDisableVertexAttribArray(colorLoc);
		glDisableVertexAttribArray(uvLoc);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		glUseProgram(0);
	}
};

//------------------------------------------------------------------------- 
struct Scene
{
	int     numModels;
	Model * Models[10];

	void    Add(Model * n)
	{
		Models[numModels++] = n;
	}

	void Render(Matrix4f view, Matrix4f proj)
	{
		for (int i = 0; i < numModels; ++i)
			Models[i]->Render(view, proj);
	}

	void Init(int includeIntensiveGPUobject)
	{
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

		GLuint    vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
		GLuint    fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);

		// Make textures
		ShaderFill * grid_material[4];
		for (int k = 0; k < 4; ++k)
		{
			static unsigned long tex_pixels[256 * 256];
			for (int j = 0; j < 256; ++j)
			{
				for (int i = 0; i < 256; ++i)
				{
					if (k == 0) tex_pixels[j * 256 + i] = (((i >> 7) ^ (j >> 7)) & 1) ? 0xffb4b4b4 : 0xff505050;// floor
					if (k == 1) tex_pixels[j * 256 + i] = (((j / 4 & 15) == 0) || (((i / 4 & 15) == 0) && ((((i / 4 & 31) == 0) ^ ((j / 4 >> 4) & 1)) == 0)))
						? 0xff3c3c3c : 0xffb4b4b4;// wall
					if (k == 2) tex_pixels[j * 256 + i] = (i / 4 == 0 || j / 4 == 0) ? 0xff505050 : 0xffb4b4b4;// ceiling
					if (k == 3) tex_pixels[j * 256 + i] = 0xffffffff;// blank
				}
			}
			TextureBuffer * generated_texture = new TextureBuffer(nullptr, false, false, Sizei(256, 256), 4, (unsigned char *)tex_pixels, 1);
			grid_material[k] = new ShaderFill(vshader, fshader, generated_texture);
		}

		glDeleteShader(vshader);
		glDeleteShader(fshader);

		// Construct geometry
		Model * m = new Model(Vector3f(0, 0, 0), grid_material[2]);  // Moving box
		m->AddSolidColorBox(0, 0, 0, +1.0f, +1.0f, 1.0f, 0xff404040);
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), grid_material[1]);  // Walls
		m->AddSolidColorBox(-10.1f, 0.0f, -20.0f, -10.0f, 4.0f, 20.0f, 0xff808080); // Left Wall
		m->AddSolidColorBox(-10.0f, -0.1f, -20.1f, 10.0f, 4.0f, -20.0f, 0xff808080); // Back Wall
		m->AddSolidColorBox(10.0f, -0.1f, -20.0f, 10.1f, 4.0f, 20.0f, 0xff808080); // Right Wall
		m->AllocateBuffers();
		Add(m);

		if (includeIntensiveGPUobject)
		{
			m = new Model(Vector3f(0, 0, 0), grid_material[0]);  // Floors
			for (float depth = 0.0f; depth > -3.0f; depth -= 0.1f)
				m->AddSolidColorBox(9.0f, 0.5f, -depth, -9.0f, 3.5f, -depth, 0x10ff80ff); // Partition
			m->AllocateBuffers();
			Add(m);
		}

		m = new Model(Vector3f(0, 0, 0), grid_material[0]);  // Floors
		m->AddSolidColorBox(-10.0f, -0.1f, -20.0f, 10.0f, 0.0f, 20.1f, 0xff808080); // Main floor
		m->AddSolidColorBox(-15.0f, -6.1f, 18.0f, 15.0f, -6.0f, 30.0f, 0xff808080); // Bottom floor
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), grid_material[2]);  // Ceiling
		m->AddSolidColorBox(-10.0f, 4.0f, -20.0f, 10.0f, 4.1f, 20.1f, 0xff808080);
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), grid_material[3]);  // Fixtures & furniture
		m->AddSolidColorBox(9.5f, 0.75f, 3.0f, 10.1f, 2.5f, 3.1f, 0xff383838);   // Right side shelf// Verticals
		m->AddSolidColorBox(9.5f, 0.95f, 3.7f, 10.1f, 2.75f, 3.8f, 0xff383838);   // Right side shelf
		m->AddSolidColorBox(9.55f, 1.20f, 2.5f, 10.1f, 1.30f, 3.75f, 0xff383838); // Right side shelf// Horizontals
		m->AddSolidColorBox(9.55f, 2.00f, 3.05f, 10.1f, 2.10f, 4.2f, 0xff383838); // Right side shelf
		m->AddSolidColorBox(5.0f, 1.1f, 20.0f, 10.0f, 1.2f, 20.1f, 0xff383838);   // Right railing   
		m->AddSolidColorBox(-10.0f, 1.1f, 20.0f, -5.0f, 1.2f, 20.1f, 0xff383838);   // Left railing  
		for (float f = 5.0f; f <= 9.0f; f += 1.0f)
		{
			m->AddSolidColorBox(f, 0.0f, 20.0f, f + 0.1f, 1.1f, 20.1f, 0xff505050);// Left Bars
			m->AddSolidColorBox(-f, 1.1f, 20.0f, -f - 0.1f, 0.0f, 20.1f, 0xff505050);// Right Bars
		}
		m->AddSolidColorBox(-1.8f, 0.8f, 1.0f, 0.0f, 0.7f, 0.0f, 0xff505000); // Table
		m->AddSolidColorBox(-1.8f, 0.0f, 0.0f, -1.7f, 0.7f, 0.1f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(-1.8f, 0.7f, 1.0f, -1.7f, 0.0f, 0.9f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(0.0f, 0.0f, 1.0f, -0.1f, 0.7f, 0.9f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(0.0f, 0.7f, 0.0f, -0.1f, 0.0f, 0.1f, 0xff505000); // Table Leg 
		m->AddSolidColorBox(-1.4f, 0.5f, -1.1f, -0.8f, 0.55f, -0.5f, 0xff202050); // Chair Set
		m->AddSolidColorBox(-1.4f, 0.0f, -1.1f, -1.34f, 1.0f, -1.04f, 0xff202050); // Chair Leg 1
		m->AddSolidColorBox(-1.4f, 0.5f, -0.5f, -1.34f, 0.0f, -0.56f, 0xff202050); // Chair Leg 2
		m->AddSolidColorBox(-0.8f, 0.0f, -0.5f, -0.86f, 0.5f, -0.56f, 0xff202050); // Chair Leg 2
		m->AddSolidColorBox(-0.8f, 1.0f, -1.1f, -0.86f, 0.0f, -1.04f, 0xff202050); // Chair Leg 2
		m->AddSolidColorBox(-1.4f, 0.97f, -1.05f, -0.8f, 0.92f, -1.10f, 0xff202050); // Chair Back high bar

		for (float f = 3.0f; f <= 6.6f; f += 0.4f)
			m->AddSolidColorBox(-3, 0.0f, f, -2.9f, 1.3f, f + 0.1f, 0xff404040); // Posts

		m->AllocateBuffers();
		Add(m);
	}

	Scene() : numModels(0) {}
	Scene(bool includeIntensiveGPUobject) :
		numModels(0)
	{
		Init(includeIntensiveGPUobject);
	}
	void Release()
	{
		while (numModels-- > 0)
			delete Models[numModels];
	}
	~Scene()
	{
		Release();
	}
};

struct PerlinNoise
{
	int seed;
	int hash[256];

	PerlinNoise() : seed(0) 
	{ 
		gen_hash();
	}

	PerlinNoise(int _seed) : seed(_seed) 
	{
		gen_hash();
	}

	void gen_hash() {
		for (int i = 0; i < 256; ++i) {
			hash[i] = rand() % 256;
		}
	}

	int noise2(int x, int y)
	{
		int tmp = hash[(y + seed) % 256];
		return hash[(tmp + x) % 256];
	}

	float lin_inter(float x, float y, float s)
	{
		return x + s * (y - x);
	}

	float smooth_inter(float x, float y, float s)
	{
		return lin_inter(x, y, s * s * (3 - 2 * s));
	}

	float noise2d(float x, float y)
	{
		int x_int = (int)x;
		int y_int = (int)y;
		float x_frac = x - x_int;
		float y_frac = y - y_int;
		int s = noise2(x_int, y_int);
		int t = noise2(x_int + 1, y_int);
		int u = noise2(x_int, y_int + 1);
		int v = noise2(x_int + 1, y_int + 1);
		float low = smooth_inter((float)s, (float)t, x_frac);
		float high = smooth_inter((float)u, (float)v, x_frac);
		return smooth_inter(low, high, y_frac);
	}

	float perlin2d(float x, float y, float freq, int depth)
	{
		float xa = x*freq;
		float ya = y*freq;
		float amp = 1.0f;
		float fin = 0;
		float div = 0.0;

		int i;
		for (i = 0; i<depth; i++)
		{
			div += 256 * amp;
			fin += noise2d(xa, ya) * amp;
			amp /= 2;
			xa *= 2;
			ya *= 2;
		}

		return fin / div;
	}

	float fractalsum2d(float x, float y, float freq, int depth)
	{
		float xa = x*freq;
		float ya = y*freq;
		float amp = 1.0f;
		float fin = 0;
		float div = 0.0;

		int i;
		for (i = 0; i<depth; i++)
		{
			div += 256 * amp;
			fin += (2.0f*noise2d(xa, ya) - 255.0f) * amp;
			amp /= 2;
			xa *= 2;
			ya *= 2;
		}

		return fin / div;
	}

	float turbulence2d(float x, float y, float freq, int depth)
	{
		float xa = x*freq;
		float ya = y*freq;
		float amp = 1.0f;
		float fin = 0;
		float div = 0.0;

		int i;
		for (i = 0; i<depth; i++)
		{
			div += 256 * amp;
			fin += fabsf(2.0f*noise2d(xa, ya) - 255.0f) * amp;
			amp /= 2;
			xa *= 2;
			ya *= 2;
		}

		return fin / div;
	}
};


//-------------------------------------------------------------------------
// Textures

// Blank, Squares, checkerboard, bricks, wood grain, marble, cellular automata
void ProcTexture_Blank(unsigned long *tex_pixels, unsigned long color)
{
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			tex_pixels[j * 256 + i] = color;
		}
	}
}

void ProcTexture_Tiles(unsigned long *tex_pixels, unsigned int color1, unsigned int color2, int tilewidth, int tilelength, int thickness)
{
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			tex_pixels[j * 256 + i] = ((i % tilewidth) / thickness == 0) || ((j % tilelength) / thickness == 0) ? color1 : color2;
		}
	}
}

void ProcTexture_Checkerboard(unsigned long *tex_pixels, unsigned int color1, unsigned int color2)
{
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			tex_pixels[j * 256 + i] = (((i >> 7) ^ (j >> 7)) & 1) ? color1 : color2;
		}
	}
}

void ProcTexture_Bricks(unsigned long *tex_pixels, unsigned int color1, unsigned int color2)
{
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			tex_pixels[j * 256 + i] = (((j / 4 & 15) == 0) || (((i / 4 & 15) == 0) && ((((i / 4 & 31) == 0) ^ ((j / 4 >> 4) & 1)) == 0)))
				? color1 : color2;
		}
	}
}

void ProcTexture_Perlin(unsigned long *tex_pixels, unsigned int color1, unsigned int color2, int octaves)
{
	PerlinNoise noise(rand() % 65535);

	float v;
	unsigned int r, g, b, a;
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			v = noise.perlin2d((float)i, (float)j, 1.0f/256.0f, octaves);
			a = (unsigned long)((v * (float)((color1 & 0xff000000) >> 24)) + ((1.0f - v) * (float)((color2 & 0xff000000) >> 24)));
			r = (unsigned long)((v * (float)((color1 & 0x00ff0000) >> 16)) + ((1.0f - v) * (float)((color2 & 0x00ff0000) >> 16)));
			g = (unsigned long)((v * (float)((color1 & 0x0000ff00) >>  8)) + ((1.0f - v) * (float)((color2 & 0x0000ff00) >>  8)));
			b = (unsigned long)((v * (float)((color1 & 0x000000ff)      )) + ((1.0f - v) * (float)((color2 & 0x000000ff)      )));
			//unsigned long c = (unsigned long)(v * 255.0f);

			//tex_pixels[j * 256 + i] = COL4(c,c,c,255);
			tex_pixels[j * 256 + i] = COL4(r, g, b, a);
		}
	}
}

void ProcTexture_Turbulence(unsigned long *tex_pixels, unsigned int color1, unsigned int color2, int octaves)
{
	PerlinNoise noise(rand() % 65535);

	float v;
	unsigned int r, g, b, a;
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			v = noise.turbulence2d(i, j, 1.0f / 256.0f, octaves);
			a = (unsigned long)((v * (float)((color1 & 0xff000000) >> 24)) + ((1.0f - v) * (float)((color2 & 0xff000000) >> 24)));
			r = (unsigned long)((v * (float)((color1 & 0x00ff0000) >> 16)) + ((1.0f - v) * (float)((color2 & 0x00ff0000) >> 16)));
			g = (unsigned long)((v * (float)((color1 & 0x0000ff00) >> 8)) + ((1.0f - v) * (float)((color2 & 0x0000ff00) >> 8)));
			b = (unsigned long)((v * (float)((color1 & 0x000000ff))) + ((1.0f - v) * (float)((color2 & 0x000000ff))));
			//unsigned long c = (unsigned long)(v * 255.0f);

			//tex_pixels[j * 256 + i] = COL4(c,c,c,255);
			tex_pixels[j * 256 + i] = COL4(r, g, b, a);
		}
	}
}

void ProcTexture_Fractalsum(unsigned long *tex_pixels, unsigned int color1, unsigned int color2, int octaves)
{
	PerlinNoise noise(rand() % 65535);

	float v;
	unsigned int r, g, b, a;
	for (int j = 0; j < 256; ++j) {
		for (int i = 0; i < 256; ++i) {
			v = fabsf(noise.fractalsum2d(i, j, 1.0f / 256.0f, octaves));
			a = (unsigned long)((v * (float)((color1 & 0xff000000) >> 24)) + ((1.0f - v) * (float)((color2 & 0xff000000) >> 24)));
			r = (unsigned long)((v * (float)((color1 & 0x00ff0000) >> 16)) + ((1.0f - v) * (float)((color2 & 0x00ff0000) >> 16)));
			g = (unsigned long)((v * (float)((color1 & 0x0000ff00) >> 8)) + ((1.0f - v) * (float)((color2 & 0x0000ff00) >> 8)));
			b = (unsigned long)((v * (float)((color1 & 0x000000ff))) + ((1.0f - v) * (float)((color2 & 0x000000ff))));
			//unsigned long c = (unsigned long)(v * 255.0f);

			//tex_pixels[j * 256 + i] = COL4(c,c,c,255);
			tex_pixels[j * 256 + i] = COL4(r, g, b, a);
		}
	}
}

//------------------------------------------------------------------------- 
struct RandomScene
{
	int     numModels;
	Model * Models[32];

	int     numTextures;
	TextureBuffer * Textures[16];

	int     numShaders;
	ShaderFill * Shaders[16];

	void    Add(Model * n)
	{
		Models[numModels++] = n;
	}

	void    Add(TextureBuffer * t)
	{
		Textures[numTextures++] = t;
	}

	void    Add(ShaderFill * s)
	{
		Shaders[numShaders++] = s;
	}

	void Render(Matrix4f view, Matrix4f proj)
	{
		for (int i = 0; i < numModels; ++i)
			Models[i]->Render(view, proj);
	}

	void GenerateShaders() 
	{
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
			"   //FragColor = oColor;\n"
			"}\n";

		GLuint    vshader = CreateShader(GL_VERTEX_SHADER, VertexShaderSrc);
		GLuint    fshader = CreateShader(GL_FRAGMENT_SHADER, FragmentShaderSrc);

		for (int k = 0; k < 8; ++k)
		{
			static unsigned long tex_pixels[256 * 256];
			if (k == MAT_FLOOR) ProcTexture_Checkerboard(tex_pixels, COL3(180, 180, 180), COL3(80, 80, 80));
			if (k == MAT_WALLS) ProcTexture_Bricks(tex_pixels, COL3(60, 60, 60), COL3(180, 180, 180));
			if (k == MAT_CEILING) ProcTexture_Tiles(tex_pixels, COL3(80, 80, 80), COL3(180, 180, 180), 128, 256, 4);
			if (k == MAT_BEAMS) ProcTexture_Perlin(tex_pixels, COL3(255, 255, 255), COL3(0, 0, 0), 8);
			if (k == MAT_TABLE) ProcTexture_Blank(tex_pixels, COL3(255, 255, 255));
			if (k == MAT_CHAIRS) ProcTexture_Blank(tex_pixels, COL3(255, 255, 255));
			if (k == MAT_OBJ1) ProcTexture_Turbulence(tex_pixels, COL3(255, 255, 255), COL3(128, 128, 128), 8);
			if (k == MAT_OBJ2) ProcTexture_Fractalsum(tex_pixels, COL3(128, 128, 128), COL3(255, 255, 255), 8);
			TextureBuffer * generated_texture = new TextureBuffer(nullptr, false, false, Sizei(256, 256), 4, (unsigned char *)tex_pixels, 1);
			ShaderFill * generated_shader = new ShaderFill(vshader, fshader, generated_texture);
			Add(generated_shader);
		}

		glDeleteShader(vshader);
		glDeleteShader(fshader);
	}

	void GenerateModels()
	{
		Model * m;
		unsigned long c1, c2;
		const float wallSegmentWidth = 1.0f;
		const float wallBuffer = 0.5f;
		const float wallThickness = 0.2f;
		const float wallHeight = 4.0f;
		float ceilingHeight = 4.0f; // Random 3.0 - 4.0

		m = new Model(Vector3f(0, 0, 0), Shaders[MAT_OBJ1]);  // Moving box
		c1 = COL3(64, 64, 64);
		m->AddSolidColorBox(0, 0, 0, +1.0f, +1.0f, 1.0f, c1);
		m->AllocateBuffers();
		Add(m);

		// Floor: texture = checkerboard, tiles, solid, perlin; colors = random
		m = new Model(Vector3f(0, 0, 0), Shaders[MAT_FLOOR]);  // Floors
		c1 = COL3(128, 128, 128);
		m->AddSolidColorBox(-10.0f, -0.1f, -wallSegmentWidth, 10.0f, 0.0f, 20.1f, c1); // Main floor
		m->AddSolidColorBox(-15.0f, -6.1f, 18.0f, 15.0f, -6.0f, 30.0f, c1); // Bottom floor
		m->AllocateBuffers();
		Add(m);

		// Ceiling: Texture = tiles, solid, perlin; colors = random; Light fixture or beams random
		m = new Model(Vector3f(0, 0, 0), Shaders[MAT_CEILING]);  // Ceiling
		c1 = COL3(128, 128, 128);
		m->AddSolidColorBox(-10.0f, ceilingHeight, -wallSegmentWidth, 10.0f, ceilingHeight+0.1f, 20.1f, c1);
		m->AllocateBuffers();
		Add(m);

		// Wall texture: tiles, solid, brick; colors = random
		c1 = COL3(128, 128, 128);
		c2 = COL3(128, 128, 128);
		Model * blankWall = new Model(Vector3f(0, 0, 0), Shaders[MAT_WALLS]);
		blankWall->AddSolidColorBox(0.0f, -0.1f, 0.0f, wallSegmentWidth, wallHeight, wallThickness, c1);
		Model * doorWall = new Model(Vector3f(0, 0, 0), Shaders[MAT_WALLS]);
		doorWall->AddSolidColorBox(0.0f, -0.1f, 0.0f, 0.1f*wallSegmentWidth, wallHeight, wallThickness, c1);
		doorWall->AddSolidColorBox(0.1f*wallSegmentWidth, 2.0f, 0.0f, 0.9f*wallSegmentWidth, wallHeight, wallThickness, c1);
		doorWall->AddSolidColorBox(0.9f*wallSegmentWidth, -0.1f, 0.0f, wallSegmentWidth, wallHeight, wallThickness, c1);
		Model * windowWall = new Model(Vector3f(0, 0, 0), Shaders[MAT_WALLS]);
		windowWall->AddSolidColorBox(0.0f, -0.1f, 0.0f, 0.1f*wallSegmentWidth, wallHeight, wallThickness, c1);
		windowWall->AddSolidColorBox(0.1f*wallSegmentWidth, 2.0f, 0.0f, 0.9f*wallSegmentWidth, wallHeight, wallThickness, c1);
		windowWall->AddSolidColorBox(0.1f*wallSegmentWidth, -0.1f, 0.0f, 0.9*wallSegmentWidth, 0.75f, wallThickness, c1);
		windowWall->AddSolidColorBox(0.9f*wallSegmentWidth, -0.1f, 0.0f, wallSegmentWidth, wallHeight, wallThickness, c1);

		int leftWallDistance = 2;
		int rightWallDistance = 4;
		int frontWallDistance = 2;
		int backWallDistance = 1;
		
		// Left wall: distance = near,med,far; door/window loc = (Nothing, Door A, Door B, Window A, Window B, Window AB; Cabinet)
		for (int i = -backWallDistance; i <= frontWallDistance; i++) {
			//m = new Model(blankWall);
			if (i==0) {
				m = new Model(doorWall);
			}
			else {
				m = new Model(blankWall);
			}
			
			m->Rot = Quatf(Matrix4f::RotationY(1.570796f));
			m->Pos = Vector3f(leftWallDistance*wallSegmentWidth+ wallBuffer, 0.0f, (i+1)*wallSegmentWidth);
			m->AllocateBuffers();
			Add(m);
		}
		
		// Right wall: distance = near,med,far; door/window loc = (Nothing, Door A, Door B, Window A, Window B, Window AB)
		for (int i = -backWallDistance; i <= frontWallDistance; i++) {
			m = new Model(blankWall);
			m->Rot = Quatf(Matrix4f::RotationY(-1.570796f));
			m->Pos = Vector3f(-(rightWallDistance*wallSegmentWidth+wallBuffer), 0.0f, i*wallSegmentWidth);
			m->AllocateBuffers();
			Add(m);
		}
		
		// Front wall: distance = near,med,far; door/window loc = (Nothing, Door A, Door B, Window A, Window B, Window AB)
		for (int i = (-rightWallDistance-1); i <= leftWallDistance; i++) {
			if (i == leftWallDistance || i == (-rightWallDistance - 1)) {
				m = new Model(blankWall);
			}
			else {
				m = new Model(windowWall);
			}
			m->Pos = Vector3f(i*wallSegmentWidth, 0.0f, frontWallDistance*wallSegmentWidth+wallBuffer);
			m->AllocateBuffers();
			Add(m);
		}

		// Back wall
		for (int i = (-rightWallDistance); i <= (leftWallDistance+1); i++) {
			m = new Model(blankWall);
			m->Rot = Quatf(Matrix4f::RotationY(3.1415926f));
			m->Pos = Vector3f(i*wallSegmentWidth, 0.0f, -backWallDistance*wallSegmentWidth + wallBuffer);
			m->AllocateBuffers();
			Add(m);
		}

		// Chance of first door is X %; second door is X/2 %. Chance is 0 for subsequent walls.

		// Table: position = randomly on grid (based on wall distances); Orientation = random; Style = random? with or without chair(s); color = random

		Model *table = new Model(Vector3f(0, 0, 1), Shaders[MAT_TABLE]);  // Fixtures & furniture
		c1 = COL3(80, 80, 0);
		table->AddSolidColorBox(-1.8f, 0.8f, 1.0f,  0.0f, 0.7f, 0.0f, c1); // Table
		table->AddSolidColorBox(-1.8f, 0.0f, 0.0f, -1.7f, 0.7f, 0.1f, c1); // Table Leg 
		table->AddSolidColorBox(-1.8f, 0.7f, 1.0f, -1.7f, 0.0f, 0.9f, c1); // Table Leg 
		table->AddSolidColorBox( 0.0f, 0.0f, 1.0f, -0.1f, 0.7f, 0.9f, c1); // Table Leg 
		table->AddSolidColorBox( 0.0f, 0.7f, 0.0f, -0.1f, 0.0f, 0.1f, c1); // Table Leg 
		m = new Model(table);
		m->AllocateBuffers();
		Add(m);

		Model *chair = new Model(Vector3f(-1.4, 0, 0.0), Shaders[MAT_TABLE]);  // Fixtures & furniture
		c1 = COL3(32, 32, 80);
		chair->AddSolidColorBox( 0.0f,  0.5f,  0.0f,   0.6f,   0.55f,  0.6f,  c1); // Chair Set
		chair->AddSolidColorBox( 0.0f,  0.0f,  0.0f,   0.06f,  1.0f,   0.06f, c1); // Chair Leg 1
		chair->AddSolidColorBox( 0.0f,  0.5f,  0.6f,   0.06f,  0.0f,   0.54f, c1); // Chair Leg 2
		chair->AddSolidColorBox( 0.6f,  0.0f,  0.6f,   0.54f,  0.5f,   0.54f, c1); // Chair Leg 2
		chair->AddSolidColorBox( 0.6f,  1.0f,  0.0f,   0.54f,  0.0f,   0.06f, c1); // Chair Leg 2
		chair->AddSolidColorBox( 0.01f, 0.97f, 0.05f,  0.59f,  0.92f,  0.0f,  c1); // Chair Back high bar
		m = new Model(chair);
		m->AllocateBuffers();
		Add(m);
		// Other Chairs: Position = random on grid; orientation = random
		
		// Object 1: sphere, cube, or sandwich; texture: checkerboard, tiles, perlin, bricks; color = random;
		//           start position, end position, rotation angle and speed
		// Object 2: Exists? Type, texture, color. Position and rotation

		delete blankWall;
		delete doorWall;
		delete windowWall;
		delete table;
		delete chair;
	}

	void Init(unsigned int seed)
	{
		srand(seed);
		GenerateShaders();
		GenerateModels();

		// Construct geometry
		/*Model * m = new Model(Vector3f(0, 0, 0), Shaders[2]);  // Moving box
		m->AddSolidColorBox(0, 0, 0, +1.0f, +1.0f, 1.0f, COL3(64,64,64));
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), Shaders[1]);  // Walls
		m->AddSolidColorBox(-10.1f, 0.0f, -20.0f, -10.0f, 4.0f, 20.0f, COL3(128,128,128)); // Left Wall
		m->AddSolidColorBox(-10.0f, -0.1f, -20.1f, 10.0f, 4.0f, -20.0f, COL3(128,128,128)); // Back Wall
		m->AddSolidColorBox(10.0f, -0.1f, -20.0f, 10.1f, 4.0f, 20.0f, COL3(128,128,128)); // Right Wall
		m->AllocateBuffers();
		Add(m);

		m = new Model(Vector3f(0, 0, 0), Shaders[0]);  // Floors
		m->AddSolidColorBox(-10.0f, -0.1f, -20.0f, 10.0f, 0.0f, 20.1f, COL3(128,128,128)); // Main floor
		m->AddSolidColorBox(-15.0f, -6.1f, 18.0f, 15.0f, -6.0f, 30.0f, COL3(128,128,128)); // Bottom floor
		m->AllocateBuffers();
		Add(m);


		m = new Model(Vector3f(0, 0, 0), Shaders[3]);  // Fixtures & furniture
		m->AddSolidColorBox(9.5f, 0.75f, 3.0f, 10.1f, 2.5f, 3.1f, COL3(56,56,56));   // Right side shelf// Verticals
		m->AddSolidColorBox(9.5f, 0.95f, 3.7f, 10.1f, 2.75f, 3.8f, COL3(56, 56, 56));   // Right side shelf
		m->AddSolidColorBox(9.55f, 1.20f, 2.5f, 10.1f, 1.30f, 3.75f, COL3(56, 56, 56)); // Right side shelf// Horizontals
		m->AddSolidColorBox(9.55f, 2.00f, 3.05f, 10.1f, 2.10f, 4.2f, COL3(56, 56, 56)); // Right side shelf
		m->AddSolidColorBox(5.0f, 1.1f, 20.0f, 10.0f, 1.2f, 20.1f, COL3(56, 56, 56));   // Right railing   
		m->AddSolidColorBox(-10.0f, 1.1f, 20.0f, -5.0f, 1.2f, 20.1f, COL3(56, 56, 56));   // Left railing  
		for (float f = 5.0f; f <= 9.0f; f += 1.0f)
		{
			m->AddSolidColorBox(f, 0.0f, 20.0f, f + 0.1f, 1.1f, 20.1f, COL3(80, 80, 80));// Left Bars
			m->AddSolidColorBox(-f, 1.1f, 20.0f, -f - 0.1f, 0.0f, 20.1f, COL3(80, 80, 80));// Right Bars
		}
		m->AddSolidColorBox(-1.8f, 0.8f, 1.0f, 0.0f, 0.7f, 0.0f, COL3(80, 80, 0)); // Table
		m->AddSolidColorBox(-1.8f, 0.0f, 0.0f, -1.7f, 0.7f, 0.1f, COL3(80, 80, 0)); // Table Leg 
		m->AddSolidColorBox(-1.8f, 0.7f, 1.0f, -1.7f, 0.0f, 0.9f, COL3(80, 80, 0)); // Table Leg 
		m->AddSolidColorBox(0.0f, 0.0f, 1.0f, -0.1f, 0.7f, 0.9f, COL3(80, 80, 0)); // Table Leg 
		m->AddSolidColorBox(0.0f, 0.7f, 0.0f, -0.1f, 0.0f, 0.1f, COL3(80, 80, 0)); // Table Leg 
		m->AddSolidColorBox(-1.4f, 0.5f, -1.1f, -0.8f, 0.55f, -0.5f, COL3(32, 32, 80)); // Chair Set
		m->AddSolidColorBox(-1.4f, 0.0f, -1.1f, -1.34f, 1.0f, -1.04f, COL3(32, 32, 80)); // Chair Leg 1
		m->AddSolidColorBox(-1.4f, 0.5f, -0.5f, -1.34f, 0.0f, -0.56f, COL3(32, 32, 80)); // Chair Leg 2
		m->AddSolidColorBox(-0.8f, 0.0f, -0.5f, -0.86f, 0.5f, -0.56f, COL3(32, 32, 80)); // Chair Leg 2
		m->AddSolidColorBox(-0.8f, 1.0f, -1.1f, -0.86f, 0.0f, -1.04f, COL3(32, 32, 80)); // Chair Leg 2
		m->AddSolidColorBox(-1.4f, 0.97f, -1.05f, -0.8f, 0.92f, -1.10f, COL3(32, 32, 80)); // Chair Back high bar

		for (float f = 3.0f; f <= 6.6f; f += 0.4f)
			m->AddSolidColorBox(-3, 0.0f, f, -2.9f, 1.3f, f + 0.1f, COL3(64, 64, 64)); // Posts

		m->AllocateBuffers();
		Add(m);*/
	}

	RandomScene() : numModels(0), numShaders(0), numTextures(0) {}
	RandomScene(unsigned int seed) :
		numModels(0), numShaders(0), numTextures(0)
	{
		Init(seed);
	}
	void Release()
	{
		while (numModels-- > 0)
			delete Models[numModels];
		while (numShaders-- > 0)
			delete Shaders[numShaders];
		while (numTextures-- > 0)
			delete Textures[numTextures];
	}
	~RandomScene()
	{
		Release();
	}
};