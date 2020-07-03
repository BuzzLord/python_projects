from collections import namedtuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT
from OpenGL.GL.ARB.depth_buffer_float import GL_DEPTH_COMPONENT32F
from generator.gl_math import *

active_texture = { 0: GL_TEXTURE0, 1: GL_TEXTURE1, 2: GL_TEXTURE2, 3: GL_TEXTURE3,
                   4: GL_TEXTURE4, 5: GL_TEXTURE5, 6: GL_TEXTURE6, 7: GL_TEXTURE7,
                   8: GL_TEXTURE8, 9: GL_TEXTURE9, 10: GL_TEXTURE10, 11: GL_TEXTURE11}

default_texnames = { 0: "Texture0", 1: "Texture1", 2: "Texture2", 3: "Texture3",
                     4: "Texture4", 5: "Texture5", 6: "Texture6", 7: "Texture7",
                     8: "Texture8", 9: "Texture9", 10: "Texture10", 11: "Texture11"}

Color = namedtuple('Color', ['r','g','b','a'])

basic_vertex_shader_src = """
            #version 150
            uniform mat4 Model;
            uniform mat4 View;
            uniform mat4 Proj;
            uniform mat3 NormalMatrix;
            
            uniform vec3 LightPos0;
            uniform vec3 LightPos1;
            
            in vec4 Position;
            in vec2 TexCoord;
            in vec3 Normal;
            //in vec4 Color;
            
            out vec2 oTexCoord;
            out vec3 oNormal;
            out vec4 oColor;
            out vec3 oLightDir0;
            out vec3 oLightDir1;
            
            void main() {
                gl_Position = (Proj * View * Model * Position);
                oTexCoord = TexCoord;
                oNormal = NormalMatrix * Normal;
                vec3 EyeCam = vec3(0.0,0.0,0.0) - (View * Model * Position).xyz;
                vec3 LightCam0 = (View * vec4(LightPos0,1.0)).xyz;
                vec3 LightCam1 = (View * vec4(LightPos1,1.0)).xyz;
                oLightDir0 = LightCam0 + EyeCam;
                oLightDir1 = LightCam1 + EyeCam;
                
                //oNormal = Normal;
                //oColor.rgb = pow(Color.rgb, vec3(2.2));
                //oColor.a = Color.a;
                oColor = vec4(1.0);
            }"""

basic_fragment_shader_src = """
            #version 150
            uniform sampler2D Texture0;
            in vec4 oColor;
            in vec2 oTexCoord;
            in vec3 oNormal;
            in vec3 oLightDir0;
            in vec3 oLightDir1;
            out vec4 FragColor;
            void main() {
                FragColor = texture2D(Texture0, oTexCoord);
                FragColor.a = 1.0;
                //FragColor = oColor;
            }"""

colormod_fragment_shader_src = """
            #version 150
            uniform sampler2D Texture0;
            uniform vec3 Color0;
            uniform vec3 Color1;
            
            uniform float LightPower0;
            uniform float LightPower1;
            uniform float AmbientLight;
            
            in vec4 oColor;
            in vec2 oTexCoord;
            in vec3 oNormal;
            in vec3 oLightDir0;
            in vec3 oLightDir1;
            out vec4 FragColor;
            void main() {
                vec4 TexColor = texture2D(Texture0, oTexCoord);
                float lightPower0 = LightPower0 / length(oLightDir0);
                float nDotVP0 = max(0.0, dot(normalize(oNormal), normalize(oLightDir0)));
                float lightPower1 = LightPower1 / length(oLightDir1);
                float nDotVP1 = max(0.0, dot(normalize(oNormal), normalize(oLightDir1)));
                float lightIntensity = max(AmbientLight, nDotVP0 * min(1.0,lightPower0) + nDotVP1 * min(1.0,lightPower1));
                FragColor.rgb = lightIntensity * (Color0 * TexColor.rgb + Color1 * (vec3(1.0) - TexColor.rgb));
                //FragColor.rgb = abs(oNormal);
                //FragColor.rgb = max(vec3(0.0), oNormal);
                //FragColor.rg = oTexCoord;
                //FragColor.b = 0.0;

                FragColor.a = 1.0;
                //FragColor = oColor;
            }"""

warp_frag_shader_src = """
            #version 150
            uniform sampler2D Texture0;  // color 0
            uniform sampler2D Texture1;  // depth 0
            uniform sampler2D Texture2;  // color 1
            uniform sampler2D Texture3;  // depth 1
            uniform sampler2D Texture4;  // color 2
            uniform sampler2D Texture5;  // depth 2
            uniform sampler2D Texture6;  // color 3
            uniform sampler2D Texture7;  // depth 3
            uniform sampler2D Texture8;  // color 4
            uniform sampler2D Texture9;  // depth 4
            in      vec4      oColor;
            in      vec2      oTexCoord;
            in      vec3      oNormal;
            out     vec4      FragColor;
            vec2 sampleCube(const vec3 v,out float faceIndex) {
                vec3 vAbs = abs(v);
                float ma;
                vec2 uv;
                if (vAbs.z >= vAbs.x && vAbs.z >= vAbs.y) {
                    faceIndex = v.z < 0.0 ? 5.0 : 4.0;
                    ma = 0.5 / vAbs.z;
                    uv = vec2(v.z < 0.0 ? -v.x : v.x, -v.y);
                } else if (vAbs.y >= vAbs.x) {
                    faceIndex = v.y < 0.0 ? 3.0 : 2.0;
                    ma = 0.5 / vAbs.y;
                    uv = vec2(v.x, v.y < 0.0 ? -v.z : v.z);
                } else {
                    faceIndex = v.x < 0.0 ? 1.0 : 0.0;
                    ma = 0.5 / vAbs.x;
                    uv = vec2(v.x < 0.0 ? v.z : -v.z, -v.y);
                }
                return uv * ma + 0.5;
            }
            vec3 toSphere(const vec2 texCoord) {
                vec2 theta = 1.570796 * ((texCoord) * 2.0 - vec2(1.0));
                float cosphi = cos(theta.y);
                vec3 v = vec3(cosphi * sin(-theta.x), sin(theta.y), cosphi * cos(-theta.x));
                return v;
            }
            vec4 sampleCubeColor(vec3 v) {
                vec4 c;
                float faceIndex;
                vec2 uv = sampleCube(v, faceIndex);
                if (faceIndex == 0.0) {
                    c = texture2D(Texture4, uv);
                } else if (faceIndex == 1.0) {
                    c = texture2D(Texture2, uv);
                } else if (faceIndex == 2.0) {
                    c = texture2D(Texture8, uv);
                } else if (faceIndex == 3.0) {
                    c = texture2D(Texture6, uv);
                } else if (faceIndex == 4.0) {
                    c = texture2D(Texture0, uv);
                } else {
                    c = vec4(0.0);
                }
                return c;
            }
            vec4 sampleCubeDepth(vec3 v) {
                vec4 c;
                float faceIndex;
                vec2 uv = sampleCube(v, faceIndex);
                if (faceIndex == 0.0) {
                    c = texture2D(Texture5, uv);
                } else if (faceIndex == 1.0) {
                    c = texture2D(Texture3, uv);
                } else if (faceIndex == 2.0) {
                    c = texture2D(Texture9, uv);
                } else if (faceIndex == 3.0) {
                    c = texture2D(Texture7, uv);
                } else if (faceIndex == 4.0) {
                    c = texture2D(Texture1, uv);
                } else {
                    c = vec4(0.0);
                }
                // Adjust depth based on UV from cube
                vec2 d =  2.0 * abs(vec2(0.5) - uv);
                float scale = 1.0 / sqrt(1.0 + d.s*d.s + d.t*d.t);
                return pow(c, vec4(scale));
            }
            vec4 sampleFivePoints(vec2 TexCoord) {
                vec4 AccColor = vec4(0.0);
                AccColor += 0.40*sampleCubeColor(toSphere(TexCoord));
                // Pixel width for 2048 is 0.000488.
                float SubStep = 0.0004;
                AccColor += 0.15*sampleCubeColor(toSphere(TexCoord+vec2(SubStep,SubStep)));
                AccColor += 0.15*sampleCubeColor(toSphere(TexCoord+vec2(SubStep,-SubStep)));
                AccColor += 0.15*sampleCubeColor(toSphere(TexCoord+vec2(-SubStep,SubStep)));
                AccColor += 0.15*sampleCubeColor(toSphere(TexCoord+vec2(-SubStep,-SubStep)));
                return AccColor;
            }
            vec4 sampleDepth(vec2 TexCoord) {
                return sampleCubeDepth(toSphere(TexCoord));
            }
            vec4 sampleNinePointsIrregular(vec2 TexCoord) {
                vec4 AccColor = vec4(0.0);
                AccColor += 0.36*sampleCubeColor(toSphere(TexCoord));
                // Pixel width for 2048 is 0.000488.
                float SubStep = 0.00006;
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(3.0*SubStep,SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(2.0*SubStep,2.5*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-SubStep,3.0*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-2.5*SubStep,2.0*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-3.0*SubStep,-SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-2.0*SubStep,-2.5*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(SubStep,-3.0*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(2.5*SubStep,-2.0*SubStep)));
                return AccColor;
            }
            vec4 sampleNinePointsRegular(vec2 TexCoord) {
                vec4 AccColor = vec4(0.0);
                AccColor += 0.36*sampleCubeColor(toSphere(TexCoord));
                // Pixel width for 2048 is 0.000488. Offsets a little under half pixel width.
                float SubStep = 0.0005;
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(SubStep,0.0)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(0.707*SubStep,0.707*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(0.0,SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-0.707*SubStep,0.707*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-SubStep,0.0)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(-0.707*SubStep,-0.707*SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(0.0,-SubStep)));
                AccColor += 0.08*sampleCubeColor(toSphere(TexCoord+vec2(0.707*SubStep,-0.707*SubStep)));
                return AccColor;
            }
            void main() {
                vec4 color = sampleFivePoints(oTexCoord);
                //vec4 depth = pow(sampleDepth(oTexCoord), vec4(8.0)); 
                //FragColor = vec4(color.rgb, depth.r);
                FragColor = vec4(color.rgb, 1.0);
            }"""


class DepthBuffer:
    def __init__(self, size, use_float=True):
        self.texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texId)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        if use_float:
            internal_format = GL_DEPTH_COMPONENT32F
            tex_type = GL_FLOAT
        else:
            internal_format = GL_DEPTH_COMPONENT24
            tex_type = GL_UNSIGNED_INT

        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, size[0], size[1], 0, GL_DEPTH_COMPONENT, tex_type, None)

    def destroy(self):
        if self.texId is not None:
            glDeleteTextures(1, self.texId)
            self.texId = None


class TextureBuffer:
    def __init__(self, render_target, size, data, mip_levels=1, clamp=None):
        self.texId = None
        self.fboId = None
        self.texSize = size

        self.texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texId)
        if render_target:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            if clamp is None:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, clamp)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, clamp)

        else:
            if mip_levels > 1:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            if clamp is None:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            else:
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, clamp)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, clamp)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, self.texSize[0], self.texSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        if mip_levels > 1:
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0)

        if render_target:
            self.fboId = glGenFramebuffers(1)

    def destroy(self):
        if self.texId is not None:
            glDeleteTextures(1, self.texId)
            self.texId = None
        if self.fboId is not None:
            glDeleteFramebuffers(1, self.fboId)
            self.fboId = None

    def get_size(self):
        return self.texSize

    def set_and_clear_render_surface(self, dbuffer):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texId, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dbuffer.texId, 0)

        glViewport(0, 0, self.texSize[0], self.texSize[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_FRAMEBUFFER_SRGB)

    def set_render_surface(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texId, 0)

    def unset_render_surface(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0)

    def copy_data(self, data, bind=True):
        if bind:
            glBindTexture(GL_TEXTURE_2D, self.texId)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, self.texSize[0], self.texSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)


class ShaderFill:

    def __init__(self, vert, frag):
        self.texture = {}
        self.uniform_locations = {}
        self.float_uniforms = {}
        self.vec3_uniforms = {}
        self.matrix_uniforms = {}
        self.destroy_textures = True
        vs = shaders.compileShader(vert, GL_VERTEX_SHADER)
        fs = shaders.compileShader(frag, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vs, fs)

    def destroy(self):
        if self.program is not None:
            glDeleteProgram(self.program)
            self.program = None
        if self.destroy_textures:
            for t in self.texture.values():
                t.destroy()
        self.texture = {}

    def add_texture_buffer(self, tex, name=None):
        _name = name
        if _name is None:
            num = len(self.texture.keys())
            _name = default_texnames[num]
        self.texture[_name] = tex

    def add_float_uniform(self, name, value):
        self.float_uniforms[name] = value

    def add_vec3_uniform(self, name, values):
        self.vec3_uniforms[name] = values

    def add_matrix_uniform(self, name, matrix):
        self.matrix_uniforms[name] = matrix

    def get_uniform_location(self, name):
        if name in self.uniform_locations:
            return self.uniform_locations[name]
        location = glGetUniformLocation(self.program, name)
        if location >= 0:
            self.uniform_locations[name] = location
        return location

# def color(red, green, blue):
#    return np.uint32(np.clip(np.uint32(red), 0, 255) << 16) + \
#           np.uint32(np.clip(np.uint32(green), 0, 255) << 8) + \
#           np.uint32(np.clip(np.uint32(blue), 0, 255))


def texture_blank(color):
    tex_pixels = np.ones((256,256,4), dtype=np.uint8)
    for i in range(len(tex_pixels)):
        for j in range(len(tex_pixels[i])):
            tex_pixels[i][j][0] = color.r
            tex_pixels[i][j][1] = color.g
            tex_pixels[i][j][2] = color.b
            tex_pixels[i][j][3] = color.a
    return tex_pixels


def texture_tiles(color1, color2, tile_width, tile_length, thickness):
    tex_pixels = np.ones((256, 256, 4), dtype=np.uint8)
    for i in range(len(tex_pixels)):
        for j in range(len(tex_pixels[i])):
            if ((j % tile_width) / thickness == 0) or ((i % tile_length) / thickness == 0):
                tex_pixels[i][j][0] = color2.r
                tex_pixels[i][j][1] = color2.g
                tex_pixels[i][j][2] = color2.b
                tex_pixels[i][j][3] = color2.a
            else:
                tex_pixels[i][j][0] = color1.r
                tex_pixels[i][j][1] = color1.g
                tex_pixels[i][j][2] = color1.b
                tex_pixels[i][j][3] = color1.a
    return tex_pixels


def texture_checkerboard(color1, color2, sfactor=(1,1)):
    tex_pixels = np.ones((256, 256, 4), dtype=np.uint8)
    for i in range(len(tex_pixels)):
        for j in range(len(tex_pixels[i])):
            if ((i >> (8 - int(sfactor[0]))) ^ (j >> (8 - int(sfactor[1])))) & 1:
                tex_pixels[i][j][0] = color1.r
                tex_pixels[i][j][1] = color1.g
                tex_pixels[i][j][2] = color1.b
                tex_pixels[i][j][3] = color1.a
            else:
                tex_pixels[i][j][0] = color2.r
                tex_pixels[i][j][1] = color2.g
                tex_pixels[i][j][2] = color2.b
                tex_pixels[i][j][3] = color2.a
    return tex_pixels


def texture_bricks(color1, color2, sfactor=(2,2)):
    tex_pixels = np.ones((256, 256, 4), dtype=np.uint8)
    for i in range(len(tex_pixels)):
        for j in range(len(tex_pixels[i])):
            if (((i >> (4-int(sfactor[0]))) & 15) == 0) or ((((j >> (4-int(sfactor[1]))) & 15) == 0) and (((((j >> (4-int(sfactor[1]))) & 31) == 0) ^ ((i >> (4-int(sfactor[0]))) >> 4) & 1) == 0)):
                tex_pixels[i][j][0] = color1.r
                tex_pixels[i][j][1] = color1.g
                tex_pixels[i][j][2] = color1.b
                tex_pixels[i][j][3] = color1.a
            else:
                tex_pixels[i][j][0] = color2.r
                tex_pixels[i][j][1] = color2.g
                tex_pixels[i][j][2] = color2.b
                tex_pixels[i][j][3] = color2.a
    return tex_pixels



