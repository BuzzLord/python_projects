from collections import namedtuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GL.EXT.texture_filter_anisotropic import GL_TEXTURE_MAX_ANISOTROPY_EXT
from OpenGL.GL.ARB.depth_buffer_float import GL_DEPTH_COMPONENT32F
from gl_math import *

active_texture = { 0: GL_TEXTURE0, 1: GL_TEXTURE1, 2: GL_TEXTURE2, 3: GL_TEXTURE3,
                   4: GL_TEXTURE4, 5: GL_TEXTURE5, 6: GL_TEXTURE6, 7: GL_TEXTURE7 }

default_texnames = { 0: "Texture0", 1: "Texture1", 2: "Texture2", 3: "Texture3",
                     4: "Texture4", 5: "Texture5", 6: "Texture6", 7: "Texture7" }

Color = namedtuple('Color', ['r','g','b','a'])

basic_vertex_shader_src = """
            #version 150
            uniform mat4 matWVP;
            in vec4 Position;
            //in vec4 Color;
            in vec2 TexCoord;
            out vec2 oTexCoord;
            out vec4 oColor;
            void main() {
                gl_Position = (matWVP * Position);
                oTexCoord = TexCoord;
                //oColor.rgb = pow(Color.rgb, vec3(2.2));
                //oColor.a = Color.a;
                oColor = vec4(1.0);
            }"""

basic_fragment_shader_src = """
            #version 150
            uniform sampler2D Texture0;
            in vec4 oColor;
            in vec2 oTexCoord;
            out vec4 FragColor;
            void main() {
                //FragColor = oColor * texture2D(Texture0, oTexCoord);
                FragColor = texture2D(Texture0, oTexCoord);
                //FragColor = FragColor * 0.25 + vec4(0.75,0.0,0.5,0.0);
                //FragColor = vec4(0.75,0.0,0.5,1.0);
                FragColor.a = 1.0;
                //FragColor = oColor;
            }"""

warp_frag_shader_src ="""
            #version 150
            uniform sampler2D Texture0;
            uniform sampler2D Texture1;
            uniform sampler2D Texture2;
            uniform sampler2D Texture3;
            uniform sampler2D Texture4;
            in      vec4      oColor;
            in      vec2      oTexCoord;
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
                    c = texture2D(Texture2, uv);
                } else if (faceIndex == 1.0) {
                    c = texture2D(Texture1, uv);
                } else if (faceIndex == 2.0) {
                    c = texture2D(Texture4, uv);
                } else if (faceIndex == 3.0) {
                    c = texture2D(Texture3, uv);
                } else if (faceIndex == 4.0) {
                    c = texture2D(Texture0, uv);
                } else {
                    c = vec4(0.0);
                }
                return c;
            }
            void main() {
                vec4 AccColor = vec4(0.0);
                AccColor += 0.40*sampleCubeColor(toSphere(oTexCoord));
                // Pixel width for 2048 is 0.000488. Offsets a little under half pixel width.
                AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(0.0002,0.0002)));
                AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(0.0002,-0.0002)));
                AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(-0.0002,0.0002)));
                AccColor += 0.15*sampleCubeColor(toSphere(oTexCoord+vec2(-0.0002,-0.0002)));
                FragColor = AccColor;
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
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
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

        glViewport(0, 0, self.texSize[0], self.texSize[1]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_FRAMEBUFFER_SRGB);

    def set_render_surface(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texId, 0)

    def unset_render_surface(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0)


class ShaderFill:

    def __init__(self, vert, frag):
        self.texture = {}
        self.float_uniforms = {}
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

    def add_matrix_uniform(self, name, matrix):
        self.matrix_uniforms[name] = matrix


#def color(red, green, blue):
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


def texture_checkerboard(color1, color2):
    tex_pixels = np.ones((256, 256, 4), dtype=np.uint8)
    for i in range(len(tex_pixels)):
        for j in range(len(tex_pixels[i])):
            if (((i >> 7) ^ (j >> 7)) & 1):
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


def texture_bricks(color1, color2):
    tex_pixels = np.ones((256, 256, 4), dtype=np.uint8)
    for i in range(len(tex_pixels)):
        for j in range(len(tex_pixels[i])):
            if (((i >> 2) & 15) == 0) or ((((j >> 2) & 15) == 0) and (((((j >> 2) & 31) == 0) ^ ((i >> 2) >> 4) & 1) == 0)):
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



