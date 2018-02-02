from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
import numpy as np
from pandas._libs import window


active_texture = {}
active_texture[0] = GL_TEXTURE0
active_texture[1] = GL_TEXTURE1
active_texture[2] = GL_TEXTURE2
active_texture[3] = GL_TEXTURE3
active_texture[4] = GL_TEXTURE4
active_texture[5] = GL_TEXTURE5
active_texture[6] = GL_TEXTURE6
active_texture[7] = GL_TEXTURE7

default_texnames = {}
default_texnames[0] = "Texture0"
default_texnames[1] = "Texture1"
default_texnames[2] = "Texture2"
default_texnames[3] = "Texture3"
default_texnames[4] = "Texture4"
default_texnames[5] = "Texture5"
default_texnames[6] = "Texture6"
default_texnames[7] = "Texture7"


class DepthBuffer:

    def __init__(self, size):
        self.texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texId)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        internal_format = GL_DEPTH_COMPONENT32F
        tex_type = GL_FLOAT

        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size, size, 0, internal_format, tex_type, 0)

    def destroy(self):
        if self.texId > 0:
            glDeleteTextures(1, self.texId)
            self.texId = 0


class TextureBuffer:
    def __init__(self, render_target, size, mip_levels, data):
        self.texId = 0
        self.fboId = 0
        self.texSize = size

        self.texId = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texId)
        if render_target:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, self.texSize[0], self.texSize[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

        if mip_levels > 1:
            glGenerateMipmap(GL_TEXTURE_2D)

        self.fboId = glGenFramebuffers(1)

    def destroy(self):
        if self.texId > 0:
            glDeleteTextures(1, self.texId)
            self.texId = 0
        if self.fboId > 0:
            glDeleteFramebuffers(1, self.fboId)
            self.fboId = 0

    def get_size(self):
        return self.texSize

    def set_and_clear_render_surface(self, dbuffer):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texId, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dbuffer.texId, 0)

    def set_render_surface(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texId, 0)

    def unset_render_surface(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fboId)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0)


class ShaderFill:

    def __init__(self, vert, frag, tex_buffer):
        self.texture = {}
        self.float_uniforms = {}
        self.matrix_uniforms = {}
        self.destroy_textures = True
        self.add_texture_buffer(tex_buffer)
        vs = shaders.compileShader(vert, GL_VERTEX_SHADER)
        fs = shaders.compileShader(frag, GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vs, fs)

    def destroy(self):
        if self.program > 0:
            glDeleteProgram(self.program)
            self.program = 0
        if self.destroy_textures:
            for t in self.texture.values():
                t.destroy()
        self.texture = {}

    def add_texture_buffer(self, tex):
        num = len(self.texture.keys())
        self.add_texture_buffer(default_texnames[num], tex)

    def add_texture_buffer(self, name, tex):
        self.texture[name] = tex

    def add_float_uniform(self, name, value):
        self.float_uniforms[name] = value

    def add_matrix_uniform(self, name, matrix):
        self.matrix_uniforms[name] = matrix


class Model:

    def __init__(self, pos, shader):
        self.pos = np.asarray(pos, dtype='float32')
        self.rot = np.zeros((4,1), dtype='float32')
        self.mat = np.identity(4, dtype='float32')
        self.shader = shader
        self.vertex_buffer = None
        self.index_buffer = None
        self.vertices = []
        self.indices = []

    def copy(self):
        model_copy = Model(np.copy(self.pos), self.shader)
        model_copy.rot = np.copy(self.rot)
        model_copy.mat = np.copy(self.mat)
        model_copy.vertices = list(self.vertices)
        model_copy.indices = list(self.indices)
        return model_copy

    def get_matrix(self):
        # this is totally wrong. Turn quaternion "rot" into 4x4 rotation matrix, left mult by pos translation
        self.mat = np.dot(self.pos, self.rot)
        return self.mat

    def add_vertex(self, v):
        self.vertices = np.append(self.vertices, v, axis=0)

    def add_index(self, i):
        self.indices = np.append(self.indices, i, axis=0)

    def allocate_buffers(self):
        self.vertex_buffer = vbo.VBO(self.vertices)
        self.index_buffer = vbo.VBO(self.indices, target=GL_ELEMENT_ARRAY_BUFFER)

    def add_solid_color_box(self, p1, p2, c, shading):
        verts = np.array([
                        # Top
                          [p1[0],p2[1],p1[2],p1[2],p1[0]],[p2[0],p2[1],p1[2],p1[2],p2[0]],
                          [p2[0],p2[1],p2[2],p2[2],p2[0]],[p1[0],p2[1],p2[2],p2[2],p1[0]],
                        # Bottom
                          [p1[0],p1[1],p1[2],p1[2],p1[0]],[p2[0],p1[1],p1[2],p1[2],p2[0]],
                          [p2[0],p1[1],p2[2],p2[2],p2[0]],[p1[0],p1[1],p2[2],p2[2],p1[0]],
                        # Left
                          [p1[0],p1[1],p2[2],p2[2],p1[1]],[p1[0],p1[1],p1[2],p1[2],p1[1]],
                          [p1[0],p2[1],p1[2],p1[2],p2[1]],[p1[0],p2[1],p2[2],p2[2],p2[1]],
                        # Right
                          [p2[0],p1[1],p2[2],p2[2],p1[1]],[p2[0],p1[1],p1[2],p1[2],p1[1]],
                          [p2[0],p2[1],p1[2],p1[2],p2[1]],[p2[0],p2[1],p2[2],p2[2],p2[1]],
                        # Back
                          [p1[0],p1[1],p1[2],p1[0],p1[1]],[p2[0],p1[1],p1[2],p2[0],p1[1]],
                          [p2[0],p2[1],p1[2],p2[0],p2[1]],[p1[0],p2[1],p1[2],p1[0],p2[1]],
                        # Front
                          [p1[0],p1[1],p2[2],p1[0],p1[1]],[p2[0],p1[1],p2[2],p2[0],p1[1]],
                          [p2[0],p2[1],p2[2],p2[0],p2[1]],[p1[0],p2[1],p2[2],p1[0],p2[1]]],
                        dtype=np.float32)

        indices = np.array([0, 1, 3, 3, 1, 2,
                            5, 4, 6, 6, 4, 7,
                            8, 9, 11, 11, 9, 10,
                            13, 12, 14, 14, 12, 15,
                            16, 17, 19, 19, 17, 18,
                            21, 20, 22, 22, 20, 23], dtype=np.int32)

        vertOffset = len(self.vertices)
        for i in indices:
            self.add_index(i + vertOffset)

        for v in verts:
            pos = v[0:3]
            uv = v[3:5]

            if shading:
                dist1 = np.linalg.norm(pos - np.array([-2, 4, -2],dtype=np.float32))
                dist2 = np.linalg.norm(pos - np.array([3, 4, -3], dtype=np.float32))
                dist3 = np.linalg.norm(pos - np.array([-4, 3, 25], dtype=np.float32))
                bri = np.random.rand(1)[0]*160
                B = np.uint32(float((c >> 16) & 0xff) * (bri + 192.0 * (0.65 + (8 / dist1) + (1 / dist2) + (4 / dist3))) / 255.0)
                G = np.uint32(float((c >> 8) & 0xff) * (bri + 192.0 * (0.65 + (8 / dist1) + (1 / dist2) + (4 / dist3))) / 255.0)
                R = np.uint32(float((c >> 0) & 0xff) * (bri + 192.0 * (0.65 + (8 / dist1) + (1 / dist2) + (4 / dist3))) / 255.0)
                col = np.array([(c & 0xff000000) + (np.clip(R, 0, 255) << 16) + (np.clip(G, 0, 255) << 8) + (np.clip(B, 0, 255))])
            else:
                col = np.array([c])

            col.dtype=np.float32
            self.add_vertex(np.append(np.append(pos,uv),col))

    def add_solid_color_box(self, p1, p2, c):
        self.add_solid_color_box(p1, p2, c, True)

    def add_oriented_quad(self, p1, p2, c):
        verts = np.array([
            [p1[0], p1[1], p1[2], 0.0, 0.0], [p2[0], p1[1], p2[2], 1.0, 0.0],
            [p2[0], p2[1], p2[2], 1.0, 1.0], [p1[0], p2[1], p1[2], 0.0, 1.0]],
            dtype=np.float32)

        indices = np.array([ 0, 3, 1, 1, 3, 2], dtype=np.int32)

        vertOffset = len(self.vertices)
        for i in indices:
            self.add_index(i + vertOffset)

        for v in verts:
            col = np.array([0xffffffff], dtype=np.uint32)
            col.dtype=np.float32
            self.add_vertex(np.append(v[0:5],col))

    def render(self, view, proj):
        combined = np.dot(proj, np.dot(view, self.get_matrix()))

        glUseProgram(self.shader.program)
        try:
            # Will this work?  combined might need casting
            glUniform4fv(glGetUniformLocation(self.shader.program, "matWVP"), 1, GL_TRUE, combined)

            for name in self.shader.matrix_uniforms.keys():
                glUniformMatrix4fv(glGetUniformLocation(self.shader.program, name), 1, GL_TRUE, self.shader.matrix_uniforms[name])

            for i, texname in enumerate(self.shader.texture.keys()):
                glUniform1i(glGetUniformLocation(self.shader.program, texname), i)
                glActiveTexture(active_texture[i])
                glBindTexture(GL_TEXTURE_2D, self.shader.texture[texname].texId)

            for name in self.shader.float_uniforms.keys():
                glUniform1f(glGetUniformLocation(self.shader.program, name), self.shader.float_uniforms[name])

            try:
                self.vertex_buffer.bind()
                self.index_buffer.bind()

                pos_loc = glGetAttribLocation(self.shader.program, "Position")
                col_loc = glGetAttribLocation(self.shader.program, "Color")
                uv_loc = glGetAttribLocation(self.shader.program, "TexCoord")

                glEnableVertexAttribArray(pos_loc)
                glEnableVertexAttribArray(uv_loc)
                glEnableVertexAttribArray(col_loc)

                glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 4 * 6, self.vertex_buffer)
                glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 4 * 6, self.vertex_buffer+12)
                glVertexAttribPointer(col_loc, 4, GL_UNSIGNED_INT, GL_TRUE, 4 * 6, self.vertex_buffer+20)

            finally:
                glDisableVertexAttribArray(pos_loc)
                glDisableVertexAttribArray(uv_loc)
                glDisableVertexAttribArray(col_loc)
                self.index_buffer.unbind()
                self.vertex_buffer.unbind()
        finally:
            glUseProgram(0)

def color(red, green, blue):
    return np.uint32(0xff000000) + \
           np.uint32(np.clip(np.uint32(red),0,255) << 16) + \
           np.uint32(np.clip(np.uint32(green),0,255) << 8) + \
           np.uint32(np.clip(np.uint32(blue), 0, 255))

def color(red, green, blue, alpha):
    return np.uint32(np.clip(np.uint32(alpha), 0, 255) << 24) + \
           np.uint32(np.clip(np.uint32(red),0,255) << 16) + \
           np.uint32(np.clip(np.uint32(green),0,255) << 8) + \
           np.uint32(np.clip(np.uint32(blue), 0, 255))

def texture_blank(color):
    tex_pixels = color * np.ones((256,256), dtype=np.uint32)

def texture_tiles(color1, color2, tilewidth, tilelength, thickness):
    tex_pixels = np.ones((256, 256), dtype=np.uint32)
    for i in len(tex_pixels):
        for j in len(tex_pixels[i]):
            if ((i % tilewidth) / thickness == 0) or ((j % tilelength) / thickness == 0):
                tex_pixels[i][j] = color1
            else:
                tex_pixels[i][j] = color2

def texture_checkerboard(color1, color2):
    tex_pixels = np.ones((256, 256), dtype=np.uint32)
    for i in len(tex_pixels):
        for j in len(tex_pixels[i]):
            if (((i >> 7) ^ (j >> 7)) & 1):
                tex_pixels[i][j] = color1
            else:
                tex_pixels[i][j] = color2

def texture_bricks(color1, color2):
    tex_pixels = np.ones((256, 256), dtype=np.uint32)
    for i in len(tex_pixels):
        for j in len(tex_pixels[i]):
            if (((j / 4 & 15) == 0) or (((i / 4 & 15) == 0) and ((((i / 4 & 31) == 0) ^ (j / 4 >> 4) & 1) == 0))):
                tex_pixels[i][j] = color1
            else:
                tex_pixels[i][j] = color2

class RandomScene:

    def __init__(self):
        self.models = []
        self.textures = []
        self.shaders = []

    def add_model(self, model):
        self.models.append(model)

    def add_texture(self, tex):
        self.textures.append(tex)

    def add_shader(self, shader):
        self.shaders.append(shader)

    def render(self, view, proj):
        for m in self.models:
            m.render(view, proj)

    def generate_shaders(self):
        self.vertex_shader_src = """
            #version 150
            uniform mat4 matWVP;
            in vec4 Position;
            in vec4 Color;
            in vec2 TexCoord;
            out vec2 oTexCoord;
            out vec4 oColor;
            void main() {
                gl_Position = (matWVP * Position);
                oTexCoord = TexCoord;
                oColor.rgb = pow(Color.rgb, vec3(2.2));
                oColor.a = Color.a;
            }"""
        self.fragment_shader_src = """
            #version 150
            uniform sampler2D Texture0;
            in vec4 oColor;
            in vec2 oTexCoord;
            out vec4 FragColor;
            void main() {
                FragColor = oColor * texture2D(Texture0, oTexCoord);
                //FragColor = oColor;
            }"""

        buffer = TextureBuffer(False, (256, 256), 4, texture_blank(color(255,255,255)))
        shader = ShaderFill(self.vertex_shader_src, self.fragment_shader_src, buffer)
        self.add_shader(shader)

        buffer = TextureBuffer(False, (256, 256), 4, texture_tiles(color(255, 255, 255), color(64, 64, 64), 128, 256, 4))
        shader = ShaderFill(self.vertex_shader_src, self.fragment_shader_src, buffer)
        self.add_shader(shader)

        buffer = TextureBuffer(False, (256, 256), 4, texture_checkerboard(color(255, 255, 255), color(64, 64, 64)))
        shader = ShaderFill(self.vertex_shader_src, self.fragment_shader_src, buffer)
        self.add_shader(shader)

        buffer = TextureBuffer(False, (256, 256), 4, texture_bricks(color(60, 60, 60), color(180, 180, 180)))
        shader = ShaderFill(self.vertex_shader_src, self.fragment_shader_src, buffer)
        self.add_shader(shader)

    def generate_models(self):

        wallSegmentWidth = 1.0
        wallBuffer = 0.5
        wallThickness = 0.2
        wallHeight = 4.0
        ceilingHeight = 4.0

        c = color(64,64,64)

        # Object
        m = Model(np.array([0.0,0.0,0.0]), self.shaders[0])
        m.add_solid_color_box((0.0,0.0,0.0),(1.0,1.0,1.0),c)
        m.allocate_buffers()
        self.add_model(m)

        # Floor
        m = Model(np.array([0.0,0.0,0.0]), self.shaders[1])
        m.add_solid_color_box((-10.,-0.1,-wallSegmentWidth),(10.,0.,20.1), color(128, 128, 128))
        m.add_solid_color_box((-15.,-6.1,18.), (15,-6.,30.), color(128, 128, 128))
        m.allocate_buffers()
        self.add_model(m)

        # Ceiling
        m = Model(np.array([0.0, 0.0 ,0.0]), self.shaders[2])
        m.add_solid_color_box((-10.0, ceilingHeight, -wallSegmentWidth), (10.0, ceilingHeight+0.1, 20.1), color(128, 128, 128))
        m.allocate_buffers()
        self.add_model(m)

        c1 = color(128,128,128)
        blankWall = Model(np.array([0.0,0.0,0.0]), self.shaders[3])
        blankWall.add_solid_color_box((0.0,-0.1,0.0), (wallSegmentWidth, wallHeight, wallThickness), c1)
        doorWall = Model(np.array([0.0,0.0,0.0]), self.shaders[3])
        doorWall.add_solid_color_box((0.0, -0.1, 0.0), (0.1*wallSegmentWidth, wallHeight, wallThickness), c1)
        doorWall.add_solid_color_box((0.1*wallSegmentWidth, 2.0, 0.0), (0.9*wallSegmentWidth, wallHeight, wallThickness), c1)
        doorWall.add_solid_color_box((0.9*wallSegmentWidth,-0.1,0.0),(wallSegmentWidth,wallHeight, wallThickness), c1)
        windowWall = Model(np.array([0.0,0.0,0.0]), self.shaders[3])
        windowWall.add_solid_color_box((0.0,-0.1,0.0),(0.1*wallSegmentWidth,wallHeight,wallThickness),c1)
        windowWall.add_solid_color_box((0.1*wallSegmentWidth,2.0,0.0),(0.9*wallSegmentWidth,wallHeight,wallThickness),c1)
        windowWall.add_solid_color_box((0.1*wallSegmentWidth,-0.1,0.0), (0.9*wallSegmentWidth,0.75,wallThickness), c1)
        windowWall.add_solid_color_box((0.9*wallSegmentWidth,-0.1,0.0),(wallSegmentWidth,wallHeight,wallThickness),c1)

        doorWall.add_solid_color_box((0.1*wallSegmentWidth, -0.1, 0.0), (0.9*wallSegmentWidth, 0.75, wallThickness), c1)
        doorWall.add_solid_color_box((0.9*wallSegmentWidth, -0.1, 0.0), (wallSegmentWidth, wallHeight, wallThickness), c1)

        leftWallDistance = 2
        rightWallDistance = 4
        frontWallDistance = 2
        backWallDistance = 1

        for i in range(-backWallDistance, frontWallDistance+1):
            if i == 0:
                m = doorWall.copy()
            else:
                m = blankWall.copy()

            m.rot = np.array([0.0,0.0,0.0,1.0])
            m.pos = np.array([leftWallDistance*wallSegmentWidth+wallBuffer, 0.0, (i+1)*wallSegmentWidth])
            m.allocate_buffers()
            self.add_model(m)

        for i in range(-backWallDistance, frontWallDistance + 1):
            m = blankWall.copy()
            # This is wrong. Make rotation mat4 or actual quat
            m.rot = np.array([0.0, 0.0, 0.0,-1.0])
            m.pos = np.array([-(rightWallDistance * wallSegmentWidth + wallBuffer), 0.0, i * wallSegmentWidth])
            m.allocate_buffers()
            self.add_model(m)

        for i in range(-rightWallDistance-1,leftWallDistance+1):
            if i == leftWallDistance or i == (-rightWallDistance-1):
                m = blankWall.copy()
            else:
                m = windowWall.copy()
            m.pos = np.array([i*wallSegmentWidth,0.0,frontWallDistance*wallSegmentWidth+wallBuffer])
            m.allocate_buffers()
            self.add_model(m)

        for i in range(-rightWallDistance, leftWallDistance+2):
            m = blankWall.copy()
            m.rot = np.array([0.0,0.0,0.0,0.0])
            m.pos = np.array([i*wallSegmentWidth, 0.0, -backWallDistance*wallSegmentWidth + wallBuffer])
            m.allocate_buffers()
            self.add_model(m)

        table = Model(np.array([0.0,0.0,1.0]), self.shaders[0])
        c1 = color(80, 80, 0)
        table.add_solid_color_box((-1.8,0.8,1.0), (0.0,0.7,0.0), c1)
        table.add_solid_color_box((-1.8, 0.0, 0.0), (-1.7, 0.7, 0.1), c1)
        table.add_solid_color_box((-1.8, 0.7, 1.0), (-1.7, 0.0, 0.9), c1)
        table.add_solid_color_box((0.0, 0.0, 1.0), (-0.1, 0.7, 0.9), c1)
        table.add_solid_color_box((0.0, 0.7, 0.0), (-0.1, 0.0, 0.1), c1)
        m = table.copy()
        m.allocate_buffers()
        self.add_model(m)

        chair = Model(np.array([-1.4,0.0,0.0]), self.shaders[0])
        c1 = color(32,32,80)
        chair.add_solid_color_box((0.0,0.5,0.0), (0.6,0.55,0.6), c1)
        chair.add_solid_color_box((0.0,0.0,0.0), (0.06,1.0,0.06), c1)
        chair.add_solid_color_box((0.0,0.5,0.6), (0.06,0.0,0.54), c1)
        chair.add_solid_color_box((0.6, 0.0, 0.6), (0.54,0.5,0.54), c1)
        chair.add_solid_color_box((0.6, 1.0, 0.0), (0.54, 0.0, 0.06), c1)
        chair.add_solid_color_box((0.01, 0.97, 0.05), (0.59,0.92,0.0), c1)
        m = chair.copy()
        m.allocate_buffers()
        self.add_model(m)

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.generate_shaders()
        self.generate_models()
