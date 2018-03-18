from collections import namedtuple
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from gl_math import *

active_texture = { 0: GL_TEXTURE0, 1: GL_TEXTURE1, 2: GL_TEXTURE2, 3: GL_TEXTURE3,
                   4: GL_TEXTURE4, 5: GL_TEXTURE5, 6: GL_TEXTURE6, 7: GL_TEXTURE7 }

default_texnames = { 0: "Texture0", 1: "Texture1", 2: "Texture2", 3: "Texture3",
                     4: "Texture4", 5: "Texture5", 6: "Texture6", 7: "Texture7" }

Color = namedtuple('Color', ['r','g','b','a'])

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
        if self.texId is not None:
            glDeleteTextures(1, self.texId)
            self.texId = None


class TextureBuffer:
    def __init__(self, render_target, size, mip_levels, data):
        self.texId = None
        self.fboId = None
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


class Model:

    def __init__(self, pos, shader):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.rot = np.identity(4, dtype=np.float32)
        self.mat = np.identity(4, dtype=np.float32)
        self.shader = shader
        self.vertex_buffer = None
        self.index_buffer = None
        self.vertices = np.array([], dtype=np.float32)
        self.indices = np.array([], dtype=np.int32)

    def copy(self):
        model_copy = Model(np.copy(self.pos), self.shader)
        model_copy.rot = np.copy(self.rot)
        model_copy.mat = np.copy(self.mat)
        model_copy.vertices = np.copy(self.vertices)
        model_copy.indices = np.copy(self.indices)
        return model_copy

    def get_matrix(self):
        self.mat = np.dot(translate(self.pos[:3]), self.rot)
        return self.mat

    def add_vertex(self, v):
        self.vertices = np.append(self.vertices, v, axis=0)

    def add_index(self, i):
        self.indices = np.append(self.indices, [i], axis=0)

    def allocate_buffers(self):
        #print(str(self.vertices))
        self.vertex_buffer = vbo.VBO(self.vertices, size=len(self.vertices)*4)
        self.index_buffer = vbo.VBO(self.indices, target=GL_ELEMENT_ARRAY_BUFFER, size=len(self.indices)*4)

    def add_solid_color_box(self, p1, p2, c, shading=False):
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

        vert_offset = np.int32(len(self.vertices) / 3)
        for i in indices:
            self.add_index(i + vert_offset)

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
                #nc = (c & 0xff000000) + (np.clip(R, 0, 255) << 16) + (np.clip(G, 0, 255) << 8) + (np.clip(B, 0, 255))
                nc = (np.clip(R, 0, 255) << 16) + (np.clip(G, 0, 255) << 8) + (np.clip(B, 0, 255))
                #print(str("New Col: " + str(nc)))
                col = np.array([nc], dtype='uint32')
            else:
                col = np.array([c], dtype='uint32')

            col.dtype=np.float32
            #print(str("New Col!: " + str(col)))

            #self.add_vertex(np.append(np.append(pos,uv),col))
            self.add_vertex(np.append(pos,uv))

    def add_oriented_quad(self, p1, p2, c):
        verts = np.array([
            [p1[0], p1[1], p1[2], 0.0, 0.0], [p2[0], p1[1], p2[2], 1.0, 0.0],
            [p2[0], p2[1], p2[2], 1.0, 1.0], [p1[0], p2[1], p1[2], 0.0, 1.0]],
            dtype=np.float32)

        indices = np.array([0, 3, 1, 1, 3, 2], dtype=np.int32)

        vert_offset = np.int32(len(self.vertices) / 3)
        for i in indices:
            self.add_index(i + vert_offset)

        for v in verts:
            col = np.array([0x00ffffff], dtype=np.uint32)
            col.dtype=np.float32
            #self.add_vertex(np.append(v[0:5],col))
            self.add_vertex(v[0:5])

    def render(self, view, proj):
        #print("view type: " + str(view.dtype) + ", shape: " + str(view.shape))
        #print("proj type: " + str(proj.dtype) + ", shape: " + str(proj.shape))
        model_matrix = self.get_matrix()
        #print("model_matrix type: " + str(model_matrix.dtype) + ", shape: " + str(model_matrix.shape))
        mvp_matrix = np.dot(proj, np.dot(view, model_matrix))
        #print("mvp_matrix type: " + str(mvp_matrix.dtype) + ", shape: " + str(mvp_matrix.shape))

        glUseProgram(self.shader.program)
        try:
            glUniformMatrix4fv(glGetUniformLocation(self.shader.program, "matWVP"), 1, GL_TRUE, mvp_matrix)

            for name in self.shader.matrix_uniforms.keys():
                glUniformMatrix4fv(glGetUniformLocation(self.shader.program, name), 1, GL_TRUE, self.shader.matrix_uniforms[name])

            for i, texname in enumerate(self.shader.texture.keys()):
                glUniform1i(glGetUniformLocation(self.shader.program, texname), i)
                glActiveTexture(active_texture[i])
                glBindTexture(GL_TEXTURE_2D, self.shader.texture[texname].texId)

            for name in self.shader.float_uniforms.keys():
                glUniform1f(glGetUniformLocation(self.shader.program, name), self.shader.float_uniforms[name])

            pos_loc = glGetAttribLocation(self.shader.program, "Position")
            #col_loc = glGetAttribLocation(self.shader.program, "Color")
            uv_loc = glGetAttribLocation(self.shader.program, "TexCoord")

            try:
                self.vertex_buffer.bind()
                self.index_buffer.bind()

                glEnableVertexAttribArray(pos_loc)
                glEnableVertexAttribArray(uv_loc)
                #glEnableVertexAttribArray(col_loc)

                glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 4 * 5, self.vertex_buffer)
                glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 4 * 5, self.vertex_buffer+12)
                #glVertexAttribPointer(col_loc, 4, GL_UNSIGNED_INT, GL_TRUE, 4 * 6, self.vertex_buffer+20)

                glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

            finally:
                glDisableVertexAttribArray(pos_loc)
                glDisableVertexAttribArray(uv_loc)
                #glDisableVertexAttribArray(col_loc)
                self.index_buffer.unbind()
                self.vertex_buffer.unbind()
        finally:
            glUseProgram(0)


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
            if ((i % tile_width) / thickness == 0) or ((j % tile_length) / thickness == 0):
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
            if (((j >> 2) & 15) == 0) or ((((i >> 2) & 15) == 0) and (((((i >> 2) & 31) == 0) ^ ((j >> 2) >> 4) & 1) == 0)):
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


class RandomScene:

    def __init__(self, seed=None):
        self.models = []
        self.textures = []
        self.shaders = []
        if seed is not None:
            np.random.seed(seed)

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
        vertex_shader_src = """
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
        fragment_shader_src = """
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

        buffer = TextureBuffer(False, (256, 256), 4, texture_blank(Color(255,255,255,255)))
        shader = ShaderFill(vertex_shader_src, fragment_shader_src)
        shader.add_texture_buffer(buffer)
        self.add_shader(shader)

        buffer = TextureBuffer(False, (256, 256), 4, texture_tiles(Color(255, 255, 255, 255), Color(64, 64, 64, 255), 128, 128, 4))
        shader = ShaderFill(vertex_shader_src, fragment_shader_src)
        shader.add_texture_buffer(buffer)
        self.add_shader(shader)

        buffer = TextureBuffer(False, (256, 256), 4, texture_checkerboard(Color(255, 255, 255, 255), Color(64, 64, 64, 255)))
        shader = ShaderFill(vertex_shader_src, fragment_shader_src)
        shader.add_texture_buffer(buffer)
        self.add_shader(shader)

        buffer = TextureBuffer(False, (256, 256), 4, texture_bricks(Color(60, 60, 60, 255), Color(180, 180, 180, 255)))
        shader = ShaderFill(vertex_shader_src, fragment_shader_src)
        shader.add_texture_buffer(buffer)
        self.add_shader(shader)

    def generate_models(self):

        wall_segment_width = 1.0
        wall_buffer = 0.5
        wall_thickness = 0.2
        wall_height = 4.0
        ceiling_height = 4.0

        c = Color(64, 64, 64, 255)

        # Object
        m = Model(np.array([-0.5, 1.0, -5.0], dtype=np.float32), self.shaders[0])
        m.rot = np.dot(roty(30.0), rotx(30.0))
        m.add_solid_color_box((0.0,0.0,0.0),(1.0,1.0,1.0),c)
        m.allocate_buffers()
        self.add_model(m)

        # Fix all these objects to be in the correct orientation
        # Floor
        m = Model(np.array([0.0,0.0,0.0], dtype=np.float32), self.shaders[1])
        m.add_solid_color_box((-10.,-0.1,-wall_segment_width),(10.,0.,20.1), Color(128, 128, 128, 255))
        m.add_solid_color_box((-15.,-6.1,18.), (15,-6.,30.), Color(128, 128, 128, 255))
        m.allocate_buffers()
        self.add_model(m)

        # Ceiling
        m = Model(np.array([0.0, 0.0 ,0.0], dtype=np.float32), self.shaders[2])
        m.add_solid_color_box((-10.0, ceiling_height, -wall_segment_width), (10.0, ceiling_height+0.1, 20.1), Color(128, 128, 128, 255))
        m.allocate_buffers()
        self.add_model(m)

        c1 = Color(128, 128, 128, 255)
        blank_wall = Model(np.array([0.0,0.0,0.0], dtype=np.float32), self.shaders[3])
        blank_wall.add_solid_color_box((0.0,-0.1,0.0), (wall_segment_width, wall_height, wall_thickness), c1)
        door_wall = Model(np.array([0.0,0.0,0.0], dtype=np.float32), self.shaders[3])
        door_wall.add_solid_color_box((0.0, -0.1, 0.0), (0.1*wall_segment_width, wall_height, wall_thickness), c1)
        door_wall.add_solid_color_box((0.1*wall_segment_width, 2.0, 0.0), (0.9*wall_segment_width, wall_height, wall_thickness), c1)
        door_wall.add_solid_color_box((0.9*wall_segment_width,-0.1,0.0),(wall_segment_width,wall_height, wall_thickness), c1)
        window_wall = Model(np.array([0.0,0.0,0.0], dtype=np.float32), self.shaders[3])
        window_wall.add_solid_color_box((0.0,-0.1,0.0),(0.1*wall_segment_width,wall_height,wall_thickness),c1)
        window_wall.add_solid_color_box((0.1*wall_segment_width,2.0,0.0),(0.9*wall_segment_width,wall_height,wall_thickness),c1)
        window_wall.add_solid_color_box((0.1*wall_segment_width,-0.1,0.0), (0.9*wall_segment_width,0.75,wall_thickness), c1)
        window_wall.add_solid_color_box((0.9*wall_segment_width,-0.1,0.0),(wall_segment_width,wall_height,wall_thickness),c1)

        door_wall.add_solid_color_box((0.1*wall_segment_width, -0.1, 0.0), (0.9*wall_segment_width, 0.75, wall_thickness), c1)
        door_wall.add_solid_color_box((0.9*wall_segment_width, -0.1, 0.0), (wall_segment_width, wall_height, wall_thickness), c1)

        left_wall_distance = 2
        right_wall_distance = 4
        front_wall_distance = 2
        back_wall_distance = 1

        for i in range(-back_wall_distance, front_wall_distance+1):
            if i == 0:
                m = door_wall.copy()
            else:
                m = blank_wall.copy()

            m.rot = roty(90.0)
            m.pos = np.array([left_wall_distance*wall_segment_width+wall_buffer, 0.0, (i+1)*wall_segment_width], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

        for i in range(-back_wall_distance, front_wall_distance + 1):
            m = blank_wall.copy()
            m.rot = roty(-90.0)
            m.pos = np.array([-(right_wall_distance * wall_segment_width + wall_buffer), 0.0, i * wall_segment_width], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

        for i in range(-right_wall_distance-1,left_wall_distance+1):
            if i == left_wall_distance or i == (-right_wall_distance-1):
                m = blank_wall.copy()
            else:
                m = window_wall.copy()
            m.pos = np.array([i*wall_segment_width,0.0,front_wall_distance*wall_segment_width+wall_buffer], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

        for i in range(-right_wall_distance, left_wall_distance+2):
            m = blank_wall.copy()
            m.rot = roty(180.0)
            m.pos = np.array([i*wall_segment_width, 0.0, -back_wall_distance*wall_segment_width + wall_buffer], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

        table = Model(np.array([0.0,0.0,1.0], dtype=np.float32), self.shaders[0])
        c1 = Color(80, 80, 0, 255)
        table.add_solid_color_box((-1.8, 0.8, 1.0), (0.0, 0.7, 0.0), c1)
        table.add_solid_color_box((-1.8, 0.0, 0.0), (-1.7, 0.7, 0.1), c1)
        table.add_solid_color_box((-1.8, 0.7, 1.0), (-1.7, 0.0, 0.9), c1)
        table.add_solid_color_box((0.0, 0.0, 1.0), (-0.1, 0.7, 0.9), c1)
        table.add_solid_color_box((0.0, 0.7, 0.0), (-0.1, 0.0, 0.1), c1)
        m = table.copy()
        m.allocate_buffers()
        self.add_model(m)

        chair = Model(np.array([-1.4,0.0,0.0], dtype=np.float32), self.shaders[0])
        c1 = Color(32,32,80,255)
        chair.add_solid_color_box((0.0,0.5,0.0), (0.6,0.55,0.6), c1)
        chair.add_solid_color_box((0.0,0.0,0.0), (0.06,1.0,0.06), c1)
        chair.add_solid_color_box((0.0,0.5,0.6), (0.06,0.0,0.54), c1)
        chair.add_solid_color_box((0.6, 0.0, 0.6), (0.54,0.5,0.54), c1)
        chair.add_solid_color_box((0.6, 1.0, 0.0), (0.54, 0.0, 0.06), c1)
        chair.add_solid_color_box((0.01, 0.97, 0.05), (0.59,0.92,0.0), c1)
        m = chair.copy()
        m.allocate_buffers()
        self.add_model(m)


