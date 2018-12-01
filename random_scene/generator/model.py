from OpenGL.arrays import vbo
from generator.gl_utils import *


class Model:

    def __init__(self, pos=np.array([0.0,0.0,0.0], dtype=np.float32), shader=None):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.rot = np.identity(4, dtype=np.float32)
        self.mat = np.identity(4, dtype=np.float32)
        self.shader = shader
        self.vertex_width = 8
        self.vertex_buffer = None
        self.index_buffer = None
        self.vertices = np.array([], dtype=np.float32)
        self.indices = np.array([], dtype=np.int32)
        self.float_uniforms = {}
        self.vec3_uniforms = {}
        self.matrix_uniforms = {}
        self.bounding_box = None
        self.bounding_box_valid = False

    def copy(self):
        model_copy = Model(pos=np.copy(self.pos), shader=self.shader)
        model_copy.rot = np.copy(self.rot)
        model_copy.mat = np.copy(self.mat)
        model_copy.vertices = np.copy(self.vertices)
        model_copy.indices = np.copy(self.indices)
        model_copy.float_uniforms = self.float_uniforms.copy()
        model_copy.vec3_uniforms = self.vec3_uniforms.copy()
        model_copy.matrix_uniforms = self.matrix_uniforms.copy()
        return model_copy

    def get_matrix(self):
        self.mat = np.dot(translate(self.pos[:3]), self.rot)
        return self.mat

    def add_vertex(self, v):
        self.vertices = np.append(self.vertices, v, axis=0)

    def add_index(self, i):
        self.indices = np.append(self.indices, [i], axis=0)

    def add_offset(self, o):
        for i in range(0, len(self.vertices), self.vertex_width):
            self.vertices[i] += o[0]
            self.vertices[i+1] += o[1]
            self.vertices[i+2] += o[2]

    def calculate_bounding_box(self):
        if self.bounding_box_valid:
            return

        if len(self.vertices) >= 3:
            self.bounding_box = np.array([self.vertices[0:3],self.vertices[0:3]])

        for i in range(0, len(self.vertices), self.vertex_width):
            if self.vertices[i] < self.bounding_box[0][0]:
                self.bounding_box[0][0] = self.vertices[i]
            if self.vertices[i+1] < self.bounding_box[0][1]:
                self.bounding_box[0][1] = self.vertices[i+1]
            if self.vertices[i+2] < self.bounding_box[0][2]:
                self.bounding_box[0][2] = self.vertices[i+2]

            if self.vertices[i] > self.bounding_box[1][0]:
                self.bounding_box[1][0] = self.vertices[i]
            if self.vertices[i+1] > self.bounding_box[1][1]:
                self.bounding_box[1][1] = self.vertices[i+1]
            if self.vertices[i+2] > self.bounding_box[1][2]:
                self.bounding_box[1][2] = self.vertices[i+2]

        self.bounding_box_valid = True

    def allocate_buffers(self):
        # print(str(self.vertices))
        # print(str(self.vertices.shape))
        self.calculate_bounding_box()

        self.vertex_buffer = vbo.VBO(self.vertices, size=len(self.vertices)*4)
        self.index_buffer = vbo.VBO(self.indices, target=GL_ELEMENT_ARRAY_BUFFER, size=len(self.indices)*4)

    def add_solid_color_box(self, p1, p2):
        self.bounding_box_valid = False
        verts = np.array([
            # Top
            [p1[0],p2[1],p1[2],p1[2],p1[0],0.0,1.0,0.0],[p2[0],p2[1],p1[2],p1[2],p2[0],0.0,1.0,0.0],
            [p2[0],p2[1],p2[2],p2[2],p2[0],0.0,1.0,0.0],[p1[0],p2[1],p2[2],p2[2],p1[0],0.0,1.0,0.0],
            # Bottom
            [p1[0],p1[1],p1[2],p1[2],p1[0],0.0,-1.0,0.0],[p2[0],p1[1],p1[2],p1[2],p2[0],0.0,-1.0,0.0],
            [p2[0],p1[1],p2[2],p2[2],p2[0],0.0,-1.0,0.0],[p1[0],p1[1],p2[2],p2[2],p1[0],0.0,-1.0,0.0],
            # Left
            [p1[0],p1[1],p2[2],p2[2],p1[1],-1.0,0.0,0.0],[p1[0],p1[1],p1[2],p1[2],p1[1],-1.0,0.0,0.0],
            [p1[0],p2[1],p1[2],p1[2],p2[1],-1.0,0.0,0.0],[p1[0],p2[1],p2[2],p2[2],p2[1],-1.0,0.0,0.0],
            # Right
            [p2[0],p1[1],p2[2],p2[2],p1[1],1.0,0.0,0.0],[p2[0],p1[1],p1[2],p1[2],p1[1],1.0,0.0,0.0],
            [p2[0],p2[1],p1[2],p1[2],p2[1],1.0,0.0,0.0],[p2[0],p2[1],p2[2],p2[2],p2[1],1.0,0.0,0.0],
            # Back
            [p1[0],p1[1],p1[2],p1[0],p1[1],0.0,0.0,-1.0],[p2[0],p1[1],p1[2],p2[0],p1[1],0.0,0.0,-1.0],
            [p2[0],p2[1],p1[2],p2[0],p2[1],0.0,0.0,-1.0],[p1[0],p2[1],p1[2],p1[0],p2[1],0.0,0.0,-1.0],
            # Front
            [p1[0],p1[1],p2[2],p1[0],p1[1],0.0,0.0,1.0],[p2[0],p1[1],p2[2],p2[0],p1[1],0.0,0.0,1.0],
            [p2[0],p2[1],p2[2],p2[0],p2[1],0.0,0.0,1.0],[p1[0],p2[1],p2[2],p1[0],p2[1],0.0,0.0,1.0]],
            dtype=np.float32)

        indices = np.array([0, 1, 3, 3, 1, 2,
                            5, 4, 6, 6, 4, 7,
                            8, 9, 11, 11, 9, 10,
                            13, 12, 14, 14, 12, 15,
                            16, 17, 19, 19, 17, 18,
                            21, 20, 22, 22, 20, 23], dtype=np.int32)

        vert_offset = np.int32(len(self.vertices) / self.vertex_width)
        for i in indices:
            self.add_index(i + vert_offset)

        for v in verts:
            pos = v[0:3]
            uv = v[3:5]
            nrm = v[5:8]
            self.add_vertex(np.append(np.append(pos,uv),nrm))

    def add_rounded_box(self, p1, p2, offset=(0.0,0.0,0.0)):
        self.bounding_box_valid = False
        verts = np.array([
            # Top 0-3
            [p1[0]+offset[0],p2[1],p1[2]+offset[2],p1[2],p1[0],0.0,1.0,0.0],
            [p2[0]-offset[0],p2[1],p1[2]+offset[2],p1[2],p2[0],0.0,1.0,0.0],
            [p2[0]-offset[0],p2[1],p2[2]-offset[2],p2[2],p2[0],0.0,1.0,0.0],
            [p1[0]+offset[0],p2[1],p2[2]-offset[2],p2[2],p1[0],0.0,1.0,0.0],
            # Bottom 4-7
            [p1[0]+offset[0],p1[1],p1[2]+offset[2],p1[2],p1[0],0.0,-1.0,0.0],
            [p2[0]-offset[0],p1[1],p1[2]+offset[2],p1[2],p2[0],0.0,-1.0,0.0],
            [p2[0]-offset[0],p1[1],p2[2]-offset[2],p2[2],p2[0],0.0,-1.0,0.0],
            [p1[0]+offset[0],p1[1],p2[2]-offset[2],p2[2],p1[0],0.0,-1.0,0.0],
            # Left 8-11
            [p1[0],p1[1]+offset[1],p2[2]-offset[2],p2[2],p1[1],-1.0,0.0,0.0],
            [p1[0],p1[1]+offset[1],p1[2]+offset[2],p1[2],p1[1],-1.0,0.0,0.0],
            [p1[0],p2[1]-offset[1],p1[2]+offset[2],p1[2],p2[1],-1.0,0.0,0.0],
            [p1[0],p2[1]-offset[1],p2[2]-offset[2],p2[2],p2[1],-1.0,0.0,0.0],
            # Right 12-15
            [p2[0],p1[1]+offset[1],p2[2]-offset[2],p2[2],p1[1],1.0,0.0,0.0],
            [p2[0],p1[1]+offset[1],p1[2]+offset[2],p1[2],p1[1],1.0,0.0,0.0],
            [p2[0],p2[1]-offset[1],p1[2]+offset[2],p1[2],p2[1],1.0,0.0,0.0],
            [p2[0],p2[1]-offset[1],p2[2]-offset[2],p2[2],p2[1],1.0,0.0,0.0],
            # Back 16-19
            [p1[0]+offset[0],p1[1]+offset[1],p1[2],p1[0],p1[1],0.0,0.0,-1.0],
            [p2[0]-offset[0],p1[1]+offset[1],p1[2],p2[0],p1[1],0.0,0.0,-1.0],
            [p2[0]-offset[0],p2[1]-offset[1],p1[2],p2[0],p2[1],0.0,0.0,-1.0],
            [p1[0]+offset[0],p2[1]-offset[1],p1[2],p1[0],p2[1],0.0,0.0,-1.0],
            # Front 20-23
            [p1[0]+offset[0],p1[1]+offset[1],p2[2],p1[0],p1[1],0.0,0.0,1.0],
            [p2[0]-offset[0],p1[1]+offset[1],p2[2],p2[0],p1[1],0.0,0.0,1.0],
            [p2[0]-offset[0],p2[1]-offset[1],p2[2],p2[0],p2[1],0.0,0.0,1.0],
            [p1[0]+offset[0],p2[1]-offset[1],p2[2],p1[0],p2[1],0.0,0.0,1.0]],
            dtype=np.float32)

        indices = np.array([
            # Main faces
            0, 1, 3, 3, 1, 2,        # top
            5, 4, 6, 6, 4, 7,        # bottom
            8, 9, 11, 11, 9, 10,     # left
            13, 12, 14, 14, 12, 15,  # right
            16, 17, 19, 19, 17, 18,  # back
            21, 20, 22, 22, 20, 23,  # front
            # Edge and corners
            3, 23, 11,               # top right front corner
            3, 11, 10, 10, 0, 3,     # top left edge
            0, 10, 19,               # top left back corner
            0, 19, 18, 18, 1, 0,     # top back edge
            1, 18, 14,               # top right back corner
            1, 14, 15, 15, 2, 1,     # top right edge
            2, 15, 22,               # top right front corner
            12, 21, 15, 15, 21, 22,  # front right edge
            2, 22, 23, 23, 3, 2,     # top front edge
            20, 8, 23, 23, 8, 11,    # front left edge
            7, 20, 6, 6, 20, 21,     # bottom front edge
            6, 21, 12,               # bottom front right corner
            6, 12, 5, 5, 12, 13,     # bottom right edge
            17, 13, 18, 18, 13, 14,  # back right edge
            5, 13, 17,               # bottom back right corner
            5, 17, 4, 4, 17, 16,     # bottom back edge
            4, 16, 9,                # bottom back left corner
            9, 16, 10, 10, 16, 19,   # back left edge
            4, 9, 7, 7, 9, 8,        # bottom left edge
            7, 8, 20                 # bottom left front corner
            ], dtype=np.int32)

        self.recalculate_normals(verts, indices)

        vert_offset = np.int32(len(self.vertices) / self.vertex_width)
        for i in indices:
            self.add_index(i + vert_offset)

        for v in verts:
            pos = v[0:3]
            uv = v[3:5]
            nrm = v[5:8]
            self.add_vertex(np.append(np.append(pos,uv),nrm))

    def add_sphere(self, r=1.0, c=(0.0,0.0,0.0), u=8, v=6, uv_scale=(1.0,1.0)):
        self.bounding_box_valid = False
        verts = np.array([[]], dtype=np.float32)
        axis = 1
        # Construct vertex positions
        for j in range(v+1):
            y = math.cos((j / v) * math.pi)
            s = math.sin((j / v) * math.pi)
            for i in range(u+1):
                x = s * math.cos((i / u) * 2.0*math.pi)
                z = s * math.sin((i / u) * 2.0*math.pi)
                new_verts = np.array([[c[0]+r*x, c[1]+r*y, c[2]+r*z, uv_scale[0]*(i/u), uv_scale[1]*(j/v), x, y, z]], dtype=np.float32)
                verts = np.append(verts, new_verts, axis=axis)
                axis = 0

        indices = np.array([], dtype=np.int32)
        for j in range(v):
                for i in range(u):
                    i0 = (j)*(u+1)+((i)%(u+1))
                    i1 = (j)*(u+1)+((i+1)%(u+1))
                    i2 = (j+1)*(u+1)+((i)%(u+1))
                    i3 = (j+1)*(u+1)+((i+1)%(u+1))
                    new_inds = np.array([i0, i2, i1, i2, i3, i1], dtype=np.int32)
                    indices = np.append(indices, new_inds)

        vert_offset = np.int32(len(self.vertices) / self.vertex_width)
        for i in indices:
            self.add_index(i + vert_offset)

        for v in verts:
            pos = v[0:3]
            uv = v[3:5]
            nrm = v[5:8]
            self.add_vertex(np.append(np.append(pos, uv), nrm))

    def add_oriented_quad(self, p1, p2):
        self.bounding_box_valid = False
        a = np.array([0.0, p1[1]-p2[1], 0.0])
        b = np.array([p2[0]-p1[0],p1[1]-p2[1],p2[2]-p1[2]])
        n = normalize(np.cross(a,b))
        verts = np.array([
            [p1[0], p1[1], p1[2], 0.0, 0.0, n[0],n[1],n[2]], [p2[0], p1[1], p2[2], 1.0, 0.0, n[0],n[1],n[2]],
            [p2[0], p2[1], p2[2], 1.0, 1.0, n[0],n[1],n[2]], [p1[0], p2[1], p1[2], 0.0, 1.0, n[0],n[1],n[2]]],
            dtype=np.float32)

        indices = np.array([0, 3, 1, 1, 3, 2], dtype=np.int32)

        vert_offset = np.int32(len(self.vertices) / self.vertex_width)
        for i in indices:
            self.add_index(i + vert_offset)

        for v in verts:
            self.add_vertex(v[0:8])

    @staticmethod
    def recalculate_normals(verts, indices):
        # print("Vertex Count: " + str(verts.shape[0]))
        for i in range(verts.shape[0]):
            # print(">>> " + str(verts[i]))
            normals = []
            for j in range(0, indices.shape[0], 3):
                # print(str(indices[j]) + "," + str(indices[j + 1]) + "," + str(indices[j + 2]))
                if i in (indices[j], indices[j+1], indices[j+2]):
                    if i == indices[j]:
                        ai = indices[j+2]
                        bi = indices[j+1]
                        oi = indices[j]
                    elif i == indices[j+1]:
                        ai = indices[j]
                        bi = indices[j+2]
                        oi = indices[j+1]
                    elif i == indices[j+2]:
                        ai = indices[j+1]
                        bi = indices[j]
                        oi = indices[j+2]
                    else:
                        raise Exception("Couldn't find index")
                    a = normalize(verts[ai][0:3] - verts[oi][0:3])
                    b = normalize(verts[bi][0:3] - verts[oi][0:3])
                    n = normalize(np.cross(a, b))
                    w = np.arccos(np.dot(a,b))
                    normals.append((w,n))

            new_normal = np.zeros((1,3), dtype=np.float32)
            total_weight = sum([n[0] for n in normals])
            for n in normals:
                new_normal += (n[0]/total_weight) * n[1]
            new_normal = normalize(new_normal)
            verts[i][5:8] = new_normal
            # print("X>> " + str(verts[i]))

    def add_float_uniform(self, name, value):
        self.float_uniforms[name] = value

    def add_vec3_uniform(self, name, values):
        self.vec3_uniforms[name] = values

    def add_matrix_uniform(self, name, matrix):
        self.matrix_uniforms[name] = matrix

    def render(self, view, proj):
        # print("view type: " + str(view.dtype) + ", shape: " + str(view.shape))
        # print("proj type: " + str(proj.dtype) + ", shape: " + str(proj.shape))
        model_matrix = self.get_matrix()
        # print("model_matrix type: " + str(model_matrix.dtype) + ", shape: " + str(model_matrix.shape))
        # mvp_matrix = np.dot(proj, np.dot(view, model_matrix))
        mv_matrix = np.dot(view, model_matrix)
        norm_matrix = invert(mv_matrix[0:3,0:3]).T
        # print("mvp_matrix type: " + str(mvp_matrix.dtype) + ", shape: " + str(mvp_matrix.shape))

        glUseProgram(self.shader.program)
        try:
            glUniformMatrix4fv(self.shader.get_uniform_location("Model"), 1, GL_TRUE, model_matrix)
            glUniformMatrix4fv(self.shader.get_uniform_location("View"), 1, GL_TRUE, view)
            glUniformMatrix4fv(self.shader.get_uniform_location("Proj"), 1, GL_TRUE, proj)
            glUniformMatrix3fv(self.shader.get_uniform_location("NormalMatrix"), 1, GL_TRUE, norm_matrix)

            for name in self.shader.matrix_uniforms.keys():
                if name in self.matrix_uniforms.keys():
                    glUniformMatrix4fv(self.shader.get_uniform_location(name), 1, GL_TRUE, self.matrix_uniforms[name])
                else:
                    glUniformMatrix4fv(self.shader.get_uniform_location(name), 1, GL_TRUE, self.shader.matrix_uniforms[name])

            for i, texname in enumerate(self.shader.texture.keys()):
                glUniform1i(self.shader.get_uniform_location(texname), i)
                glActiveTexture(active_texture[i])
                glBindTexture(GL_TEXTURE_2D, self.shader.texture[texname].texId)

            for name in self.shader.float_uniforms.keys():
                if name in self.float_uniforms.keys():
                    glUniform1f(self.shader.get_uniform_location(name), self.float_uniforms[name])
                else:
                    glUniform1f(self.shader.get_uniform_location(name), self.shader.float_uniforms[name])

            for name in self.shader.vec3_uniforms.keys():
                if name in self.vec3_uniforms.keys():
                    glUniform3fv(self.shader.get_uniform_location(name), 1, self.vec3_uniforms[name])
                else:
                    glUniform3fv(self.shader.get_uniform_location(name), 1, self.shader.vec3_uniforms[name])

            pos_loc = glGetAttribLocation(self.shader.program, "Position")
            uv_loc = glGetAttribLocation(self.shader.program, "TexCoord")
            nrm_loc = glGetAttribLocation(self.shader.program, "Normal")
            # col_loc = glGetAttribLocation(self.shader.program, "Color")

            try:
                self.vertex_buffer.bind()
                self.index_buffer.bind()

                glEnableVertexAttribArray(pos_loc)
                if uv_loc > 0: glEnableVertexAttribArray(uv_loc)
                if nrm_loc > 0: glEnableVertexAttribArray(nrm_loc)
                # glEnableVertexAttribArray(col_loc)

                glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 4 * self.vertex_width, self.vertex_buffer)
                if uv_loc > 0: glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, 4 * self.vertex_width, self.vertex_buffer+12)
                if nrm_loc > 0: glVertexAttribPointer(nrm_loc, 3, GL_FLOAT, GL_TRUE, 4 * self.vertex_width, self.vertex_buffer+20)
                # glVertexAttribPointer(col_loc, 4, GL_UNSIGNED_INT, GL_TRUE, 4 * self.vertex_width, self.vertex_buffer+20)

                glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

            finally:
                glDisableVertexAttribArray(pos_loc)
                if uv_loc > 0: glDisableVertexAttribArray(uv_loc)
                if nrm_loc > 0: glDisableVertexAttribArray(nrm_loc)
                # if col_loc > 0: glDisableVertexAttribArray(col_loc)
                self.index_buffer.unbind()
                self.vertex_buffer.unbind()
        finally:
            glUseProgram(0)

