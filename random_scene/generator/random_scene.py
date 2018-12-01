from generator.model_factory import *


class RandomScene:

    def __init__(self, seed=None):
        self.models = []
        self.animated_models = []
        self.textures = []
        self.shaders = {}
        self.shader_distribution = []
        self.model_factory = ModelFactory()
        if seed is not None:
            np.random.seed(seed)

        self.left_wall_distance = 2
        self.right_wall_distance = 2
        self.front_wall_distance = 2
        self.back_wall_distance = 2
        self.wall_segment_width = 1.0
        self.wall_buffer = 0.5
        self.wall_thickness = 0.2
        self.wall_height = 4.0
        self.ceiling_height = 4.0
        self.furniture_buffer = 0.0
        self.bounds = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0]], dtype=np.float32)

        self.light_pos_0 = np.array([-1.5, 1.6, 1.0])
        self.light_pos_1 = np.array([1.5, 1.0, -0.5])
        self.light_pow_0 = 0.5
        self.light_pow_1 = 0.25
        self.ambient_light = 0.01
        self.light_wall_buffer = 0.1

        self.clear_color = (0.0,0.0,0.0)

        self.room_type = 0
        self.furniture = {}
        self.door_prob = 0.25
        self.window_prob = 0.5

        self.bed_prob = 0.2
        self.dresser_prob = 0.2
        self.small_table_prob = 0.2
        self.stool_prob = 0.2
        self.couch3_prob = 0.2
        self.couch2_prob = 0.2
        self.big_chair_prob = 0.2
        self.coffee_table_prob = 0.2

    def random_uniform(self, prob):
        return np.random.uniform() < prob

    def add_model(self, model):
        self.models.append(model)

    def add_animated_model(self, model):
        self.animated_models.append(model)

    def add_texture(self, tex):
        self.textures.append(tex)

    def render(self, view, proj):
        for m in self.models:
            m.render(view, proj)

        for m in self.animated_models:
            m.render(view, proj)

    def update(self, delta_time):
        m = self.animated_models[0]
        old_rot = m.rot
        m.rot = np.dot(roty(float(delta_time)/30.0), old_rot)

    def sample_distribution(self, distribution):
        total = sum([s[0] for s in distribution])
        i = total * np.random.uniform()
        for s in distribution:
            if i < s[0]:
                return s[1]
            i -= s[0]
        print("Error with sample distribution! i = " + str(i))
        return distribution[-1][1]

    def generate_room_properties(self):
        self.generate_wall_dimensions()
        if self.random_uniform(0.5):
            self.room_type = 1

        rx = np.random.uniform(float(-self.right_wall_distance)+self.light_wall_buffer, float(self.left_wall_distance)-self.light_wall_buffer)
        ry = np.random.uniform(0.5*self.ceiling_height, self.ceiling_height-self.light_wall_buffer)
        rz = np.random.uniform(0.0, float(self.front_wall_distance)-self.light_wall_buffer)
        self.light_pos_0 = np.array([rx, ry, rz])

        rx = np.random.uniform(float(-self.right_wall_distance)*0.5+self.light_wall_buffer, float(self.left_wall_distance)*0.5-self.light_wall_buffer)
        ry = np.random.uniform(self.light_wall_buffer, self.ceiling_height * 0.5)
        rz = np.random.uniform(0.0, float(-self.back_wall_distance)+self.light_wall_buffer)
        self.light_pos_1 = np.array([rx, ry, rz])
        self.light_pow_0 = np.random.uniform()
        self.light_pow_1 = np.random.uniform(1.0-self.light_pow_0, 1.0) * 0.75

        b = np.random.uniform()
        g = np.random.uniform() * 0.6 * b
        r = np.random.uniform() * 0.5 * g
        sky_light = 0.2989 * r + 0.5870 * g + 0.1140 * b
        self.clear_color = (r, g, b)

        self.ambient_light = (self.light_pow_0 + self.light_pow_1 + sky_light + np.random.uniform()) * 0.05

    def configure_shaders(self):
        for s in self.shaders.values():
            s.add_vec3_uniform("LightPos0", self.light_pos_0)
            s.add_vec3_uniform("LightPos1", self.light_pos_1)
            s.add_float_uniform("LightPower0", self.light_pow_0)
            s.add_float_uniform("LightPower1", self.light_pow_1)
            s.add_float_uniform("AmbientLight", self.ambient_light)

    def generate_shaders(self, shaders=None, shader_dist=None):
        if shaders is not None:
            if shader_dist is None:
                raise Exception("Missing shader distribution")
            self.shaders = shaders.copy()
            self.shader_distribution = shader_dist.copy()
            return

        buffer = TextureBuffer(False, (256, 256), texture_blank(Color(255,255,255,255)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([0.6,0.6,0.6]))
        shader.add_vec3_uniform("Color1", np.array([0.0,0.0,0.0]))
        shader.add_texture_buffer(buffer)
        self.shaders["flat_shader"] = shader
        self.shader_distribution.append((6.0, "flat_shader"))

        buffer = TextureBuffer(False, (256, 256), texture_tiles(Color(255, 255, 255, 255), Color(0, 0, 0, 255), 128, 128, 4), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["tile_shader_1"] = shader
        self.shader_distribution.append((1.5, "tile_shader_1"))

        buffer = TextureBuffer(False, (256, 256), texture_tiles(Color(255, 255, 255, 255), Color(0, 0, 0, 255), 64, 64, 4), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["tile_shader_2"] = shader
        self.shader_distribution.append((1.0, "tile_shader_2"))

        buffer = TextureBuffer(False, (256, 256), texture_tiles(Color(255, 255, 255, 255), Color(0, 0, 0, 255), 64, 64, 8), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["tile_shader_3"] = shader
        self.shader_distribution.append((1.0, "tile_shader_3"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(1,1)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_1"] = shader
        self.shader_distribution.append((1.5, "checker_shader_1"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(2,1)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_2"] = shader
        self.shader_distribution.append((0.5, "checker_shader_2"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(1,2)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_3"] = shader
        self.shader_distribution.append((0.5, "checker_shader_3"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(2,2)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_4"] = shader
        self.shader_distribution.append((0.75, "checker_shader_4"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(2,3)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_5"] = shader
        self.shader_distribution.append((0.5, "checker_shader_5"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(3,2)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_6"] = shader
        self.shader_distribution.append((0.5, "checker_shader_6"))

        buffer = TextureBuffer(False, (256, 256), texture_checkerboard(Color(255, 255, 255, 255), Color(0, 0, 0, 255), sfactor=(3,3)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_vec3_uniform("Color0", np.array([1.0,1.0,1.0]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        shader.add_texture_buffer(buffer)
        self.shaders["checker_shader_7"] = shader
        self.shader_distribution.append((0.5, "checker_shader_7"))

        buffer = TextureBuffer(False, (256, 256), texture_bricks(Color(0, 0, 0, 255), Color(255, 255, 255, 255), sfactor=(2,2)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_texture_buffer(buffer)
        shader.add_vec3_uniform("Color0", np.array([0.8,0.8,0.8]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        self.shaders["brick_shader_1"] = shader
        self.shader_distribution.append((1.5, "brick_shader_1"))

        buffer = TextureBuffer(False, (256, 256), texture_bricks(Color(0, 0, 0, 255), Color(255, 255, 255, 255), sfactor=(2,3)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_texture_buffer(buffer)
        shader.add_vec3_uniform("Color0", np.array([0.8,0.8,0.8]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        self.shaders["brick_shader_2"] = shader
        self.shader_distribution.append((1.0, "brick_shader_2"))

        buffer = TextureBuffer(False, (256, 256), texture_bricks(Color(0, 0, 0, 255), Color(255, 255, 255, 255), sfactor=(3,2)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_texture_buffer(buffer)
        shader.add_vec3_uniform("Color0", np.array([0.8,0.8,0.8]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        self.shaders["brick_shader_3"] = shader
        self.shader_distribution.append((1.0, "brick_shader_3"))

        buffer = TextureBuffer(False, (256, 256), texture_bricks(Color(0, 0, 0, 255), Color(255, 255, 255, 255), sfactor=(3,3)), mip_levels=4)
        shader = ShaderFill(basic_vertex_shader_src, colormod_fragment_shader_src)
        shader.add_texture_buffer(buffer)
        shader.add_vec3_uniform("Color0", np.array([0.8,0.8,0.8]))
        shader.add_vec3_uniform("Color1", np.array([0.2,0.2,0.2]))
        self.shaders["brick_shader_4"] = shader
        self.shader_distribution.append((1.0, "brick_shader_4"))

        self.configure_shaders()

    def generate_wall_dimensions(self):
        self.wall_segment_width = 1.0
        self.wall_buffer = self.wall_segment_width * 0.5
        self.wall_thickness = 0.2
        self.wall_height = np.random.uniform(2.5, 5)
        self.ceiling_height = self.wall_height
        self.furniture_buffer = 0.05

        self.left_wall_distance = np.random.randint(2,5)
        self.right_wall_distance = np.random.randint(2,5)
        self.front_wall_distance = np.random.randint(2,6)
        self.back_wall_distance = np.random.randint(2,4)

        self.calculate_room_bounds()

    def calculate_room_bounds(self):
        self.bounds[0][0] = -float(self.right_wall_distance) - self.wall_buffer + self.furniture_buffer
        self.bounds[0][1] = 0.0
        self.bounds[0][2] = -float(self.back_wall_distance) + self.wall_buffer + self.furniture_buffer
        self.bounds[1][0] = float(self.left_wall_distance) + self.wall_buffer - self.furniture_buffer
        self.bounds[1][1] = self.ceiling_height
        self.bounds[1][2] = float(self.front_wall_distance) + self.wall_buffer - self.furniture_buffer

    def generate_furniture(self, location):
        bedroom_distribution = [
            (6.0, []),
            (0.8, ["beddoubleframe1", "beddouble", "beddoubleframeheadboard1"]),
            (0.8, ["beddoubleframe1", "beddouble", "beddoubleframeheadboard2"]),
            (1.0, ["dresser"]),
            (1.0, ["smalltable"]),
            (1.0, ["stool"])
        ]
        living_room_distribution = [
            (6.0, []),
            (0.5, ["couch3"]),
            (0.5, ["couch2"]),
            (0.5, ["couch3", "pluscoffeetable"]),
            (0.5, ["couch2", "pluscoffeetable"]),
            (1.0, ["bigchair"]),
            (0.3, ["table"]),
            (0.3, ["table", "tablechair1"]),
            (0.3, ["table", "tablechair2"]),
            (0.3, ["table", "tablechair1", "tablechair2"]),
            (1.0, ["smalltable"]),
            (1.0, ["stool"])
        ]
        if location not in self.furniture.keys():
            if self.room_type == 0:  # bedroom
                m = self.sample_distribution(bedroom_distribution)
            elif self.room_type == 1:  # living room
                m = self.sample_distribution(living_room_distribution)
            else:
                raise Exception("Unknown room type")

            if len(m) > 0:
                self.furniture[location] = m

    def construct_furniture(self):
        models = []
        for loc in self.furniture.keys():
            local_models = []
            if len(self.furniture[loc]) > 0:
                position = np.array([loc[0], 0.0, loc[1]], dtype=np.float32)
                a = np.arctan2(loc[0], loc[1])
                b = np.arcsin(np.random.uniform(-0.5, 0.5))
                rot = roty( 90.0 * np.round((a + b) / (np.pi/2.0)) )
                for k in self.furniture[loc]:
                    m = self.model_factory.get_model_by_key(k)
                    m.pos = position
                    m.shader = self.shaders[self.sample_distribution(self.shader_distribution)]
                    m.rot = rot
                    m.add_vec3_uniform("Color0",
                                       np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
                    m.add_vec3_uniform("Color1",
                                       np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
                    local_models.append(m)

                o = np.zeros((4,1), dtype=np.float32)
                for i, m in enumerate(local_models):
                    m.calculate_bounding_box()
                    b = np.zeros((4,4), dtype=np.float32)
                    t = m.get_matrix()
                    b[0] = np.dot(t, np.array([m.bounding_box[0][0],m.bounding_box[0][1],m.bounding_box[0][2], 1.0], dtype=np.float32))
                    b[1] = np.dot(t, np.array([m.bounding_box[1][0],m.bounding_box[0][1],m.bounding_box[0][2], 1.0], dtype=np.float32))
                    b[2] = np.dot(t, np.array([m.bounding_box[0][0],m.bounding_box[0][1],m.bounding_box[1][2], 1.0], dtype=np.float32))
                    b[3] = np.dot(t, np.array([m.bounding_box[1][0],m.bounding_box[0][1],m.bounding_box[1][2], 1.0], dtype=np.float32))
                    for j in range(4):
                        if b[j][0] < self.bounds[0][0]:
                            o[0] = max(o[0], self.bounds[0][0] - b[j][0])
                        if b[j][0] > self.bounds[1][0]:
                            o[0] = min(o[0], self.bounds[1][0] - b[j][0])
                        if b[j][2] < self.bounds[0][2]:
                            o[2] = max(o[2], self.bounds[0][2] - b[j][2])
                        if b[j][2] > self.bounds[1][2]:
                            o[2] = min(o[2], self.bounds[1][2] - b[j][2])

                position[0] += o[0]
                position[1] += o[1]
                position[2] += o[2]
                models.extend(local_models)

        models = self.fix_models_positions(models)
        for m in models:
            m.allocate_buffers()
            self.models.append(m)

    def fix_models_positions(self, models):
        #print("Fixing")
        working_models = [m for m in models]
        #print("Working Models: " + str(len(working_models)))

        np.random.shuffle(models)
        # print("Models: " + str(len(models)))
        for m in models:
            if m not in working_models:
                continue

            delete_models = []
            # print("Fixing model")

            for n in working_models:
                # print("Compare")
                if m is n:
                    # print("Skipping, same object")
                    continue
                if m.pos is n.pos:
                    # print("Skipping, same position")
                    continue

                # Need better bounding box intersection algorithm.
                m.calculate_bounding_box()
                pm = np.zeros((4, 4), dtype=np.float32)
                tm = m.get_matrix()
                pm[0] = np.dot(tm, np.array([m.bounding_box[0][0], m.bounding_box[0][1], m.bounding_box[0][2], 1.0],
                                          dtype=np.float32))
                pm[1] = np.dot(tm, np.array([m.bounding_box[1][0], m.bounding_box[0][1], m.bounding_box[0][2], 1.0],
                                          dtype=np.float32))
                pm[2] = np.dot(tm, np.array([m.bounding_box[0][0], m.bounding_box[0][1], m.bounding_box[1][2], 1.0],
                                          dtype=np.float32))
                pm[3] = np.dot(tm, np.array([m.bounding_box[1][0], m.bounding_box[0][1], m.bounding_box[1][2], 1.0],
                                          dtype=np.float32))
                bm = np.zeros((2, 2), dtype=np.float32)
                bm[0][0] = np.min(np.transpose(pm)[0])
                bm[0][1] = np.min(np.transpose(pm)[2])
                bm[1][0] = np.max(np.transpose(pm)[0])
                bm[1][1] = np.max(np.transpose(pm)[2])

                n.calculate_bounding_box()
                pn = np.zeros((4, 4), dtype=np.float32)
                tn = n.get_matrix()
                pn[0] = np.dot(tn, np.array([n.bounding_box[0][0], n.bounding_box[0][1], n.bounding_box[0][2], 1.0],
                                          dtype=np.float32))
                pn[1] = np.dot(tn, np.array([n.bounding_box[1][0], n.bounding_box[0][1], n.bounding_box[0][2], 1.0],
                                          dtype=np.float32))
                pn[2] = np.dot(tn, np.array([n.bounding_box[0][0], n.bounding_box[0][1], n.bounding_box[1][2], 1.0],
                                          dtype=np.float32))
                pn[3] = np.dot(tn, np.array([n.bounding_box[1][0], n.bounding_box[0][1], n.bounding_box[1][2], 1.0],
                                          dtype=np.float32))
                bn = np.zeros((2, 2), dtype=np.float32)
                bn[0][0] = np.min(np.transpose(pn)[0])
                bn[0][1] = np.min(np.transpose(pn)[2])
                bn[1][0] = np.max(np.transpose(pn)[0])
                bn[1][1] = np.max(np.transpose(pn)[2])

                intersect = True
                if bm[0][0] > bn[1][0] or bn[0][0] > bm[1][0]:
                    intersect = False
                if bm[0][1] > bn[1][1] or bn[0][1] > bm[1][1]:
                    intersect = False

                if intersect:
                    # print("Removing object")
                    delete_models.append(n)
                    for d in working_models:
                        if d is not n and d.pos is n.pos and d not in delete_models:
                            delete_models.append(d)

            # update working models to remove deleted models.
            working_models = [m for m in working_models if m not in delete_models]

        # print("Number of models remaining: " + str(len(working_models)))
        return working_models

    def generate_near_models(self):
        # Box
        p1 = np.array([np.random.uniform(-0.5,0.5), np.random.uniform(0.75,1.75),np.random.uniform(0.7,1.25)], dtype=np.float32)
        m = Model(pos=p1, shader=self.shaders["flat_shader"])
        m.rot = np.dot(roty(np.random.uniform(-45.0, 45.0)), rotx(np.random.uniform(-45.0, 45.0)))
        x1 = (-np.random.uniform(0.15,0.35), -np.random.uniform(0.15,0.35), -np.random.uniform(0.15,0.35))
        x2 = ( np.random.uniform(0.15,0.35), np.random.uniform(0.15,0.35), np.random.uniform(0.15,0.35))
        o = (np.random.uniform(0.0,0.15), np.random.uniform(0.0,0.15), np.random.uniform(0.0,0.15))
        m.add_rounded_box(x1, x2, offset=o)
        m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.add_vec3_uniform("Color1", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.allocate_buffers()
        self.add_animated_model(m)

        while True:
            p2 = np.array([np.random.uniform(-0.5,0.5), np.random.uniform(0.5,1.5),np.random.uniform(0.6,1.25)], dtype=np.float32)
            if magnitude(p2 - p1) > 0.8:
                break
        m = Model(pos=p2, shader=self.shaders[self.sample_distribution(self.shader_distribution)])
        m.rot = np.dot(rotx(np.random.uniform(-180.0, 180.0)), rotz(np.random.uniform(-180.0,180.0)))
        # m.add_solid_color_box((-1.5,-0.25,-0.25),(-1.0,0.25,0.25))
        # m.add_sphere(r=0.25, u=16, v=12, c=np.array([-1.25,0.0,0.0], dtype=np.float32), uv_scale=(4.0,2.0))
        m.add_sphere(r=np.random.uniform(0.15,0.5), u=16, v=12, c=(0.0, 0.0, 0.0), uv_scale=(4.0,2.0))
        m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.add_vec3_uniform("Color1", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.allocate_buffers()
        self.add_animated_model(m)

    def generate_models(self):

        self.generate_near_models()

        # Floor
        m = Model(shader=self.shaders[self.sample_distribution(self.shader_distribution)])
        m.add_solid_color_box((-float(self.right_wall_distance+1)*self.wall_segment_width,-0.1,-float(self.back_wall_distance+1)*self.wall_segment_width),
                              (float(self.left_wall_distance+1)*self.wall_segment_width,0.,float(self.front_wall_distance+1)*self.wall_segment_width))
        # m.add_solid_color_box((-15.,-6.1,18.), (15,-6.,30.), Color(128, 128, 128, 255))
        m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.add_vec3_uniform("Color1", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.allocate_buffers()
        self.add_model(m)

        # Ceiling
        m = Model(shader=self.shaders[self.sample_distribution(self.shader_distribution)])
        m.add_solid_color_box((-float(self.right_wall_distance+1)*self.wall_segment_width, self.ceiling_height, -float(self.back_wall_distance+1)*self.wall_segment_width),
                              (float(self.left_wall_distance+1)*self.wall_segment_width, self.ceiling_height+0.1, float(self.front_wall_distance+1)*self.wall_segment_width))
        m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.add_vec3_uniform("Color1", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
        m.allocate_buffers()
        self.add_model(m)

        wall_color0 = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
        wall_color1 = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
        wall_shader_index = self.sample_distribution(self.shader_distribution)
        blank_wall = Model(shader=self.shaders[wall_shader_index])
        blank_wall.add_solid_color_box((0.0,-0.1,0.0), (self.wall_segment_width, self.wall_height, self.wall_thickness))
        blank_wall.add_vec3_uniform("Color0", wall_color0)
        blank_wall.add_vec3_uniform("Color1", wall_color1)
        door_wall = Model(shader=self.shaders[wall_shader_index])
        door_wall.add_solid_color_box((0.0, -0.1, 0.0), (0.1*self.wall_segment_width, self.wall_height, self.wall_thickness))
        door_wall.add_solid_color_box((0.1*self.wall_segment_width, 2.0, 0.0), (0.9*self.wall_segment_width, self.wall_height, self.wall_thickness))
        door_wall.add_solid_color_box((0.9*self.wall_segment_width,-0.1,0.0),(self.wall_segment_width,self.wall_height, self.wall_thickness))
        door_wall.add_vec3_uniform("Color0", wall_color0)
        door_wall.add_vec3_uniform("Color1", wall_color1)
        window_wall = Model(shader=self.shaders[wall_shader_index])
        window_wall.add_solid_color_box((0.0,-0.1,0.0),(0.1*self.wall_segment_width,self.wall_height,self.wall_thickness))
        window_wall.add_solid_color_box((0.1*self.wall_segment_width,2.0,0.0),(0.9*self.wall_segment_width,self.wall_height,self.wall_thickness))
        window_wall.add_solid_color_box((0.1*self.wall_segment_width,-0.1,0.0), (0.9*self.wall_segment_width,0.75,self.wall_thickness))
        window_wall.add_solid_color_box((0.9*self.wall_segment_width,-0.1,0.0),(self.wall_segment_width,self.wall_height,self.wall_thickness))
        window_wall.add_vec3_uniform("Color0", wall_color0)
        window_wall.add_vec3_uniform("Color1", wall_color1)

        door_wall.add_solid_color_box((0.1*self.wall_segment_width, -0.1, 0.0), (0.9*self.wall_segment_width, -0.05, self.wall_thickness))
        door_wall.add_solid_color_box((0.9*self.wall_segment_width, -0.1, 0.0), (self.wall_segment_width, self.wall_height, self.wall_thickness))

        # Left wall
        for i in range(-self.back_wall_distance, self.front_wall_distance+1):
            has_door = False
            if i == -self.back_wall_distance or i == self.front_wall_distance:
                m = blank_wall.copy()
            elif self.random_uniform(self.door_prob):
                m = door_wall.copy()
                self.door_prob *= 0.25
                has_door = True
                self.furniture[(self.left_wall_distance, i)] = []
                self.furniture[(self.left_wall_distance, i+1)] = []
            else:
                if self.random_uniform(self.window_prob):
                    m = window_wall.copy()
                else:
                    m = blank_wall.copy()

            m.rot = roty(90.0)
            m.pos = np.array([self.left_wall_distance*self.wall_segment_width+self.wall_buffer, 0.0, (i+1)*self.wall_segment_width], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

            if has_door:
                m = self.model_factory.get_door_1(dims=(self.wall_segment_width*0.8,2.0,0.05))
                m.shader = self.shaders[self.sample_distribution(self.shader_distribution)]
                m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
                m.rot = roty(90.0)
                m.pos = np.array(
                    [self.left_wall_distance * self.wall_segment_width + self.wall_buffer +
                     (self.wall_thickness * 0.5), 0.0, (i+0.9) * self.wall_segment_width],
                    dtype=np.float32)
                m.allocate_buffers()
                self.add_model(m)

        # Right wall
        for i in range(-self.back_wall_distance, self.front_wall_distance + 1):
            has_door = False
            if i == -self.back_wall_distance or i == self.front_wall_distance:
                m = blank_wall.copy()
            elif self.random_uniform(self.door_prob):
                m = door_wall.copy()
                self.door_prob *= 0.25
                has_door = True
                self.furniture[(-self.right_wall_distance, i)] = []
                self.furniture[(-self.right_wall_distance, i+1)] = []
            else:
                if self.random_uniform(self.window_prob):
                    m = window_wall.copy()
                else:
                    m = blank_wall.copy()

            m.rot = roty(-90.0)
            m.pos = np.array([-(self.right_wall_distance * self.wall_segment_width + self.wall_buffer), 0.0, i * self.wall_segment_width], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

            if has_door:
                m = self.model_factory.get_door_1(dims=(self.wall_segment_width*0.8,2.0,0.05))
                m.shader = self.shaders[self.sample_distribution(self.shader_distribution)]
                m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
                m.rot = roty(-90.0)
                m.pos = np.array(
                    [-(self.right_wall_distance * self.wall_segment_width + self.wall_buffer) -
                     (self.wall_thickness * 0.5), 0.0, (i+0.1) * self.wall_segment_width],
                    dtype=np.float32)
                m.allocate_buffers()
                self.add_model(m)

        # Front wall
        for i in range(-self.right_wall_distance-1,self.left_wall_distance+1):
            has_door = False
            if i == (-self.right_wall_distance-1) or i == self.left_wall_distance:
                m = blank_wall.copy()
            elif self.random_uniform(self.door_prob):
                m = door_wall.copy()
                self.door_prob *= 0.25
                has_door = True
                self.furniture[(i, self.front_wall_distance)] = []
                self.furniture[(i+1, self.front_wall_distance)] = []
            else:
                if self.random_uniform(self.window_prob):
                    m = window_wall.copy()
                else:
                    m = blank_wall.copy()

            m.pos = np.array([i*self.wall_segment_width,0.0,self.front_wall_distance*self.wall_segment_width+self.wall_buffer], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

            if has_door:
                m = self.model_factory.get_door_1(dims=(self.wall_segment_width*0.8,2.0,0.05))
                m.shader = self.shaders[self.sample_distribution(self.shader_distribution)]
                m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
                m.pos = np.array(
                    [(i+0.1)*self.wall_segment_width, 0.0,
                     self.front_wall_distance*self.wall_segment_width+self.wall_buffer + (self.wall_thickness * 0.5)],
                    dtype=np.float32)
                m.allocate_buffers()
                self.add_model(m)

        # Back wall
        for i in range(-self.right_wall_distance, self.left_wall_distance+2):
            has_door = False
            if i == -self.right_wall_distance or i == self.left_wall_distance+1:
                m = blank_wall.copy()
            elif self.random_uniform(self.door_prob):
                m = door_wall.copy()
                self.door_prob *= 0.25
                has_door = True
                self.furniture[(i, -self.back_wall_distance+1)] = []
                self.furniture[(i-1, -self.back_wall_distance+1)] = []
            else:
                if self.random_uniform(self.window_prob):
                    m = window_wall.copy()
                else:
                    m = blank_wall.copy()

            m.rot = roty(180.0)
            m.pos = np.array([i*self.wall_segment_width, 0.0, -self.back_wall_distance*self.wall_segment_width + self.wall_buffer], dtype=np.float32)
            m.allocate_buffers()
            self.add_model(m)

            if has_door:
                m = self.model_factory.get_door_1(dims=(self.wall_segment_width*0.8,2.0,0.05))
                m.shader = self.shaders[self.sample_distribution(self.shader_distribution)]
                m.add_vec3_uniform("Color0", np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()]))
                m.rot = roty(180.0)
                m.pos = np.array(
                    [(i-0.1)*self.wall_segment_width, 0.0,
                     -self.back_wall_distance*self.wall_segment_width + self.wall_buffer - (self.wall_thickness * 0.5)],
                    dtype=np.float32)
                m.allocate_buffers()
                self.add_model(m)
        #
        # Furniture (indicies s.t. each position only generated once)
        # Front wall furniture
        for i in range(-self.right_wall_distance, self.left_wall_distance):
            location = (i, self.front_wall_distance)
            self.generate_furniture(location)

        # Left wall furniture
        for i in range(-self.back_wall_distance+2, self.front_wall_distance+1):
            location = (self.left_wall_distance, i)
            self.generate_furniture(location)

        # Right wall furniture
        for i in range(-self.back_wall_distance+1, self.front_wall_distance):
            location = (-self.right_wall_distance, i)
            self.generate_furniture(location)

        # Back wall furniture
        for i in range(-self.right_wall_distance+1, self.left_wall_distance+1):
            location = (i, -self.back_wall_distance+1)
            self.generate_furniture(location)

        self.construct_furniture()
