
from datetime import datetime as dt
from OpenGL.GLUT import *
from generator.random_scene import *
from PIL import Image

Cubemap = namedtuple('Cubemap', ['tex', 'depth', 'rotation'])


def get_filename(ver, seed, tag, position, direction):
    return "ss" + ver + "_{:06d}".format(seed) + \
           "_{:+.3f}_{:+.3f}_{:+.3f}".format(position[0], position[1], position[2]) + \
           "_{:+.4e}_{:+.4e}".format(direction[0], direction[1]) + \
           "_" + tag + ".png"


class ApplicationSiren:

    def __init__(self, seed=1234):
        self.window_name = b'Application Siren'
        self.window_size = (2048, 2048)
        self.effective_resolution = 2048
        self.camera_yaw = 180.0
        self.camera_pitch = 0.0
        self.camera_position = np.array([0.0, 1.5, 0.0, 1.], dtype='float32')
        self.camera_offset = np.array([0.0, 0.0, 0.0, 0.0], dtype='float32')
        self.cube_clock = 0.0
        self.proj = perspective(60.0, 1.0, 0.1, 200.0)

        self.save_screenshot = False
        self.generation_mode = False
        self.generation_count = 1

        self.seeds = [seed]
        self.scene = RandomScene(seed)
        self.next_scene = None
        self.generate_next_scene = False
        self.return_previous_scene = False
        self.activate_next_scene = False
        self.cubemap = [None,None,None,None,None]
        self.cube_proj = perspective(90.0, 1.0, 0.1, 200.0)
        self.cubemap_size = (self.effective_resolution*2, self.effective_resolution*2)

        self.position_delta = 2.0 / self.window_size[0]
        self.angle_delta = math.degrees(math.atan(2.0 / self.window_size[0]))

        self.warp_tex = None
        self.warp_dep = None
        self.warp_shader = None
        self.screen_space_quad = None
        self.ssq_view = lookat(np.array([0.0,0.0,0.0], dtype=np.float32),
                               np.array([0.0,0.0,-1.0], dtype=np.float32),
                               np.array([0.0,1.0,0.0], dtype=np.float32))
        self.ssq_proj = ortho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        self.render_warp = False
        self.last_time = 0
        self.animate = False

    def initialize(self, args=[]):
        glutInit(args)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(self.window_size[0], self.window_size[1])
        glutCreateWindow(self.window_name)
        glutDisplayFunc(self.__render_loop)
        glutIdleFunc(self.__update_func)
        glutSpecialFunc(self.__special_func)
        glutKeyboardFunc(self.__keyboard_func)

        glEnable(GL_DEPTH_TEST)
        glFrontFace(GL_CW)
        glEnable(GL_CULL_FACE)
        glEnable(GL_FRAMEBUFFER_SRGB)
        self.last_time = glutGet(GLUT_ELAPSED_TIME)

        print(str(dt.now()) + " Generate room props")
        self.scene.generate_room_properties()
        print(str(dt.now()) + " Generate shaders")
        self.scene.generate_shaders()
        print(str(dt.now()) + " Generate models")
        self.scene.generate_models()
        print(str(dt.now()) + " Generate Done")
        glClearColor(self.scene.clear_color[0], self.scene.clear_color[1], self.scene.clear_color[2], 1.0)

        self.__setup_render_textures()

    def __generate_next_scene(self, seed):
        while seed in self.seeds:
            seed = np.random.randint(99,999999)
        self.seeds.append(seed)

        print(str(dt.now()) + " New scene")
        new_scene = RandomScene(seed)
        print(str(dt.now()) + " Generate room props")
        new_scene.generate_room_properties()
        print(str(dt.now()) + " Generate shaders")  # Reuse shaders
        new_scene.generate_shaders(shaders=self.scene.shaders, shader_dist=self.scene.shader_distribution)
        print(str(dt.now()) + " Generate models")
        new_scene.generate_models()
        print(str(dt.now()) + " Generate Done")
        self.next_scene = new_scene

    def __activate_next_scene(self):
        if self.next_scene is None:
            raise Exception("Next scene is None!")
        self.scene = self.next_scene
        self.scene.configure_shaders()
        glClearColor(self.scene.clear_color[0], self.scene.clear_color[1], self.scene.clear_color[2], 1.0)
        self.camera_position[0] = np.random.uniform(-0.25, 0.25)
        self.camera_position[1] = np.random.uniform( 1.0, 2.0)
        self.camera_position[2] = np.random.uniform(-0.25, 0.25)
        self.camera_pitch = np.random.uniform(-10.0, 10.0)
        self.camera_yaw = np.random.uniform(180.0 - 10.0, 180.0 + 10.0)
        self.next_scene = None
        self.activate_next_scene = False

    def __setup_render_textures(self):
        self.cubemap[0] = Cubemap(tex=TextureBuffer(True, self.cubemap_size, None),
                                  depth=DepthBuffer(self.cubemap_size),
                                  rotation=roty(0.0))
        self.cubemap[1] = Cubemap(tex=TextureBuffer(True, self.cubemap_size, None),
                                  depth=DepthBuffer(self.cubemap_size),
                                  rotation=roty(90.0))
        self.cubemap[2] = Cubemap(tex=TextureBuffer(True, self.cubemap_size, None),
                                  depth=DepthBuffer(self.cubemap_size),
                                  rotation=roty(-90.0))
        self.cubemap[3] = Cubemap(tex=TextureBuffer(True, self.cubemap_size, None),
                                  depth=DepthBuffer(self.cubemap_size),
                                  rotation=rotx(90.0))
        self.cubemap[4] = Cubemap(tex=TextureBuffer(True, self.cubemap_size, None),
                                  depth=DepthBuffer(self.cubemap_size),
                                  rotation=rotx(-90.0))

        self.warp_tex = TextureBuffer(True, self.window_size, None)
        self.warp_dep = DepthBuffer(self.warp_tex.get_size())

        self.warp_shader = ShaderFill(basic_vertex_shader_src, warp_frag_shader_src)
        self.warp_shader.add_float_uniform("Resolution", float(self.effective_resolution))
        for cubemap in self.cubemap:
            self.warp_shader.add_texture_buffer(cubemap.tex)
            self.warp_shader.add_texture_buffer(cubemap.depth)

        self.screen_space_quad = Model(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), shader=self.warp_shader)
        self.screen_space_quad.add_oriented_quad((1.0,1.0,0.5),(0.0,0.0,0.5))
        self.screen_space_quad.allocate_buffers()

    def __render_warped_view(self, position, position_offset=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                             camera_direction=np.array([0.0, 0.0], dtype=np.float32)):
        roll_pitch_yaw = np.dot(roty(self.camera_yaw + camera_direction[0]), rotx(self.camera_pitch + camera_direction[1]))
        pos_offset = transform(roll_pitch_yaw, position + position_offset)
        final_pos = transform(translate(pos_offset[:3]), self.camera_position)

        for cubemap in self.cubemap:
            cubemap.tex.set_and_clear_render_surface(cubemap.depth)
            final_roll_pitch_yaw = np.dot(roll_pitch_yaw, cubemap.rotation)
            final_up = transform(final_roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
            final_forward = transform(final_roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
            # shifted_eye_pos = transform(translate(np.array([0.0, 0.0, 0.0], dtype='float32')), self.pos2)

            view = lookat(final_pos, final_pos + final_forward, final_up)
            self.scene.render(view, self.cube_proj)
            cubemap.tex.unset_render_surface()

        # final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
        # final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
        # view = lookat(final_pos, final_pos + final_forward, final_up)
        # rot_matrix = invert(view[0:3, 0:3]).T

        self.warp_tex.set_and_clear_render_surface(self.warp_dep)
        # self.screen_space_quad.shader.add_matrix3_uniform("RotateMatrix", rot_matrix)
        self.screen_space_quad.render(self.ssq_view, self.ssq_proj)
        self.warp_tex.unset_render_surface()

    def __render_loop(self):
        if self.render_warp:
            self.__render_warped_view(self.camera_offset)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.render_warp:
            self.screen_space_quad.render(self.ssq_view, self.ssq_proj)
        else:
            roll_pitch_yaw = np.dot(roty(self.camera_yaw), rotx(self.camera_pitch))
            final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
            final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
            rotated_offset = transform(roll_pitch_yaw, self.camera_offset)
            shifted_eye_pos = transform(translate(rotated_offset[:3]), self.camera_position)

            view = lookat(shifted_eye_pos, shifted_eye_pos + final_forward, final_up)
            self.scene.render(view, self.proj)

        glutSwapBuffers()
        return

    def __update_func(self):
        if self.activate_next_scene:
            self.__activate_next_scene()

        if self.animate and not self.generation_mode:
            current_time = glutGet(GLUT_ELAPSED_TIME)
            delta_time = current_time - self.last_time
            self.last_time = current_time
            self.scene.update(delta_time)

        glutPostRedisplay()

        if self.generation_mode:
            print("Saving scene " + str(self.generation_count))
            save_cubemap = False
            positions, rotations = self.__generate_random_positions(1, 1)  # 8
            # positions, rotations = self.__generate_random_positions(1, 4)  # 32
            # positions, rotations = self.__generate_random_positions(2, 2)  # 128
            # positions, rotations = self.__generate_random_positions(4, 1)   # 512
            # positions, rotations = self.__generate_random_positions(4, 4)   # 2048
            # positions, rotations = self.__generate_random_positions(8, 1)   # 4096
            # positions, rotations = self.__generate_random_positions(8, 4)   # 16k
            offset = positions[0][0]
            print("Eye offset=" + str(offset))
            for i, (pos,rot) in enumerate(zip(positions, rotations)):
                print("> " + str(pos))
                if i == 0:
                    tag = "r"
                elif i == 1:
                    tag = "l"
                else:
                    tag = "g"
                self.__save_screenshot(pos, eye_offset=offset, filename_tag=tag, save_cubemap=save_cubemap,
                                       camera_direction=rot)
                # self.__save_offset_screenshots(pos, eye_offset=offset, filename_tag=tag, n_way_offset=1, save_cubemap=save_cubemap)

            print("Done")
            self.generation_count -= 1
            if self.generation_count > 0:
                # self.generation_mode = False
                self.generate_next_scene = True
            else:
                print("Exiting")
                sys.exit(0)

        if self.save_screenshot:
            self.save_screenshot = False
            self.__save_screenshot(self.camera_offset, save_cubemap=False)

        if self.generate_next_scene:
            if self.return_previous_scene:
                self.return_previous_scene = False
                if len(self.seeds) > 1:
                    seed = self.seeds[-2]
                    self.seeds = self.seeds[:-2]
                else:
                    seed = self.seeds[0]
            else:
                seed = np.random.randint(99,999999)
            self.generate_next_scene = False
            self.__generate_next_scene(seed)
            self.activate_next_scene = True

    def __save_screenshot(self, camera_position, camera_offset=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                          camera_direction=np.array([0.0, 0.0], dtype=np.float32),
                          eye_offset=1.0, filename_tag="x", save_cubemap=False):

        self.__render_warped_view(camera_position, camera_offset, camera_direction)

        self.warp_tex.set_render_surface()
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGBA, GL_UNSIGNED_BYTE)
        self.warp_tex.unset_render_surface()

        image = Image.frombytes("RGBA", self.window_size, data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        version = "4"
        save_path = "screens4_mixed\\2048"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Normalize position
        pos = (camera_position[0]/eye_offset, camera_position[1]/eye_offset, camera_position[2]/eye_offset)
        # Flip sign on theta, since handedness changes
        dir = (-camera_direction[0], camera_direction[1])

        filename = get_filename(version, self.seeds[-1], filename_tag, pos, dir)
        file_path = os.path.join(save_path, filename)
        image.save(file_path, 'png')

        if save_cubemap:
            for i, cubemap in enumerate(self.cubemap):
                cubemap.tex.set_render_surface()
                glViewport(0, 0, self.cubemap_size[0], self.cubemap_size[1])
                glReadBuffer(GL_COLOR_ATTACHMENT0)
                data = glReadPixels(0, 0, self.cubemap_size[0], self.cubemap_size[1], GL_RGBA, GL_UNSIGNED_BYTE)
                cubemap.tex.unset_render_surface()

                image = Image.frombytes("RGBA", self.cubemap_size, data)
                image.thumbnail(self.window_size)
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                cubemap_filename = get_filename(version, self.seeds[-1], "c{}".format(i),
                                                (camera_position[0] / eye_offset,
                                                 camera_position[1] / eye_offset,
                                                 camera_position[2] / eye_offset))
                file_path = os.path.join(save_path, cubemap_filename)
                image.save(file_path, 'png')

    def __generate_random_positions(self, num, count=1, test_mode=False, full_360=False, naive_angle=True):
        """
        Splits each dimension into num*2 sections, then creates count position vectors in each.
        Also appends L,R images.
        :param num: num splits for each x,y,z dimension
        :param count: num images in each region
        :return: num*num*num*8 * count + 2 pos vectors
        """
        eye_offset = 0.025
        # eye_offset = np.random.uniform(0.015, 0.04)
        positions = np.array([[eye_offset,0.0,0.0,0.0],[-eye_offset,0.0,0.0,0.0]], dtype=np.float32)
        rotations = np.array([[0.0,0.0],[0.0,0.0]], dtype=np.float32)
        x_bound = 4.0 * eye_offset
        y_bound = 3.0 * eye_offset
        z_bound = 3.0 * eye_offset
        bounds_power = 1.0

        if full_360:
            u_scale = 0.5
            v_scale = 0.25

            def get_rotation():
                u = np.random.normal(0.0, u_scale)
                v = np.random.normal(0.0, v_scale)
                return np.array([[u * 90.0, v * 90.0]], dtype=np.float32)
        elif naive_angle:
            u_bound = 90.0 / self.window_size[0]
            v_bound = 90.0 / self.window_size[1]

            def get_rotation():
                u = np.random.uniform(-1.0, 1.0)
                v = np.random.uniform(-1.0, 1.0)
                return np.array([[u * u_bound, v * v_bound]], dtype=np.float32)
        else:
            res_angles = {
                2:    (1 / 2,    3.918265e-1),
                4:    (1 / 4,    1.000796e-1),
                8:    (1 / 8,    2.469218e-2),
                16:   (1 / 16,   6.145642e-3),
                32:   (1 / 32,   1.534594e-3),
                64:   (1 / 64,   3.835311e-4),
                128:  (1 / 128,  9.587665e-5),
                256:  (1 / 256,  2.396875e-5),
                512:  (1 / 512,  5.992043e-6),
                1024: (1 / 1024, 1.497961e-6),
                2048: (1 / 2048, 3.744597e-7),
                4096: (1 / 4096, 9.361220e-8)
            }

            u_bound = 90.0 * res_angles[self.window_size[0]][0]
            v_bound = 90.0 * res_angles[self.window_size[1]][1]

            def get_rotation():
                u = np.random.uniform(-1.0, 1.0)
                v = np.random.uniform(-1.0, 1.0) * (1 - abs(u))
                return np.array([[u * u_bound, v * v_bound]], dtype=np.float32)

        if test_mode:
            s = 2.0 * eye_offset
            pos = [(0,0,0),(s,0,0),(-s,0,0),(0,s,0),(0,-s,0),(0,0,s),(0,0,-s)]
            angles = [(0,0),(-1,0),(1,0),(0,1),(0,-1),(0.5,0.5),(-0.5,0.5),(0.5,-0.5),(-0.5,-0.5)]
            for x, y, z in pos:
                for u, v in angles:
                    positions = np.append(positions, np.array([[x, y, z, 0.0]], dtype=np.float32), axis=0)
                    rotations = np.append(rotations, np.array([[u * u_bound, v * v_bound]], dtype=np.float32), axis=0)
            return positions, rotations

        x_range = np.linspace(0.0, 1.0, num=num + 1)
        y_range = np.linspace(0.0, 1.0, num=num + 1)
        z_range = np.linspace(0.0, 1.0, num=num + 1)

        for i in range(num):
            for j in range(num):
                for k in range(num):
                    for c in range(count):
                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[x, y, z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[-x, y, z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[x, -y, z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[-x, -y, z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[x, y, -z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[-x, y, -z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[x, -y, -z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

                        x = np.random.uniform(np.power(x_range[i], bounds_power),
                                              np.power(x_range[i + 1], bounds_power)) * x_bound
                        y = np.random.uniform(np.power(y_range[j], bounds_power),
                                              np.power(y_range[j + 1], bounds_power)) * y_bound
                        z = np.random.uniform(np.power(z_range[k], bounds_power),
                                              np.power(z_range[k + 1], bounds_power)) * z_bound
                        p = np.array([[-x, -y, -z, 0.0]], dtype=np.float32)
                        positions = np.append(positions, p, axis=0)
                        rotations = np.append(rotations, get_rotation(), axis=0)

        return positions, rotations

    def __print_position(self):
        print("({:+0.5f},{:+0.5f},{:+0.5f},{:+0.6f},{:+0.6f}".format(self.camera_position[0] * 4,
                                                                     (self.camera_position[1] - 1.5) * 3,
                                                                     self.camera_position[2] * 3,
                                                                     self.camera_yaw - 180.0,
                                                                     self.camera_pitch))

    def __keyboard_func(self, key, x, y):
        if self.generation_mode:
            if key == b'\x1b':
                self.generation_mode = False
                return

        # print("Keyboard func saw: " + str(key))
        speed = 0.0001
        if key == b'\x1b':
            sys.exit(0)
        elif key == b'\x20':
            self.render_warp = not self.render_warp
        elif key == b'a':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw - 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw - 90.0)), 0.0])
            self.camera_position += speed * dir
            self.__print_position()
        elif key == b'd':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw + 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw + 90.0)), 0.0])
            self.camera_position += speed * dir
            self.__print_position()
        elif key == b'w':
            dir = np.array([-np.sin(np.deg2rad(self.camera_yaw)), 0.0, -np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += speed * dir
            self.__print_position()
        elif key == b's':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw)), 0.0, np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += speed * dir
            self.__print_position()
        elif key == b'e':
            dir = np.array([0.0, 1.0, 0.0, 0.0])
            self.camera_position += speed * dir
            self.__print_position()
        elif key == b'c':
            dir = np.array([0.0, -1.0, 0.0, 0.0])
            self.camera_position += speed * dir
            self.__print_position()
        elif key == b'A':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw - 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw - 90.0)), 0.0])
            self.camera_position += 100.0 * speed * dir
            self.__print_position()
        elif key == b'D':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw + 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw + 90.0)), 0.0])
            self.camera_position += 100.0 * speed * dir
            self.__print_position()
        elif key == b'W':
            dir = np.array([-np.sin(np.deg2rad(self.camera_yaw)), 0.0, -np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += 100.0 * speed * dir
            self.__print_position()
        elif key == b'S':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw)), 0.0, np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += 100.0 * speed * dir
            self.__print_position()
        elif key == b'E':
            dir = np.array([0.0, 1.0, 0.0, 0.0])
            self.camera_position += 100.0 * speed * dir
            self.__print_position()
        elif key == b'C':
            dir = np.array([0.0, -1.0, 0.0, 0.0])
            self.camera_position += 100.0 * speed * dir
            self.__print_position()
        elif key == b'.':
            self.generate_next_scene = True
        elif key == b',':
            self.generate_next_scene = True
            self.return_previous_scene = True
        # elif key == b'z':
        #     self.camera_offset = np.array([-0.03, 0.0, 0.0, 0.0], dtype=np.float32)
        # elif key == b'x':
        #     self.camera_offset = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # elif key == b'c':
        #     self.camera_offset = np.array([0.03, 0.0, 0.0, 0.0], dtype=np.float32)

        elif key == b'p':
            self.save_screenshot = True
        elif key == b'g':
            self.generation_mode = True

    def __special_func(self, key, x, y):
        states = glutGetModifiers()
        if states & GLUT_ACTIVE_SHIFT:
            d = 1.0
        else:
            d = 0.001

        #  print("Special func saw: " + str(key))
        if key == GLUT_KEY_LEFT:
            self.camera_yaw = (self.camera_yaw + d) % 360.0
            self.__print_position()
        elif key == GLUT_KEY_RIGHT:
            self.camera_yaw = (self.camera_yaw - d) % 360.0
            self.__print_position()
        elif key == GLUT_KEY_UP:
            self.camera_pitch = min(self.camera_pitch + d, 90.0)
            self.__print_position()
        elif key == GLUT_KEY_DOWN:
            self.camera_pitch = max(self.camera_pitch - d, -90.0)
            self.__print_position()

    @staticmethod
    def start():
        glutMainLoop()


if __name__ == "__main__":
    app = ApplicationSiren(seed=335248)
    app.initialize()
    app.start()

