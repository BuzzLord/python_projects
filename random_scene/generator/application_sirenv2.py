
from datetime import datetime as dt
from OpenGL.GLUT import *
from generator.random_scene import *
from PIL import Image
import torch
from dataloader06 import rotate_vector, vector_to_angle
import logging

Cubemap = namedtuple('Cubemap', ['tex', 'depth', 'rotation'])


def get_filename(ver, seed, tag, position, direction, fov=None, frame=None):
    if ver == "3":
        return "ss" + ver + "_{:06d}".format(seed) + \
               "_{:+.3f}_{:+.3f}_{:+.3f}".format(position[0], position[1], position[2]) + \
               "_" + tag + ".png"
    elif ver == "4":
        return "ss" + ver + "_{:06d}".format(seed) + \
               "_{:+.3f}_{:+.3f}_{:+.3f}".format(position[0], position[1], position[2]) + \
               "_{:+.4e}_{:+.4e}".format(direction[0], direction[1]) + \
               "_" + tag + ".png"
    elif ver == "5":
        if fov is None:
            fov = 90.0
        if frame is None:
            frame = 0.0
        return "ss" + ver + "_{:06d}".format(seed) + \
               "_{:.1f}_{:07.3f}".format(fov, frame) + \
               "_{:+.3f}_{:+.3f}_{:+.3f}".format(position[0], position[1], position[2]) + \
               "_{:+.4e}_{:+.4e}".format(direction[0], direction[1]) + \
               "_" + tag + ".png"
    else:
        return "ss" + ver + "_{:06d}".format(seed) + \
               "_{:+.3f}_{:+.3f}_{:+.3f}".format(position[0], position[1], position[2]) + \
               "_" + tag + ".png"


class ApplicationSiren:

    def __init__(self, seed=1234):
        self.window_name = b'Application Siren v2'
        self.window_size = (1024, 1024)
        self.effective_resolution = 1024
        self.camera_yaw = 180.0
        self.camera_pitch = 0.0
        self.camera_position = np.array([0.0, 1.5, 0.0, 1.], dtype='float32')
        self.camera_offset = np.array([0.0, 0.0, 0.0, 0.0], dtype='float32')
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

        self.supersample_tex = None
        self.supersample_depth = None
        self.supersample_shader = None
        self.supersample_fov = 90.0
        self.supersample_proj = perspective(self.supersample_fov, 1.0, 0.1, 200.0)
        self.render_gbuffer = False
        self.gbuffer_size = (2*self.effective_resolution, 2*self.effective_resolution)
        self.gbuffer_texture = None
        self.gbuffer_depth = None
        self.screen_space_quad = None
        self.ssq_view = lookat(np.array([0.0,0.0,0.0], dtype=np.float32),
                               np.array([0.0,0.0,-1.0], dtype=np.float32),
                               np.array([0.0,1.0,0.0], dtype=np.float32))
        self.ssq_proj = ortho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        self.last_time = 0
        self.animate = False
        self.animate_time = 5.0
        self.animate_frametime = 1.0 / 60.0

    def initialize(self, args=None):
        if args is None:
            args = []
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
        print(str(dt.now()) + " Generate model animations")
        self.scene.generate_model_animations(self.animate_time)
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
        self.gbuffer_texture = TextureBuffer(True, self.gbuffer_size, None)
        self.gbuffer_depth = DepthBuffer(self.gbuffer_size)

        self.supersample_tex = TextureBuffer(True, self.window_size, None)
        self.supersample_depth = DepthBuffer(self.window_size)

        self.supersample_shader = ShaderFill(basic_vertex_shader_src, supersample_frag_shader_src)
        self.supersample_shader.add_float_uniform("Resolution", float(self.window_size[0]))
        self.supersample_shader.add_texture_buffer(self.gbuffer_texture)

        self.screen_space_quad = Model(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), shader=self.supersample_shader)
        self.screen_space_quad.add_oriented_quad((0.0,0.0,0.5),(1.0,1.0,0.5))
        self.screen_space_quad.allocate_buffers()

    def __render_gbuffer_view(self, position, position_offset=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                              camera_direction=np.array([0.0, 0.0], dtype=np.float32)):
        roll_pitch_yaw = np.dot(roty(self.camera_yaw + camera_direction[0]), rotx(self.camera_pitch + camera_direction[1]))
        pos_offset = transform(roll_pitch_yaw, position + position_offset)
        final_pos = transform(translate(pos_offset[:3]), self.camera_position)

        self.gbuffer_texture.set_and_clear_render_surface(self.gbuffer_depth)
        final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
        final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
        # shifted_eye_pos = transform(translate(np.array([0.0, 0.0, 0.0], dtype='float32')), self.pos2)

        view = lookat(final_pos, final_pos + final_forward, final_up)
        self.scene.render(view, self.supersample_proj)
        self.gbuffer_texture.unset_render_surface()

        # final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
        # final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
        # view = lookat(final_pos, final_pos + final_forward, final_up)
        # rot_matrix = invert(view[0:3, 0:3]).T

        self.supersample_tex.set_and_clear_render_surface(self.supersample_depth)
        # self.screen_space_quad.shader.add_matrix3_uniform("RotateMatrix", rot_matrix)
        self.screen_space_quad.render(self.ssq_view, self.ssq_proj)
        self.supersample_tex.unset_render_surface()

    def __render_loop(self):
        if self.render_gbuffer:
            self.__render_gbuffer_view(self.camera_offset)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.render_gbuffer:
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
            self.scene.update((current_time/1000.0) % self.animate_time)

        glutPostRedisplay()

        if self.generation_mode:
            print("Saving scene " + str(self.generation_count))
            # positions, rotations = self.__generate_random_positions(1, 1)  # 8
            # positions, rotations = self.__generate_random_positions(1, 4)  # 32
            # positions, rotations = self.__generate_random_positions(2, 2)  # 128
            # positions, rotations = self.__generate_random_positions(2, 4)  # 256
            positions, rotations = self.__generate_random_positions(4, 1)   # 512
            # positions, rotations = self.__generate_random_positions(4, 2)   # 1024
            # positions, rotations = self.__generate_random_positions(4, 3)   # 1536
            # positions, rotations = self.__generate_random_positions(4, 4)   # 2048
            # positions, rotations = self.__generate_random_positions(8, 1)   # 4096
            # positions, rotations = self.__generate_random_positions(8, 2)   # 8k
            # positions, rotations = self.__generate_random_positions(8, 4)   # 16k
            offset = positions[0][0]
            print("Eye offset=" + str(offset))
            for i, (pos,rot) in enumerate(zip(positions, rotations)):
                # print("> " + str(pos))
                if i == 0:
                    tag = "r"
                elif i == 1:
                    tag = "l"
                else:
                    tag = "g"
                self.__save_screenshot(pos, eye_offset=offset, filename_tag=tag, camera_direction=rot)
                # self.__save_offset_screenshots(pos, eye_offset=offset, filename_tag=tag, n_way_offset=1)

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
            self.__save_screenshot(self.camera_offset)

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
                          eye_offset=1.0, filename_tag="x"):

        self.__render_gbuffer_view(camera_position, camera_offset, camera_direction)

        self.supersample_tex.set_render_surface()
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGBA, GL_UNSIGNED_BYTE)
        self.supersample_tex.unset_render_surface()

        image = Image.frombytes("RGBA", self.window_size, data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        version = "5"
        save_path = "screens5\\test"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # Normalize position
        pos = (camera_position[0]/eye_offset, camera_position[1]/eye_offset, camera_position[2]/eye_offset)
        # Flip sign on theta, since handedness changes
        dir = (-camera_direction[0], camera_direction[1])

        filename = get_filename(version, self.seeds[-1], filename_tag, pos, dir)
        file_path = os.path.join(save_path, filename)
        image.save(file_path, 'png')

    def __generate_random_positions(self, num, count=1, test_mode=False, full_360=False, normal_180=True,
                                    naive_angle=False):
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
        elif normal_180:
            positions = np.append(positions, np.array([[eye_offset, 0.0, 0.0, 0.0],
                                                      [eye_offset, 0.0, 0.0, 0.0],
                                                      [eye_offset, 0.0, 0.0, 0.0],
                                                      [eye_offset, 0.0, 0.0, 0.0],
                                                      [-eye_offset, 0.0, 0.0, 0.0],
                                                      [-eye_offset, 0.0, 0.0, 0.0],
                                                      [-eye_offset, 0.0, 0.0, 0.0],
                                                      [-eye_offset, 0.0, 0.0, 0.0]], dtype=np.float32), axis=0)
            rotations = np.append(rotations, np.array([[-self.supersample_fov, 0.0],
                                                      [self.supersample_fov, 0.0],
                                                      [0.0, -self.supersample_fov],
                                                      [0.0, self.supersample_fov],
                                                      [-self.supersample_fov, 0.0],
                                                      [self.supersample_fov, 0.0],
                                                      [0.0, -self.supersample_fov],
                                                      [0.0, self.supersample_fov]], dtype=np.float32), axis=0)

            u_scale = 0.3
            v_scale = 0.2

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

    @staticmethod
    def generate_samples(count=1024, spatial=16, angular=16, project_dim=32, num_samples=16, fov=None):
        device = torch.device("cuda:0")

        torch.cuda.manual_seed_all(123456)
        if fov is None or not isinstance(fov, list):
            fov = [90.0]
        fov_set = torch.stack([torch.tensor(i, device=device) for i in fov], dim=0)

        # Build the look vectors for each FOV, for use during sampling
        dx = torch.linspace(-1 + (1 / project_dim), 1 - (1 / project_dim), project_dim, device=device)
        dx = dx.repeat((project_dim, 1))
        dx = dx.unsqueeze(0).repeat((fov_set.shape[0], 1, 1))
        dy = torch.linspace(-1 + (1 / project_dim), 1 - (1 / project_dim), project_dim, device=device)
        dy = dy.repeat((project_dim, 1)).transpose(dim0=0, dim1=1)
        dy = dy.unsqueeze(0).repeat((fov_set.shape[0], 1, 1))
        dz_distances = 1 / torch.tan(fov_set * 0.5 * 0.017453292519943).reshape((fov_set.shape[0], 1, 1))
        dz = -torch.ones(fov_set.shape[0], project_dim, project_dim, device=device) * dz_distances

        vecs_unnormlized = torch.stack((dx, dy, dz), dim=3)
        look_vecs = vecs_unnormlized / torch.norm(vecs_unnormlized, dim=3).unsqueeze(3)
        look_vecs = look_vecs.reshape((fov_set.shape[0], project_dim*project_dim, 3))

        # Construct initial sample influence tensor, to hold sample vector counts
        total_influence = torch.zeros((spatial, spatial, spatial, angular, angular), device=device)
        sample_details = []

        # Spatial location of each grid entry, for calculating influence
        grid_location_list = []
        dim_expansion = torch.linspace(-1+(1/spatial), 1-(1/spatial), spatial, device=device)
        for z in dim_expansion:
            for y in dim_expansion:
                for x in dim_expansion:
                    grid_location_list.append(torch.tensor([x, y, z], device=device))
        grid_locations = torch.stack(grid_location_list, dim=0)
        import_factor = 1.0
        import_exp = 2.0

        c = 0
        while c < count:
            c += 1
            logging.info("--- Generating sample {} / {} -------------------------".format(c, count))
            fov_select = torch.randint(fov_set.shape[0], (num_samples,), device=device)
            sample_fov = fov_set.index_select(0, fov_select).unsqueeze(0)
            sample_look_vecs = look_vecs.index_select(0, fov_select)
            # Positions are uniform in [-1, 1]
            sample_positions = torch.rand((3, num_samples), device=device) * 2 - 1
            # Rotations are normal distributed, zero mean, 0.5 stdev
            sample_rotations = torch.normal(0.0, 0.5, size=(2, num_samples), device=device)
            sample_params = torch.cat((sample_positions, sample_rotations, sample_fov), dim=0)
            sample_influence = torch.zeros((num_samples, spatial, spatial, spatial, angular, angular), device=device)
            for i in range(num_samples):
                # Rotate each of the 'num_samples' vecs, using sample_params[3:5, i] as theta,phi
                # note: (sample_look_vecs.shape = [num_samples, project_dim**2, 3])
                # rotated view vecs := [project_dim**2, 3]
                rotated_view_vecs = rotate_vector(sample_look_vecs[i,:,:], sample_params[3,i], sample_params[4,i])
                # Select only those vectors that are in the positive half space, z < 0.0
                view_vec_select = (rotated_view_vecs[:,2] < 0.0).to(torch.long).nonzero().reshape(-1)
                # view vecs := [num_pos_vecs, 3]
                view_vecs = rotated_view_vecs.index_select(0, view_vec_select)
                # local grid points := [spatial**3, 3]
                local_grid_points = grid_locations - sample_params[0:3, i]
                # vec dot grid := [num_pos_vecs, spatial**3]
                vec_dot_grid = torch.matmul(view_vecs, local_grid_points.transpose(dim0=0, dim1=1))
                # multi view vecs := [spatial**3, num_pos_vecs, 3]
                multi_view_vecs = view_vecs.unsqueeze(0).repeat((local_grid_points.shape[0],1,1))
                # vec grid points := [spatial**3, num_pos_vecs, 3]
                vec_grid_points = local_grid_points.unsqueeze(1).repeat((1,vec_dot_grid.shape[0],1))
                # perpendicular vecs := [spatial**3, num_pos_vecs, 3]
                perpendicular_vecs = vec_grid_points - vec_dot_grid.transpose(dim0=0, dim1=1).unsqueeze(2) * multi_view_vecs

                # Testing, do a dot between perpendicular and multi-view-vecs
                #test_dot = torch.sum(perpendicular_vecs.reshape(-1,3) * multi_view_vecs.reshape(-1,3), dim=1)

                # distance := [spatial**3, num_pos_vecs]
                distance = torch.norm(perpendicular_vecs, dim=2) * spatial
                # importance := [spatial**3, num_pos_vecs]
                importance = (1 - import_factor * distance.pow(import_exp)).clamp_min(0.0)
                #logging.info("{} : {} = {} => {}".format(c, i, torch.sum(importance.reshape(-1)), sample_params[0:5, i]))

                theta, phi = vector_to_angle(view_vecs)
                theta_index = torch.floor((theta * 0.5 + 0.5) * angular).to(torch.long)
                phi_index = torch.floor((phi * 0.5 + 0.5) * angular).to(torch.long)
                imp_vals = importance.reshape((spatial, spatial, spatial, importance.shape[1]))
                sample_influence[i, :, :, :, theta_index, phi_index] = imp_vals

            # Score each of the samples by comparing it to the current total_influence
            # Only keep the best sample, add it to total
            current_score = torch.sum(torch.log(1 + total_influence).reshape(-1))
            sample_scores = torch.zeros(num_samples)
            for i in range(num_samples):
                current_sample_influence = total_influence + sample_influence[i,:,:,:,:,:]
                current_sample_score = torch.sum(torch.log(1 + current_sample_influence).reshape(-1))
                sample_scores[i] = current_sample_score - current_score

            sample_score_list_strings = ["{:.3f}".format(float(sample_scores[i])) for i in range(num_samples)]
            logging.info("Samples influence improvement: {}".format(", ".join(sample_score_list_strings)))
            best_score_index = torch.argmax(sample_scores)
            logging.info("Keeping sample index {}".format(best_score_index))
            sample_details.append(sample_params[:, best_score_index])
            total_influence = total_influence + sample_influence[best_score_index,:,:,:,:,:]
            num_low_samples = torch.sum((total_influence < 0.25).to(torch.long).reshape(-1))
            num_good_samples = torch.sum((total_influence >= 0.25).to(torch.long).reshape(-1))
            mean_influence = torch.mean(total_influence.reshape(-1))
            max_influence = torch.max(total_influence.reshape(-1))
            logging.info("Num Samples: {} : {}".format(num_low_samples, num_good_samples))
            logging.info("Influence: mean {} , max {}".format(mean_influence, max_influence))

        return torch.stack(sample_details, dim=0).cpu(), total_influence.cpu()

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
            self.render_gbuffer = not self.render_gbuffer
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
    #app = ApplicationSiren(seed=335248)
    #app.initialize()
    #app.start()
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    spatial_dim = 16
    angular_dim = 16
    projected_dim = 32
    sample_count = 16
    total_samples = 1024
    # ApplicationSiren.generate_samples(fov=[60.0, 90.0])
    samples, influence = ApplicationSiren.generate_samples(count=total_samples, spatial=spatial_dim,
                                                           angular=angular_dim, project_dim=projected_dim,
                                                           num_samples=sample_count)

    from torchvision.utils import save_image
    inf_max = torch.max(influence.reshape(-1))
    influence = influence / inf_max
    influence = influence.reshape((spatial_dim, 1, spatial_dim*spatial_dim, angular_dim*angular_dim)).repeat((1,3,1,1))
    num_rows = int(math.sqrt(float(spatial_dim)))
    save_image(influence, "influence.png", nrow=num_rows)

