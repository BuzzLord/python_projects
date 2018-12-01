
from datetime import datetime as dt
from OpenGL.GLUT import *
from generator.random_scene import *
from PIL import Image

Cubemap = namedtuple('Cubemap', ['tex', 'depth', 'rotation'])


class Application:

    def __init__(self, seed=1234):
        self.window_name = b'Application'
        self.window_size = (256, 256)
        self.camera_yaw = 180.0
        self.camera_pitch = 0.0
        self.camera_position = np.array([0.0, 1.5, 0.0, 1.], dtype='float32')
        self.camera_offset = np.array([0.0, 0.0, 0.0, 0.0], dtype='float32')
        self.cube_clock = 0.0
        self.proj = perspective(60.0, 1.0, 0.1, 200.0)

        self.save_screenshot = False
        self.generation_mode = False
        self.generation_count = 200

        self.seeds = [seed]
        self.scene = RandomScene(seed)
        self.next_scene = None
        self.generate_next_scene = False
        self.return_previous_scene = False
        self.activate_next_scene = False
        self.cubemap = [None,None,None,None,None]
        self.cube_proj = perspective(90.0, 1.0, 0.1, 200.0)

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
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
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
        cubemap_size = (1024, 1024)
        self.cubemap[0] = Cubemap(tex=TextureBuffer(True, cubemap_size, None),
                                  depth=DepthBuffer(cubemap_size),
                                  rotation=roty(0.0))
        self.cubemap[1] = Cubemap(tex=TextureBuffer(True, cubemap_size, None),
                                  depth=DepthBuffer(cubemap_size),
                                  rotation=roty(90.0))
        self.cubemap[2] = Cubemap(tex=TextureBuffer(True, cubemap_size, None),
                                  depth=DepthBuffer(cubemap_size),
                                  rotation=roty(-90.0))
        self.cubemap[3] = Cubemap(tex=TextureBuffer(True, cubemap_size, None),
                                  depth=DepthBuffer(cubemap_size),
                                  rotation=rotx(90.0))
        self.cubemap[4] = Cubemap(tex=TextureBuffer(True, cubemap_size, None),
                                  depth=DepthBuffer(cubemap_size),
                                  rotation=rotx(-90.0))

        self.warp_tex = TextureBuffer(True, (256, 256), None)
        self.warp_dep = DepthBuffer(self.warp_tex.get_size())

        self.warp_shader = ShaderFill(basic_vertex_shader_src, warp_frag_shader_src)
        for cubemap in self.cubemap:
            self.warp_shader.add_texture_buffer(cubemap.tex)

        self.screen_space_quad = Model(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), shader=self.warp_shader)
        self.screen_space_quad.add_oriented_quad((1.0,1.0,0.5),(0.0,0.0,0.5))
        self.screen_space_quad.allocate_buffers()

    def __render_warped_view(self, position):
        roll_pitch_yaw = np.dot(roty(self.camera_yaw), rotx(self.camera_pitch))
        pos_offset = transform(roll_pitch_yaw, position)
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

        self.warp_tex.set_and_clear_render_surface(self.warp_dep)
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
            positions = self.__generate_random_positions(2)
            offset = positions[0][0]
            print("Eye offset=" + str(offset))
            for i, pos in enumerate(positions):
                print("> " + str(pos))
                if i == 0:
                    tag = "r"
                elif i == 1:
                    tag = "l"
                else:
                    tag = "g"
                self.__save_screenshot(pos, eye_offset=offset, filename_tag=tag)
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

    def __keyboard_func(self, key, x, y):
        if self.generation_mode:
            if key == b'\x1b':
                self.generation_mode = False
                return

        # print("Keyboard func saw: " + str(key))
        speed = 0.05
        if key == b'\x1b':
            sys.exit(0)
        elif key == b'\x20':
            self.render_warp = not self.render_warp
        elif key == b'a':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw - 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw - 90.0)), 0.0])
            self.camera_position += speed * dir
        elif key == b'd':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw + 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw + 90.0)), 0.0])
            self.camera_position += speed * dir
        elif key == b'w':
            dir = np.array([-np.sin(np.deg2rad(self.camera_yaw)), 0.0, -np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += speed * dir
        elif key == b's':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw)), 0.0, np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += speed * dir
        elif key == b'A':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw - 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw - 90.0)), 0.0])
            self.camera_position += 4.0 * speed * dir
        elif key == b'D':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw + 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw + 90.0)), 0.0])
            self.camera_position += 4.0 * speed * dir
        elif key == b'W':
            dir = np.array([-np.sin(np.deg2rad(self.camera_yaw)), 0.0, -np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += 4.0 * speed * dir
        elif key == b'S':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw)), 0.0, np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += 4.0 * speed * dir
        elif key == b'.':
            self.generate_next_scene = True
        elif key == b',':
            self.generate_next_scene = True
            self.return_previous_scene = True
        elif key == b'z':
            self.camera_offset = np.array([-0.03, 0.0, 0.0, 0.0], dtype=np.float32)
        elif key == b'x':
            self.camera_offset = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif key == b'c':
            self.camera_offset = np.array([0.03, 0.0, 0.0, 0.0], dtype=np.float32)

        elif key == b'p':
            self.save_screenshot = True
        elif key == b'g':
            self.generation_mode = True

    def __special_func(self, key, x, y):
        states = glutGetModifiers()
        if states & GLUT_ACTIVE_SHIFT:
            d = 5.0
        else:
            d = 1.0
        
        #  print("Special func saw: " + str(key))
        if key == GLUT_KEY_LEFT:
            self.camera_yaw = (self.camera_yaw + d) % 360.0
        elif key == GLUT_KEY_RIGHT:
            self.camera_yaw = (self.camera_yaw - d) % 360.0
        elif key == GLUT_KEY_UP:
            self.camera_pitch = min(self.camera_pitch + d, 90.0)
        elif key == GLUT_KEY_DOWN:
            self.camera_pitch = max(self.camera_pitch - d, -90.0)

    def __save_screenshot(self, camera_offset, eye_offset=1.0, filename_tag="x"):
        self.__render_warped_view(camera_offset)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.screen_space_quad.render(self.ssq_view, self.ssq_proj)

        glReadBuffer(GL_BACK)
        data = glReadPixels(0, 0, self.window_size[0], self.window_size[1], GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", self.window_size, data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        version = "1"

        save_path = "test_256"
        filename = "ss" + version + "_{:06d}".format(self.seeds[-1]) + filename_tag + "_{:.2f}_{:.2f}_{:.2f}.png".format(camera_offset[0]/eye_offset,camera_offset[1]/eye_offset,camera_offset[2]/eye_offset)
        file_path = os.path.join(save_path, filename)
        image.save(file_path, 'png')

    @staticmethod
    def __generate_random_positions(num):
        eye_offset = np.random.uniform(0.015, 0.04)
        positions = np.array([[eye_offset,0.0,0.0,0.0],[-eye_offset,0.0,0.0,0.0]], dtype=np.float32)
        x_bound = 16.0 * eye_offset
        y_bound = 12.0 * eye_offset
        z_bound = 12.0 * eye_offset
        bounds_power = 2.0

        x_range = np.linspace(0.0, 1.0, num=num + 1)
        y_range = np.linspace(0.0, 1.0, num=num + 1)
        z_range = np.linspace(0.0, 1.0, num=num + 1)

        for i in range(num):
            for j in range(num):
                for k in range(num):
                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[x, y, z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[-x, y, z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[x, -y, z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[-x, -y, z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[x, y, -z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[-x, y, -z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[x, -y, -z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

                    x = np.random.uniform(np.power(x_range[i], bounds_power),
                                          np.power(x_range[i + 1], bounds_power)) * x_bound
                    y = np.random.uniform(np.power(y_range[j], bounds_power),
                                          np.power(y_range[j + 1], bounds_power)) * y_bound
                    z = np.random.uniform(np.power(z_range[k], bounds_power),
                                          np.power(z_range[k + 1], bounds_power)) * z_bound
                    p = np.array([[-x, -y, -z, 0.0]], dtype=np.float32)
                    positions = np.append(positions, p, axis=0)

        return positions

    @staticmethod
    def start():
        glutMainLoop()


if __name__ == "__main__":
    # m = Model()
    # m.add_rounded_box(np.array([-1.0,-1.0,-1.0], dtype=np.float32),np.array([1.0,1.0,1.0], dtype=np.float32), offset=np.array([0.1,0.1,0.1], dtype=np.float32))
    #app = Application(seed=5461)
    # app = Application(seed=87753) # screens_256
    # app = Application(seed=4268) # screens_256 (x 100)
    app = Application(seed=9268)
    app.initialize()
    app.start()

