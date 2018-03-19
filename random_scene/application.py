import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from random_scene import *


Cubemap = namedtuple('Cubemap', ['tex','depth','rotation'])


class Application:

    def __init__(self, seed=1234):
        self.window_name = b'Application'
        self.window_size = (400, 400)
        self.yaw = 180.0
        self.pitch = 0.0
        self.pos2 = np.array([0.,1.5,0.,1.], dtype='float32')
        self.cube_clock = 0.0
        self.proj = perspective(60.0, 1.0, 0.1, 200.0)

        self.scene = RandomScene(seed)
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
        glClearColor(0.15, 0.25, 0.15, 1.0)

        self.scene.generate_shaders()
        self.scene.generate_models()
        self.__setup_render_textures()

    def __setup_render_textures(self):
        cubemap_size = (256, 256)
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

        self.screen_space_quad = Model(np.array([0.0, 0.0, 0.0], dtype=np.float32), self.warp_shader)
        self.screen_space_quad.add_oriented_quad((1.0,1.0,0.5),(0.0,0.0,0.5))
        self.screen_space_quad.allocate_buffers()

    def __render_warped_view(self, position):
        roll_pitch_yaw = np.dot(roty(self.yaw), rotx(self.pitch))
        pos_offset = transform(roll_pitch_yaw, -position)
        final_pos = transform(translate(pos_offset[:3]), self.pos2)

        for cubemap in self.cubemap:
            cubemap.tex.set_and_clear_render_surface(cubemap.depth)
            final_roll_pitch_yaw = np.dot(roll_pitch_yaw, cubemap.rotation)
            final_up = transform(final_roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
            final_forward = transform(final_roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
            #shifted_eye_pos = transform(translate(np.array([0.0, 0.0, 0.0], dtype='float32')), self.pos2)

            view = lookat(final_pos, final_pos + final_forward, final_up)
            self.scene.render(view, self.cube_proj)
            cubemap.tex.unset_render_surface()

        self.warp_tex.set_and_clear_render_surface(self.warp_dep)
        self.screen_space_quad.render(self.ssq_view, self.ssq_proj)
        self.warp_tex.unset_render_surface()

    def __render_loop(self):

        self.__render_warped_view(np.array([0.0,0.0,0.0,1.0], dtype=np.float32))

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #self.screen_space_quad.render(self.ssq_view, self.ssq_proj)

        roll_pitch_yaw = np.dot(roty(self.yaw), rotx(self.pitch))
        final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
        final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
        shifted_eye_pos = transform(translate(np.array([0.0, 0.0, 0.0], dtype='float32')), self.pos2)

        view = lookat(shifted_eye_pos, shifted_eye_pos + final_forward, final_up)
        self.scene.render(view, self.proj)

        glutSwapBuffers()
        return

    def __update_func(self):
        glutPostRedisplay()

    def __keyboard_func(self, key, x, y):
        #print("Keyboard func saw: " + str(key))
        if key == b'\x1b':
            sys.exit(0)

    def __special_func(self, key, x, y):
        #print("Special func saw: " + str(key))
        if key == GLUT_KEY_LEFT:
            self.yaw = (self.yaw + 1.0) % 360.0
        elif key == GLUT_KEY_RIGHT:
            self.yaw = (self.yaw - 1.0) % 360.0
        elif key == GLUT_KEY_UP:
            self.pitch = min(self.pitch + 1.0, 90.0)
        elif key == GLUT_KEY_DOWN:
            self.pitch = max(self.pitch - 1.0, -90.0)

    def start(self):
        glutMainLoop()


if __name__ == "__main__":
    app = Application()
    app.initialize()
    app.start()

