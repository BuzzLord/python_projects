import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from gl_utils import *
from gl_math import *


class Application:

    def __init__(self, seed=1234):
        self.window_name = b'Application'
        self.window_size = (400, 400)
        self.yaw = 0.0
        self.pitch = 0.0
        self.pos2 = np.array([0.,0.,0.,1.], dtype='float32')
        self.cube_clock = 0.0

        self.scene = RandomScene(seed)

    def initialize(self, args=[]):
        glutInit(args)
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.window_size[0], self.window_size[1])
        glutCreateWindow(self.window_name)
        glutDisplayFunc(self.__render_loop)

        glEnable(GL_DEPTH_TEST)
        glFrontFace(GL_CCW)
        glDisable(GL_CULL_FACE)
        glClearColor(0.15, 0.25, 0.15, 1.0)

        self.scene.generate_shaders()
        self.scene.generate_models()

    def start(self):
        glutMainLoop()

    def __render_loop(self):
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        roll_pitch_yaw = np.dot(roty(self.yaw), rotx(self.pitch))
        final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
        final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))
        shifted_eye_pos = transform(translate(np.array([0.0, 1.5, 0.0], dtype='float32')), self.pos2)

        view = lookat(shifted_eye_pos, shifted_eye_pos + final_forward, final_up)
        proj = perspective(60.0, 1.0, 0.1, 200.0)
        self.scene.render(view, proj)

        glutSwapBuffers()
        return


if __name__ == "__main__":

    app = Application()
    app.initialize()

    app.start()

