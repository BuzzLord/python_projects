from __future__ import print_function
from os.path import join
from OpenGL.GLUT import *
from generator.model import *

import sys
sys.path.insert(1, join('..', 'autoencoder'))
from siren01 import *

vertex_shader_src = """
            #version 150
            uniform mat4 Model;
            uniform mat4 View;
            uniform mat4 Proj;
            uniform mat3 NormalMatrix;

            in vec4 Position;
            in vec2 TexCoord;

            out vec2 oTexCoord;

            void main() {
                gl_Position = (Proj * View * Model * Position);
                oTexCoord = TexCoord;
            }"""

fragment_shader_src = """
            #version 150
            uniform sampler2D Texture0;
            in vec2 oTexCoord;
            out vec4 FragColor;
            void main() {
                FragColor = texture2D(Texture0, oTexCoord);
                FragColor.a = 1.0;
            }"""


class RenderSiren:
    def __init__(self, siren, device):
        self.window_name = b'Render Siren'
        self.window_size = (512, 512)
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self.camera_position = np.array([0.0, 0.0, 0.0, 1.], dtype='float32')
        self.cube_clock = 0.0
        self.proj = perspective(60.0, 1.0, 0.1, 200.0)
        self.fov = 60.0
        self.aspect = self.window_size[0] / self.window_size[1]

        self.siren = siren
        self.device = device

        self.screen_space_quad = None
        self.ssq_tex = None
        self.ssq_shader = None
        self.ssq_view = lookat(np.array([0.0, 0.0, 0.0], dtype=np.float32),
                               np.array([0.0, 0.0, -1.0], dtype=np.float32),
                               np.array([0.0, 1.0, 0.0], dtype=np.float32))
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
        glClearColor(0.5, 0.5, 0.5, 1.0)

        # Should already be set to eval, but disable gradients
        self.siren.eval()
        torch.no_grad()

        self.__setup_render_textures()

    def __setup_render_textures(self):
        self.ssq_tex = TextureBuffer(False, self.window_size, None, clamp=GL_CLAMP_TO_EDGE)
        self.ssq_shader = ShaderFill(vertex_shader_src, fragment_shader_src)
        self.ssq_shader.add_texture_buffer(self.ssq_tex)

        self.screen_space_quad = Model(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), shader=self.ssq_shader)
        self.screen_space_quad.add_oriented_quad((1.0, 1.0, 0.5), (0.0, 0.0, 0.5))
        self.screen_space_quad.allocate_buffers()

    def __generate_view(self):
        roll_pitch_yaw = np.dot(roty(self.camera_yaw), rotx(self.camera_pitch))
        final_up = transform(roll_pitch_yaw, np.array([0.0, 1.0, 0.0, 0.0], dtype='float32'))
        final_forward = transform(roll_pitch_yaw, np.array([0.0, 0.0, -1.0, 0.0], dtype='float32'))

        theta = np.linspace(-0.7, 0.7, self.window_size[0]) * np.ones((self.window_size[1], 1)) + self.camera_yaw
        phi = (np.linspace(-0.7, 0.7, self.window_size[1]) * np.ones((self.window_size[0], 1))).transpose() - self.camera_pitch
        xyz = np.zeros((self.window_size[0],self.window_size[1],3)) + self.camera_position[0:3]

        np_input = np.concatenate((xyz, phi.reshape(self.window_size[0],self.window_size[1],1),
                                   theta.reshape(self.window_size[0],self.window_size[1],1)), axis=2)
        view_input = torch.from_numpy(np_input).view((self.window_size[0]*self.window_size[1],5)).to(self.device, dtype=torch.float32)
        view_output = self.siren(view_input).cpu().data
        view_output = ((view_output * 0.5) + 0.5).clamp(0, 1).view((self.window_size[0],self.window_size[1],3))
        view_output = np.concatenate((np.array(view_output.numpy(), dtype=np.float32),
                                      np.ones((self.window_size[0],self.window_size[1], 1), dtype=np.float32)), axis=2)
        view_output = np.array(view_output * 255.0, np.uint8)
        self.ssq_tex.copy_data(view_output, bind=True)

    def __render_loop(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.window_size[0], self.window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.screen_space_quad.render(self.ssq_view, self.ssq_proj)

        glutSwapBuffers()
        return

    def __update_func(self):
        self.__generate_view()
        glutPostRedisplay()

    def __keyboard_func(self, key, x, y):
        # print("Keyboard func saw: " + str(key))
        speed = 0.05
        if key == b'\x1b':
            sys.exit(0)
        elif key == b'a':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw - 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw - 90.0)), 0.0])
            self.camera_position += speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b'd':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw + 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw + 90.0)), 0.0])
            self.camera_position += speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b'w':
            dir = np.array([-np.sin(np.deg2rad(self.camera_yaw)), 0.0, -np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b's':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw)), 0.0, np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b'A':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw - 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw - 90.0)), 0.0])
            self.camera_position += 4.0 * speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b'D':
            dir = np.array(
                [np.sin(np.deg2rad(self.camera_yaw + 90.0)), 0.0, np.cos(np.deg2rad(self.camera_yaw + 90.0)), 0.0])
            self.camera_position += 4.0 * speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b'W':
            dir = np.array([-np.sin(np.deg2rad(self.camera_yaw)), 0.0, -np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += 4.0 * speed * dir
            logging.info("Position: {}".format(self.camera_position))
        elif key == b'S':
            dir = np.array([np.sin(np.deg2rad(self.camera_yaw)), 0.0, np.cos(np.deg2rad(self.camera_yaw)), 0.0])
            self.camera_position += 4.0 * speed * dir
            logging.info("Position: {}".format(self.camera_position))

    def __special_func(self, key, x, y):
        states = glutGetModifiers()
        if states & GLUT_ACTIVE_SHIFT:
            d = 0.05
        else:
            d = 0.01

        #  print("Special func saw: " + str(key))
        if key == GLUT_KEY_LEFT:
            self.camera_yaw = (self.camera_yaw + d) % 360.0
        elif key == GLUT_KEY_RIGHT:
            self.camera_yaw = (self.camera_yaw - d) % 360.0
        elif key == GLUT_KEY_UP:
            self.camera_pitch = min(self.camera_pitch + d, 90.0)
        elif key == GLUT_KEY_DOWN:
            self.camera_pitch = max(self.camera_pitch - d, -90.0)

    @staticmethod
    def start():
        glutMainLoop()


def main(custom_args=None):
    model_number = "01"
    parser = argparse.ArgumentParser(description='PyTorch SIREN Model ' + model_number + ' Renderer')

    parser.add_argument('--load-model-state', type=str, metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--model-path', type=str, metavar='PATH',
                        help='pathname for this models output (default siren'+model_number+')')
    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    parser.add_argument('--hidden-size', type=int, default=384, metavar='H',
                        help='Hidden layer size of Siren (default: 384)')
    parser.add_argument('--hidden-layers', type=int, default=6, metavar='N',
                        help='Hidden layer count of Siren (default: 6)')
    parser.add_argument('--pos-encoding', type=int, default=10, metavar='L',
                        help='Positional encoding harmonics (default: 10)')
    args = parser.parse_args(args=custom_args)

    if args.model_path is None:
        args.model_path = join("..", "autoencoder", "siren{}".format(model_number),
                               "model_{:d}_{:d}_{:d}".format(args.pos_encoding, args.hidden_size, args.hidden_layers))

    if not os.path.exists(args.model_path):
        raise RuntimeError("Model path {} does not exist".format(args.model_path))

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler(join(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    device = torch.device("cuda")

    logging.info("Siren configured with pos_encoding = {}, hidden_size = {}, hidden_layers = {}".format(
        args.pos_encoding, args.hidden_size, args.hidden_layers))
    model = Siren(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, pos_encoding_levels=args.pos_encoding)
    if args.load_model_state is None:
        raise RuntimeError("Missing load model state argument!")

    model_path = os.path.join(args.model_path, args.load_model_state)
    if not os.path.exists(model_path):
        raise RuntimeError("Could not find model state: {}".format(model_path))

    logging.info("Loading model from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device, dtype=torch.float32)

    app = RenderSiren(siren=model, device=device)
    app.initialize()
    app.start()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(threadName)s] %(message)s')
    main()
