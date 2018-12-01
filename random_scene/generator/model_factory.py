from generator.model import *


class ModelFactory:

    def __init__(self):
        self.models = {}
        self.get_calls = {
            "chair": self.get_chair,
            "table": self.get_table,
            "tablechair1": self.get_table_chair_1,
            "tablechair2": self.get_table_chair_2,
            "dresser": self.get_dresser,
            "beddoubleframe1": self.get_bed_double_frame_1,
            "beddoubleframeheadboard1": self.get_bed_double_frame_headboard_1,
            "beddoubleframeheadboard2": self.get_bed_double_frame_headboard_2,
            "beddouble": self.get_bed_double,
            "stool": self.get_stool,
            "couch3": self.get_couch_3,
            "couch2": self.get_couch_2,
            "bigchair": self.get_big_chair,
            "smalltable": self.get_small_table,
            "coffeetable": self.get_coffee_table,
            "pluscoffeetable": self.get_plus_coffee_table,
            "door1": self.get_door_1
        }

    def get_model_by_key(self, key):
        return self.get_calls[key]()

    def get_chair(self):
        key = "chair"
        if key in self.models.keys():
            return self.models[key].copy()
        chair = Model()
        chair.add_solid_color_box((-0.30, 0.5, -0.30), ( 0.30, 0.55, 0.30))
        chair.add_solid_color_box((-0.30, 0.0, -0.30), (-0.24, 1.0, -0.24))
        chair.add_solid_color_box((-0.30, 0.0,  0.24), (-0.24, 0.5,  0.30))
        chair.add_solid_color_box(( 0.24, 0.0,  0.24), ( 0.30, 0.5,  0.30))
        chair.add_solid_color_box(( 0.24, 0.0, -0.30), ( 0.30, 1.0, -0.24))
        chair.add_solid_color_box((-0.29, 0.92,-0.30), ( 0.29, 0.97,-0.25))
        # chair.add_offset(np.array([0.0,0.0,0.0], dtype=np.float32))
        self.models[key] = chair
        return chair.copy()

    def get_table(self):
        key = "table"
        if key in self.models.keys():
            return self.models[key].copy()
        table = Model()
        table.add_solid_color_box((-0.9, 0.7,-0.5), ( 0.9, 0.8, 0.5))
        table.add_solid_color_box((-0.9, 0.0,-0.5), (-0.8, 0.7,-0.4))
        table.add_solid_color_box((-0.9, 0.0, 0.4), (-0.8, 0.7, 0.5))
        table.add_solid_color_box(( 0.8, 0.0, 0.4), ( 0.9, 0.7, 0.5))
        table.add_solid_color_box(( 0.8, 0.0,-0.5), ( 0.9, 0.7,-0.4))
        self.models[key] = table
        return table.copy()

    def get_table_chair_1(self):
        key = "tablechair1"
        if key in self.models.keys():
            return self.models[key].copy()
        chair = Model()
        chair.add_solid_color_box((-0.30, 0.5, -0.30), ( 0.30, 0.55, 0.30))
        chair.add_solid_color_box((-0.30, 0.0, -0.30), (-0.24, 1.0, -0.24))
        chair.add_solid_color_box((-0.30, 0.0,  0.24), (-0.24, 0.5,  0.30))
        chair.add_solid_color_box(( 0.24, 0.0,  0.24), ( 0.30, 0.5,  0.30))
        chair.add_solid_color_box(( 0.24, 0.0, -0.30), ( 0.30, 1.0, -0.24))
        chair.add_solid_color_box((-0.29, 0.92,-0.30), ( 0.29, 0.97,-0.25))
        chair.add_offset(np.array([0.4,0.0,-0.6], dtype=np.float32))
        self.models[key] = chair
        return chair.copy()

    def get_table_chair_2(self):
        key = "tablechair2"
        if key in self.models.keys():
            return self.models[key].copy()
        chair = Model()
        chair.add_solid_color_box((-0.30, 0.5, -0.30), ( 0.30, 0.55, 0.30))
        chair.add_solid_color_box((-0.30, 0.0, -0.30), (-0.24, 1.0, -0.24))
        chair.add_solid_color_box((-0.30, 0.0,  0.24), (-0.24, 0.5,  0.30))
        chair.add_solid_color_box(( 0.24, 0.0,  0.24), ( 0.30, 0.5,  0.30))
        chair.add_solid_color_box(( 0.24, 0.0, -0.30), ( 0.30, 1.0, -0.24))
        chair.add_solid_color_box((-0.29, 0.92,-0.30), ( 0.29, 0.97,-0.25))
        chair.add_offset(np.array([-0.4,0.0,-0.7], dtype=np.float32))
        self.models[key] = chair
        return chair.copy()

    def get_dresser(self):
        key = "dresser"
        if key in self.models.keys():
            return self.models[key].copy()
        dresser = Model()
        # main body
        dresser.add_solid_color_box((-0.5, 0.03, -0.25), ( 0.5, 1.3, 0.25))
        # legs
        dresser.add_solid_color_box((-0.5, 0.0, -0.25), (-0.4, 0.03, -0.15))
        dresser.add_solid_color_box(( 0.4, 0.0, -0.25), ( 0.5, 0.03, -0.15))
        dresser.add_solid_color_box((-0.5, 0.0,  0.15), (-0.4, 0.03,  0.25))
        dresser.add_solid_color_box(( 0.4, 0.0,  0.15), ( 0.5, 0.03,  0.25))
        # drawers
        dresser.add_solid_color_box((-0.45, 0.13, -0.28), ( 0.45, 0.43, -0.2))
        dresser.add_solid_color_box((-0.45, 0.53, -0.28), ( 0.45, 0.83, -0.2))
        dresser.add_solid_color_box((-0.45, 0.93, -0.28), ( 0.45, 1.23, -0.2))
        # handles
        dresser.add_solid_color_box((-0.35, 0.255, -0.3), ( -0.15, 0.305, -0.2))
        dresser.add_solid_color_box(( 0.15, 0.255, -0.3), (  0.35, 0.305, -0.2))
        dresser.add_solid_color_box((-0.35, 0.655, -0.3), ( -0.15, 0.705, -0.2))
        dresser.add_solid_color_box(( 0.15, 0.655, -0.3), (  0.35, 0.705, -0.2))
        dresser.add_solid_color_box((-0.35, 1.055, -0.3), ( -0.15, 1.105, -0.2))
        dresser.add_solid_color_box(( 0.15, 1.055, -0.3), (  0.35, 1.105, -0.2))
        #dresser.add_offset(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.models[key] = dresser
        return dresser.copy()

    def get_bed_double_frame_1(self):
        key = "beddoubleframe1"
        if key in self.models.keys():
            return self.models[key].copy()
        frame = Model()
        # base
        frame.add_solid_color_box((-0.7, 0.12, -1.01), ( 0.7, 0.2, 1.06))
        # legs
        frame.add_solid_color_box((-0.7, 0.0, -1.01),  (-0.65, 0.12, -0.96))
        frame.add_solid_color_box((-0.7, 0.0,  1.01), (-0.65, 0.12,  1.06))
        frame.add_solid_color_box(( 0.65, 0.0, -1.01),  ( 0.7, 0.12, -0.96))
        frame.add_solid_color_box(( 0.65, 0.0,  1.01), ( 0.7, 0.12,  1.06))
        #frame.add_offset(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.models[key] = frame
        return frame.copy()

    def get_bed_double_frame_headboard_1(self):
        key = "beddoubleframeheadboard1"
        if key in self.models.keys():
            return self.models[key].copy()
        headboard = Model()
        # bars
        headboard.add_solid_color_box((-0.7, 0.2,  1.01),  (-0.65, 1.0, 1.06))
        headboard.add_solid_color_box(( 0.65, 0.2,  1.01), ( 0.7, 1.0,  1.06))
        headboard.add_solid_color_box((-0.35, 0.2,  1.01),  (-0.3, 1.0, 1.06))
        headboard.add_solid_color_box(( 0.3, 0.2,  1.01), ( 0.35, 1.0,  1.06))
        headboard.add_solid_color_box(( -0.025, 0.2,  1.01), ( 0.025, 1.0, 1.06))
        # top
        headboard.add_solid_color_box((-0.7, 1.0, 1.01), ( 0.7, 1.2, 1.06))
        #headboard.add_offset(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.models[key] = headboard
        return headboard.copy()

    def get_bed_double_frame_headboard_2(self):
        key = "beddoubleframeheadboard2"
        if key in self.models.keys():
            return self.models[key].copy()
        headboard = Model()
        # back
        headboard.add_solid_color_box((-0.7, 0.2, 1.01), ( 0.7, 0.9, 1.06))
        # top
        headboard.add_rounded_box((-0.7, 0.9, 0.99), ( 0.7, 1.2, 1.06), offset=(0.0, 0.01, 0.0))
        #headboard.add_offset(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.models[key] = headboard
        return headboard.copy()

    def get_bed_double(self):
        key = "beddouble"
        if key in self.models.keys():
            return self.models[key].copy()
        bed = Model()
        bed.add_rounded_box((-0.68, 0.2, -1.0), ( 0.68, 0.6, 1.0), offset=(0.02, 0.02, 0.02))
        bed.add_rounded_box((-0.6, 0.6,  0.6),  (-0.05, 0.7, 0.96), offset=(0.02, 0.02, 0.02))
        bed.add_rounded_box(( 0.05, 0.6, 0.6), ( 0.6, 0.7, 0.96), offset=(0.02, 0.02, 0.02))
        # bed.add_offset(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.models[key] = bed
        return bed.copy()

    def get_stool(self):
        key = "stool"
        if key in self.models.keys():
            return self.models[key].copy()
        stool = Model()
        stool.add_solid_color_box((-0.30, 0.5, -0.30), ( 0.30, 0.55, 0.30))
        stool.add_solid_color_box((-0.30, 0.0, -0.30), (-0.24, 0.5, -0.24))
        stool.add_solid_color_box((-0.30, 0.0,  0.24), (-0.24, 0.5,  0.30))
        stool.add_solid_color_box(( 0.24, 0.0,  0.24), ( 0.30, 0.5,  0.30))
        stool.add_solid_color_box(( 0.24, 0.0, -0.30), ( 0.30, 0.5, -0.24))
        self.models[key] = stool
        return stool.copy()

    def get_couch_3(self):
        key = "couch3"
        if key in self.models.keys():
            return self.models[key].copy()
        couch = Model()
        # base & back
        couch.add_solid_color_box((-0.9, 0.1, -0.35), (0.9, 0.3, 0.4))
        couch.add_solid_color_box((-0.9, 0.1,  0.35), (0.9, 1.1, 0.4))
        # arm rests
        couch.add_rounded_box((-1.1, 0.0, -0.35), (-0.9, 0.8, 0.4), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box(( 0.9, 0.0, -0.35), ( 1.1, 0.8, 0.4), offset=(0.02, 0.02, 0.02))
        # cushions
        couch.add_rounded_box((-0.9, 0.3, -0.4),  (-0.31, 0.6, 0.2), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box((-0.29, 0.3, -0.4), ( 0.29,  0.6, 0.2), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box(( 0.31, 0.3, -0.4), ( 0.9,  0.6, 0.2), offset=(0.02, 0.02, 0.02))
        # back cushions
        couch.add_rounded_box((-0.9, 0.6, 0.2),  (-0.31, 1.2, 0.36), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box((-0.29, 0.6, 0.2), ( 0.29, 1.2, 0.36), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box(( 0.31, 0.6, 0.2), ( 0.9,  1.2, 0.36), offset=(0.02, 0.02, 0.02))
        self.models[key] = couch
        return couch.copy()

    def get_couch_2(self):
        key = "couch2"
        if key in self.models.keys():
            return self.models[key].copy()
        couch = Model()
        # base & back
        couch.add_solid_color_box((-0.6, 0.1, -0.35), (0.6, 0.3, 0.4))
        couch.add_solid_color_box((-0.6, 0.1,  0.35), (0.6, 1.1, 0.4))
        # arm rests
        couch.add_rounded_box((-0.8, 0.0, -0.35), (-0.6, 0.8, 0.4), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box(( 0.6, 0.0, -0.35), ( 0.8, 0.8, 0.4), offset=(0.02, 0.02, 0.02))
        # cushions
        couch.add_rounded_box((-0.6, 0.3, -0.4), (-0.01, 0.6, 0.2), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box((0.01, 0.3, -0.4), ( 0.6,  0.6, 0.2), offset=(0.02, 0.02, 0.02))
        # back cushions
        couch.add_rounded_box((-0.6, 0.6, 0.2), (-0.01, 1.2, 0.36), offset=(0.02, 0.02, 0.02))
        couch.add_rounded_box((0.01, 0.6, 0.2), ( 0.6,  1.2, 0.36), offset=(0.02, 0.02, 0.02))
        self.models[key] = couch
        return couch.copy()

    def get_big_chair(self):
        key = "bigchair"
        if key in self.models.keys():
            return self.models[key].copy()
        bigchair = Model()
        # base & back
        bigchair.add_solid_color_box((-0.35, 0.1, -0.35), (0.35, 0.3, 0.4))
        bigchair.add_solid_color_box((-0.35, 0.1,  0.35), (0.35, 1.1, 0.4))
        # arm rests
        bigchair.add_rounded_box((-0.55, 0.0, -0.35), (-0.35, 0.8, 0.4), offset=(0.02, 0.02, 0.02))
        bigchair.add_rounded_box(( 0.35, 0.0, -0.35), ( 0.55, 0.8, 0.4), offset=(0.02, 0.02, 0.02))
        # cushions
        bigchair.add_rounded_box((-0.35, 0.3, -0.4), ( 0.35, 0.6, 0.2), offset=(0.02, 0.02, 0.02))
        # back cushions
        bigchair.add_rounded_box((-0.35, 0.6, 0.2), ( 0.35, 1.2, 0.36), offset=(0.02, 0.02, 0.02))
        self.models[key] = bigchair
        return bigchair.copy()

    def get_small_table(self):
        key = "smalltable"
        if key in self.models.keys():
            return self.models[key].copy()
        table = Model()
        table.add_solid_color_box((-0.4, 0.5,-0.4), ( 0.4, 0.6, 0.4))
        table.add_solid_color_box((-0.4, 0.0,-0.4), (-0.3, 0.5,-0.3))
        table.add_solid_color_box((-0.4, 0.0, 0.3), (-0.3, 0.5, 0.4))
        table.add_solid_color_box(( 0.3, 0.0, 0.3), ( 0.4, 0.5, 0.4))
        table.add_solid_color_box(( 0.3, 0.0,-0.4), ( 0.4, 0.5,-0.3))
        self.models[key] = table
        return table.copy()

    def get_coffee_table(self):
        key = "coffeetable"
        if key in self.models.keys():
            return self.models[key].copy()
        table = Model()
        table.add_solid_color_box((-0.9, 0.5,-0.5), ( 0.9, 0.6, 0.5))
        table.add_solid_color_box((-0.9, 0.0,-0.5), (-0.8, 0.5,-0.4))
        table.add_solid_color_box((-0.9, 0.0, 0.4), (-0.8, 0.5, 0.5))
        table.add_solid_color_box(( 0.8, 0.0, 0.4), ( 0.9, 0.5, 0.5))
        table.add_solid_color_box(( 0.8, 0.0,-0.5), ( 0.9, 0.5,-0.4))
        self.models[key] = table
        return table.copy()

    def get_plus_coffee_table(self):
        key = "pluscoffeetable"
        if key in self.models.keys():
            return self.models[key].copy()
        table = Model()
        table.add_solid_color_box((-0.9, 0.5,-0.5), ( 0.9, 0.6, 0.5))
        table.add_solid_color_box((-0.9, 0.0,-0.5), (-0.8, 0.5,-0.4))
        table.add_solid_color_box((-0.9, 0.0, 0.4), (-0.8, 0.5, 0.5))
        table.add_solid_color_box(( 0.8, 0.0, 0.4), ( 0.9, 0.5, 0.5))
        table.add_solid_color_box(( 0.8, 0.0,-0.5), ( 0.9, 0.5,-0.4))
        table.add_offset(np.array([0.0, 0.0, -1.5], dtype=np.float32))
        self.models[key] = table
        return table.copy()

    def get_door_1(self, dims=(0.8, 2.0, 0.05), gap=0.02):
        key = "door1"
        if key in self.models.keys():
            return self.models[key].copy()
        door = Model()
        x, z, h = dims[0], dims[2]/2.0, dims[1]
        door.add_solid_color_box(( 0.0, gap,-z), (x, h, z))
        door.add_rounded_box(( 0.75 * x - 0.02, 0.45 * h - 0.02, -z - 0.06),
                             ( 0.75 * x + 0.02, 0.45 * h + 0.02, z + 0.06),
                             offset=(0.01, 0.01, 0.01))
        door.add_sphere(c=( 0.75 * x, 0.45 * h, -z - 0.08), r=0.04)
        door.add_sphere(c=( 0.75 * x, 0.45 * h, z + 0.08), r=0.04)
        self.models[key] = door
        return door.copy()
