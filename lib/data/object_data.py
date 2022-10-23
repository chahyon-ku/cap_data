import numpy


class ObjectData:
    def __init__(self, name: str, shape_pair: tuple, material_pair: tuple, color_pair: tuple, scale_pair: tuple,
                 pose: numpy.ndarray):
        self.name = name
        self.shape_pair = shape_pair
        self.material_pair = material_pair
        self.color_pair = color_pair
        self.scale_pair = scale_pair
        self.pose = pose
