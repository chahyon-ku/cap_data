import numpy


def empty():
    return ObjectData(None, None, None, None, None, None)


class ObjectData:
    def __init__(self, name, shape_pair, material_pair, color_pair, scale_pair, pose):
        self.name = name
        self.shape_pair = shape_pair
        self.material_pair = material_pair
        self.color_pair = color_pair
        self.scale_pair = scale_pair
        self.pose = pose

