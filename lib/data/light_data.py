import numpy


class LightData:
    def __init__(self, name: str, type: str, energy: float, pose: numpy.ndarray):
        self.name = name
        self.type = type
        self.energy = energy
        self.pose = pose


def empty():
    return LightData(None, None, None, None)