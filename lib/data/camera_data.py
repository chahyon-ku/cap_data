import numpy


class CameraData:
    def __init__(self, name: str, pose: numpy.ndarray):
        self.name = name
        self.pose = pose


def empty():
    return CameraData(None, None)