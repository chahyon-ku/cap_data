import json
import os

import numpy

import lib.data.scene_data


def empty():
    return RenderData(None, None, None, None, None, None, None, None, None, None)


def from_args(name, args):
    output_dir = os.path.abspath(args.output_dir)
    return RenderData(name, output_dir, args.width, args.height, args.render_tile_size, args.device_type,
                      args.render_num_samples, args.render_min_bounces, args.render_max_bounces, args.modes)


class RenderData:
    def __init__(self, name, output_dir, width, height, render_tile_size, device_type, render_num_samples,
                 render_min_bounces, render_max_bounces, modes):
        self.name = name
        
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.render_tile_size = render_tile_size
        self.device_type = device_type
        self.render_num_samples = render_num_samples
        self.render_min_bounces = render_min_bounces
        self.render_max_bounces = render_max_bounces
        self.modes = modes

        self.scenes_data = {}


def to_object(input):
    if input is None:
        return None

    if isinstance(input, (str, bool, int, float)):
        return input

    if isinstance(input, (tuple, list)):
        output = []
        for value in input:
            output.append(to_object(value))
        return output

    if isinstance(input, dict):
        output = {}
        for key, value in input.items():
            output[key] = to_object(value)
        return output

    if isinstance(input, numpy.ndarray):
        return input.tolist()

    if isinstance(input, (lib.data.render_data.RenderData, lib.data.scene_data.SceneData,
                          lib.data.camera_data.CameraData, lib.data.object_data.ObjectData,
                          lib.data.light_data.LightData)):
        return to_object(input.__dict__)

    raise Exception('unknown type', type(input))


def from_object(input, type):
    if type == RenderData:
        output = lib.data.render_data.empty()
        for name, value in input.items():
            if name == 'scenes_data':
                output.__setattr__(name, {key: from_object(v, lib.data.scene_data.SceneData)
                                          for key, v in value.items()})
            else:
                output.__setattr__(name, value)
        return output
    elif type == lib.data.scene_data.SceneData:
        output = lib.data.scene_data.empty()
        for name, value in input.items():
            if name == 'objects_data':
                output.__setattr__(name, {key: from_object(v, lib.data.object_data.ObjectData)
                                          for key, v in value.items()})
            elif name == 'lights_data':
                output.__setattr__(name, {key: from_object(v, lib.data.light_data.LightData)
                                          for key, v in value.items()})
            elif name == 'cameras_data':
                output.__setattr__(name, {key: from_object(v, lib.data.camera_data.CameraData)
                                          for key, v in value.items()})
            else:
                output.__setattr__(name, value)
        return output
    elif type == lib.data.object_data.ObjectData:
        output = lib.data.object_data.empty()
        for name, value in input.items():
            if name in ('color_pair', 'scale_pair'):
                output.__setattr__(name, from_object(value, tuple))
            elif name == 'pose':
                output.__setattr__(name, from_object(value, numpy.ndarray))
            else:
                output.__setattr__(name, value)
    elif type == lib.data.camera_data.CameraData:
        output = lib.data.camera_data.empty()
        for name, value in input.items():
            if name == 'pose':
                output.__setattr__(name, from_object(value, numpy.ndarray))
            else:
                output.__setattr__(name, value)
    elif type == lib.data.light_data.LightData:
        output = lib.data.light_data.empty()
        for name, value in input.items():
            if name == 'pose':
                output.__setattr__(name, from_object(value, numpy.ndarray))
            else:
                output.__setattr__(name, value)
    elif type == tuple:
        output = (input[0], from_object(input[1], numpy.ndarray))
    elif type == numpy.ndarray:
        output = numpy.array(input)
    else:
        raise Exception('unknown type', type)

    return output

