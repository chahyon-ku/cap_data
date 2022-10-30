import json
import os


def empty():
    return SceneData(None, None, None, None, None, None)


def from_args(name, args, reset_scene):
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
    if args.base_scene_blendfile is None:
        base_blend_scene_blendfile = None
    else:
        base_blend_scene_blendfile = os.path.abspath(args.base_scene_blendfile)
    return SceneData(name, base_blend_scene_blendfile, properties, args.shape_dir, args.material_dir, reset_scene)


class SceneData:
    def __init__(self, name: str, base_scene_blendfile, properties, shape_dir, material_dir, reset_scene):
        self.name = name
        self.base_scene_blendfile = base_scene_blendfile
        self.reset_scene = reset_scene
        self.properties = properties
        self.shape_dir = shape_dir
        self.material_dir = material_dir
        self.objects_data = {}
        self.cameras_data = {}
        self.lights_data = {}
