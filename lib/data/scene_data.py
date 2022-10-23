import json
import os


class SceneData:
    def __init__(self, args):
        self.base_scene_blendfile = os.path.abspath(args.base_scene_blendfile)
        with open(args.properties_json, 'r') as f:
            self.properties = json.load(f)
        self.shape_dir = os.path.abspath(args.shape_dir)
        self.material_dir = os.path.abspath(args.material_dir)
        self.objects_data = {}
        self.cameras_data = {}