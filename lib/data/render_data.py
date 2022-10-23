import os


class RenderData:
    def __init__(self, dir, name, args):
        self.dir = os.path.abspath(dir)
        self.name = name
        self.width = args.width
        self.height = args.height
        self.render_tile_size = args.render_tile_size
        self.device_type = args.device_type
        self.render_num_samples = args.render_num_samples
        self.render_min_bounces = args.render_min_bounces
        self.render_max_bounces = args.render_max_bounces
        self.modes = args.modes
