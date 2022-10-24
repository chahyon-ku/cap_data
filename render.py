import argparse
import json
import os
import time

import lib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--renders_dir', type=str, default='output/caps')
    args = parser.parse_args()

    start_time = time.time()
    n_renders = 0
    n_scenes = 0
    for dir_entry in os.scandir(args.renders_dir):
        if dir_entry.name.endswith('.json'):
            with open(dir_entry.path, 'r') as f:
                render_data = lib.data.render_data.from_object(json.load(f), lib.data.render_data.RenderData)
                lib.blend.blend_render(render_data)
                n_renders += 1
                n_scenes += len(render_data.scenes_data)

    print('finished rendering', n_renders, 'renders with', n_scenes, f'scenes in {time.time() - start_time:.3f} seconds')
