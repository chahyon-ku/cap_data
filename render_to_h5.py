import argparse
import collections
import copy
import io
import json
import os
import tqdm
import numpy
from PIL import Image
import h5py
import lib


def write(render_data, render_dir, h5f):
    render_group = h5f.create_group(render_data.name)
    render_group.create_dataset('render_data', data=json.dumps(lib.data.render_data.to_object(render_data)))
    for scene_name, scene_data in tqdm.tqdm(render_data.scenes_data.items(), leave=False):
        scene_group = render_group.create_group(scene_name)
        for camera_name in scene_data.cameras_data:
            camera_group = scene_group.create_group(camera_name)
            for mode in render_data.modes:
                render_data.output_dir = render_dir
                png_path = os.path.join(render_data.output_dir, render_data.name,
                                        f'{scene_name}_{camera_name}_{mode}.png')
                with Image.open(png_path) as imf:
                    buf = io.BytesIO()
                    imf.save(buf, 'png')
                camera_group.create_dataset(mode, data=numpy.array(buf.getvalue()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_dir', type=str, default='./output/bunny_1d')
    parser.add_argument('--output_h5', type=str, default='./output/bunny_1d.h5')
    args = parser.parse_args()

    with h5py.File(args.output_h5, 'w') as h5f:
        dir_entries = list(os.scandir(args.render_dir))
        dir_postfix = collections.OrderedDict()
        dir_tqdm = tqdm.tqdm(dir_entries)
        for dir_entry in dir_tqdm:
            dir_postfix['file'] = dir_entry.path
            dir_tqdm.set_postfix(dir_postfix)
            if dir_entry.name.endswith('json'):
                with open(dir_entry.path, 'r') as jsf:
                    render_data = lib.data.render_data.from_object(json.load(jsf),  lib.data.render_data.RenderData)
                write(render_data, args.render_dir, h5f)


if __name__ == '__main__':
    main()