import argparse
import io
import json
import os
import tqdm
import numpy
from PIL import Image
import h5py
import lib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_json', type=str, default='./output/clevr/clevr.json')
    parser.add_argument('--output_h5', type=str, default='./output/clevr/clevr.h5')
    args = parser.parse_args()

    with open(args.render_json, 'r') as jsf:
        render_data = lib.data.render_data.from_object(json.load(jsf),  lib.data.render_data.RenderData)

    with h5py.File(args.output_h5, 'w') as h5f:
        h5f.create_dataset('render_data', data=json.dumps(lib.data.render_data.to_object(render_data)))
        for scene_name, scene_data in tqdm.tqdm(render_data.scenes_data.items()):
            scene_group = h5f.create_group(scene_name)
            for camera_name in scene_data.cameras_data:
                camera_group = scene_group.create_group(camera_name)
                for mode in render_data.modes:
                    png_path = os.path.join(render_data.output_dir, render_data.name, f'{scene_name}_{camera_name}_{mode}.png')
                    with Image.open(png_path) as imf:
                        buf = io.BytesIO()
                        imf.save(buf, 'png')
                    camera_group.create_dataset(mode, data=numpy.array(buf.getvalue()))
