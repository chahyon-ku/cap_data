import argparse
import json
import lib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_data_path', type=str, default='output/cap.json')
    args = parser.parse_args()

    with open(args.render_data_path, 'r') as f:
        render_data = lib.data.render_data.from_object(json.load(f), lib.data.render_data.RenderData)

    lib.blend.blend_render(render_data)