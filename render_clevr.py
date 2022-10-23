import argparse
import math
import os
import random

import numpy

import arguments
import lib

try:
    import bpy, bpy_extras
    import mathutils
except ImportError as e:
    print(e)


def get_object_data(scene_data: lib.data.scene_data.SceneData):
    shape_name = random.choice([*scene_data.properties['shapes'].items()])[0]
    shape_value = os.path.join(scene_data.shape_dir, f'{shape_name}.blend', 'Object', shape_name)
    shape_pair = (shape_name, shape_value)

    color_pair = random.choice([*scene_data.properties['colors'].items()])
    color_pair = (color_pair[0], numpy.append(numpy.array(color_pair[1], dtype=float) / 255, 1.0))

    material_pair = random.choice([*scene_data.properties['materials'].items()])

    scale_pair = random.choice([*scene_data.properties['sizes'].items()])
    scale_pair = (scale_pair[0], numpy.array(scale_pair[1], dtype=float))

    x = random.uniform(-2.2, 2.2)
    y = random.uniform(-2.2, 2.2)
    z = bpy.context.view_layer.objects.active.dimensions[2] / 2
    r_x = 0
    r_y = 0
    r_z = random.uniform(0, 360)
    pose = numpy.array([x, y, z, r_x, r_y, r_z], dtype=float)

    shape_count = sum([1 for object_data in scene_data.objects_data if object_data.name.startswith(shape_name)])
    name = f'{shape_name}_{shape_count}'

    object_data = lib.data.object_data.ObjectData(name, shape_pair, material_pair, color_pair, scale_pair, pose)
    return object_data


def get_scene_data(args) -> lib.data.scene_data.SceneData:
    scene_data = lib.data.scene_data.SceneData(args)
    object_data = get_object_data(scene_data)
    scene_data.objects_data[object_data.name] = object_data

    return scene_data


def get_render_data(dir, name, args) -> lib.data.render_data.RenderData:
    render_data = lib.data.render_data.RenderData(dir, name, args)
    return render_data


def main():
    parser = argparse.ArgumentParser()
    arguments.add_input_arguments(parser)
    arguments.add_output_arguments(parser)
    arguments.add_render_arguments(parser)
    args = parser.parse_args()

    for scene_i in range(args.start_idx, args.start_idx + args.num_scenes):
        render_data = get_render_data(args.output_dir, f'{scene_i:06d}', args)
        scene_data = get_scene_data(args)
        lib.blend.blend_scene(scene_data)
        bpy.context.view_layer.update()

        blend_path = os.path.join(os.path.abspath(args.output_dir), f'{scene_i:06d}.blend')
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)

        lib.blend.blend_render(render_data)


if __name__ == '__main__':
    main()
