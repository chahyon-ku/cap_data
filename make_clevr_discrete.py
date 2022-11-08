import argparse
import json
import math
import os
import random
import numpy
import lib

try:
    import bpy, bpy_extras
    import mathutils
except ImportError as e:
    print(e)


def get_ground_object_data(scene_data: lib.data.scene_data.SceneData):
    shape_pair = ('plane', 'plane')
    color_pair = ('white', numpy.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    material_pair = ('solid', 'solid')
    scale_pair = ('1', numpy.array((1, 1, 1), dtype=float))
    pose = numpy.array([0, 0, 0, 0, 0, 0], dtype=float)
    name = f'ground'

    object_data = lib.data.object_data.ObjectData(name, shape_pair, material_pair, color_pair, scale_pair, pose)
    return object_data


def get_object_data(scene_data: lib.data.scene_data.SceneData, rotation):
    shape_name = random.choice([*scene_data.properties['shapes'].items()])[0]
    shape_value = os.path.join(scene_data.shape_dir, f'{shape_name}.blend', 'Object', shape_name)
    shape_pair = (shape_name, shape_value)

    color_pair = random.choice([*scene_data.properties['colors'].items()])
    color_pair = (color_pair[0], numpy.append(numpy.array(color_pair[1], dtype=float) / 255, 1.0))

    material_pair = random.choice([*scene_data.properties['materials'].items()])

    scale_pair = random.choice([*scene_data.properties['sizes'].items()])
    scale_pair = (scale_pair[0], numpy.array(scale_pair[1], dtype=float))

    pose_range = numpy.array(scene_data.properties['pose_range'])
    transform = numpy.random.uniform(pose_range[:3, 0], pose_range[:3, 1])
    pose = numpy.concatenate((transform, rotation))

    shape_count = sum([1 for object_name, object_data in scene_data.objects_data.items()
                       if object_data.name.startswith(shape_name)])
    name = f'{shape_name}_{shape_count}'

    object_data = lib.data.object_data.ObjectData(name, shape_pair, material_pair, color_pair, scale_pair, pose)
    return object_data


def get_camera_data(scene_data: lib.data.scene_data.SceneData):
    name = f'cam{len(scene_data.cameras_data):02d}'

    d = 8
    r_x = 60
    r_y = 0
    r_z = 0
    euler = mathutils.Euler((math.radians(r_x), math.radians(r_y), math.radians(r_z)), 'XYZ')
    pos = mathutils.Vector((0, 0, d))
    pos.rotate(euler)
    pose = numpy.array([pos[0], pos[1], pos[2], r_x, r_y, r_z], dtype=float)

    camera_data = lib.data.camera_data.CameraData(name, pose)
    return camera_data


def get_light_data(scene_data: lib.data.scene_data.SceneData):
    name = 'light_0'
    type = 'POINT'
    energy = 1000.0

    d = 10
    r_x = 45
    r_y = 0
    r_z = 45
    euler = mathutils.Euler((math.radians(r_x), math.radians(r_y), math.radians(r_z)))
    pos = mathutils.Vector((0, 0, d))
    pos.rotate(euler)
    pose = numpy.array([pos[0], pos[1], pos[2], r_x, r_y, r_z], dtype=float)

    light_data = lib.data.light_data.LightData(name, type, energy, pose)
    return light_data


def get_scene_data(name, args, reset_scene, rotation) -> lib.data.scene_data.SceneData:
    scene_data = lib.data.scene_data.from_args(name, args, reset_scene)

    ground_object_data = get_ground_object_data(scene_data)
    scene_data.objects_data[ground_object_data.name] = ground_object_data

    object_data = get_object_data(scene_data, rotation)
    scene_data.objects_data[object_data.name] = object_data

    camera_data = get_camera_data(scene_data)
    scene_data.cameras_data[camera_data.name] = camera_data

    light_data = get_light_data(scene_data)
    scene_data.lights_data[light_data.name] = light_data

    return scene_data


def get_render_data(name, args) -> lib.data.render_data.RenderData:
    render_data = lib.data.render_data.from_args(name, args)

    n_groups = 4
    groups = 360 * numpy.arange(n_groups) / n_groups
    rotations = numpy.reshape(numpy.stack(numpy.meshgrid(groups, groups, groups), -1), (-1, 3))
    print(rotations)
    for scene_i, rotation in enumerate(rotations):
        scene_data = get_scene_data(f'{scene_i:06d}', args, scene_i == 0, rotation)
        render_data.scenes_data[scene_data.name] = scene_data
    return render_data


def main():
    parser = argparse.ArgumentParser()
    # scene
    parser.add_argument('--base_scene_blendfile', default=None)

    # object
    parser.add_argument('--properties_json', default='data/properties/bunny_easy_properties.json')
    parser.add_argument('--shape_dir', default='data/shapes')
    parser.add_argument('--material_dir', default='data/materials')

    # output
    parser.add_argument('--num_scenes', default=10, type=int)
    parser.add_argument('--save_blend', default=False, type=bool)
    parser.add_argument('--output_dir', default='./output/clevr_discrete_90/')
    parser.add_argument('--render_name', default='render', type=str)
    parser.add_argument('--device_type', default='OPTIX', type=str, choices=('CPU', 'CUDA', 'OPTIX'))

    # image
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--height', default=256, type=int)
    parser.add_argument('--render_num_samples', default=512, type=int)
    parser.add_argument('--render_min_bounces', default=8, type=int)
    parser.add_argument('--render_max_bounces', default=8, type=int)
    parser.add_argument('--render_tile_size', default=256, type=int)
    parser.add_argument('--modes', default=('rgba', 'nocs', 'depth'), type=int, nargs='+')
    args = parser.parse_args()

    render_data = get_render_data('clevr', args)

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    with open(os.path.join(render_data.output_dir, f'{render_data.name}.json'), 'w') as f:
        json.dump(lib.data.render_data.to_object(render_data), f, indent=2)


if __name__ == '__main__':
    main()