import argparse
import copy
import json
import math
import os
import random
import numpy
import lib

try:
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


def get_object_data(shape_name, scene_data: lib.data.scene_data.SceneData):
    shape_value = os.path.join(scene_data.shape_dir, f'{shape_name}.blend', 'Object', shape_name)
    shape_pair = (shape_name, shape_value)

    color_pair = random.choice([*scene_data.properties['colors'].items()])
    color_pair = (color_pair[0], numpy.append(numpy.array(color_pair[1], dtype=float) / 255, 1.0))

    material_pair = None

    scale_pair = ('scale_down', numpy.array([0.15, 0.15, 0.15]))

    x = random.uniform(-2.0, 2.0)
    y = random.uniform(-2.0, 2.0)
    z = 1.3085 * 0.75 if shape_name == 'swell_bottle' else 0.2796 * 0.75
    r_x = 0
    r_y = 0
    r_z = 0
    pose = numpy.array([x, y, z, r_x, r_y, r_z], dtype=float)

    shape_count = sum([1 for object_name, object_data in scene_data.objects_data.items()
                       if object_data.name.startswith(shape_name)])
    name = f'{shape_name}_{shape_count}'

    object_data = lib.data.object_data.ObjectData(name, shape_pair, material_pair, color_pair, scale_pair, pose)
    return object_data


def get_camera_data(d, r_x, r_y, r_z, scene_data: lib.data.scene_data.SceneData):
    name = f'cam{len(scene_data.cameras_data):02d}'

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


def get_scene_data(name, args, reset_scene) -> lib.data.scene_data.SceneData:
    scene_data = lib.data.scene_data.from_args(name, args, reset_scene)

    ground_object_data = get_ground_object_data(scene_data)
    scene_data.objects_data[ground_object_data.name] = ground_object_data

    cap_data = get_object_data('swell_cap', scene_data)
    scene_data.objects_data[cap_data.name] = cap_data
    bottle_data = get_object_data('swell_bottle', scene_data)
    scene_data.objects_data[bottle_data.name] = bottle_data

    for r_x in numpy.linspace(60, 0, 3):
        for r_z in numpy.linspace(0, 300, 6):
            camera_data = get_camera_data(10, r_x, 0, r_z, scene_data)
            scene_data.cameras_data[camera_data.name] = camera_data

    light_data = get_light_data(scene_data)
    scene_data.lights_data[light_data.name] = light_data

    return scene_data


def get_render_data(name, args) -> lib.data.render_data.RenderData:
    render_data = lib.data.render_data.from_args(name, args)
    for i_scene in range(3 * args.num_scenes):
        i_stage = i_scene // args.num_scenes
        if i_scene == 0:
            scene_data = get_scene_data(f'{i_scene:06d}', args, True)
            render_data.scenes_data[scene_data.name] = scene_data

            bottle_pose = scene_data.objects_data['swell_bottle_0'].pose
            cap_goal_pose = numpy.array([[bottle_pose[0], bottle_pose[1], (1.3085 * 1.5) * random.uniform(1.1, 1.5), 0, 0, 0],
                                         [bottle_pose[0], bottle_pose[1], 1.3085 * 1.5, 0, 0, 0]])
        elif i_stage < 2:
            scene_data = copy.deepcopy(scene_data)
            scene_data.name = f'{i_scene:06d}'
            scene_data.reset_scene = False
            cap_pose_diff = cap_goal_pose[i_stage] - scene_data.objects_data['swell_cap_0'].pose
            scene_data.objects_data['swell_cap_0'].pose += cap_pose_diff / (args.num_scenes - i_scene % args.num_scenes)
            render_data.scenes_data[scene_data.name] = scene_data
        else:
            scene_data = copy.deepcopy(scene_data)
            scene_data.name = f'{i_scene:06d}'
            scene_data.reset_scene = False
            euler = mathutils.Euler((math.radians(0), math.radians(0), math.radians(-10)))
            euler.rotate(mathutils.Euler((math.radians(scene_data.objects_data['swell_cap_0'].pose[3]),
                                          math.radians(scene_data.objects_data['swell_cap_0'].pose[4]),
                                          math.radians(scene_data.objects_data['swell_cap_0'].pose[5]))))
            scene_data.objects_data['swell_cap_0'].pose[3:] = numpy.array([math.degrees(euler.x),
                                                                           math.degrees(euler.y),
                                                                           math.degrees(euler.z)])
            scene_data.objects_data['swell_cap_0'].pose[2] -= 0.01
            render_data.scenes_data[scene_data.name] = scene_data

    return render_data


def main():
    parser = argparse.ArgumentParser()
    # scene
    parser.add_argument('--base_scene_blendfile', default=None)

    # object
    parser.add_argument('--properties_json', default='data/properties/cap_properties.json')
    parser.add_argument('--shape_dir', default='data/shapes')
    parser.add_argument('--material_dir', default='data/materials')

    # output
    parser.add_argument('--num_renders', default=1, type=int)
    parser.add_argument('--num_scenes', default=10, type=int)
    parser.add_argument('--output_dir', default='./output/caps_onlycap/')
    parser.add_argument('--save_blend', default=False, type=bool)
    parser.add_argument('--device_type', default='OPTIX', type=str, choices=('CPU', 'CUDA', 'OPTIX'))
    parser.add_argument('--modes', default=('rgba', 'nocs', 'depth'), type=int, nargs='+')

    # image
    parser.add_argument('--width', default=480, type=int)
    parser.add_argument('--height', default=320, type=int)
    parser.add_argument('--render_num_samples', default=512, type=int)
    parser.add_argument('--render_min_bounces', default=8, type=int)
    parser.add_argument('--render_max_bounces', default=8, type=int)
    parser.add_argument('--render_tile_size', default=256, type=int)
    args = parser.parse_args()

    for render_i in range(args.num_renders):
        render_data = get_render_data(f'{render_i:06d}', args)

        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
        with open(os.path.join(render_data.output_dir, f'{render_data.name}.json'), 'w') as f:
            json.dump(lib.data.render_data.to_object(render_data), f, indent=2)


if __name__ == '__main__':
    main()
