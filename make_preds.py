import argparse
import collections

import numpy as np
import pytorch3d.transforms
import torch
import tqdm
import json
import math
import os
import random

from matplotlib import pyplot as plt

import lib
import torchvision
import cv2

try:
    import mathutils
except ImportError as e:
    print(e)


def get_ground_object_data(scene_data: lib.data.scene_data.SceneData):
    shape_pair = ('plane', 'plane')
    color_pair = ('white', np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    material_pair = ('solid', 'solid')
    scale_pair = ('1', np.array((1, 1, 1), dtype=float))
    pose = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    name = f'ground'

    object_data = lib.data.object_data.ObjectData(name, shape_pair, material_pair, color_pair, scale_pair, pose)
    return object_data


def get_object_data(shape_name, scene_data: lib.data.scene_data.SceneData):
    shape_value = os.path.join(scene_data.shape_dir, f'{shape_name}.blend', 'Object', shape_name)
    shape_pair = (shape_name, shape_value)

    color_pair = random.choice([*scene_data.properties['colors'].items()])
    color_pair = (color_pair[0], np.append(np.array(color_pair[1], dtype=float) / 255, 1.0))

    material_pair = None

    scale_pair = ('scale_down', np.array([0.10, 0.10, 0.10]))

    x = random.uniform(-1.5, 1.5)
    y = random.uniform(-1,5, 1.5)
    z = 1.3085 * 0.5 if shape_name == 'swell_bottle' else 0.2796 * 0.5
    r_x = 0
    r_y = 0
    r_z = 0
    pose = np.array([x, y, z, r_x, r_y, r_z], dtype=float)

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
    pose = np.array([pos[0], pos[1], pos[2], r_x, r_y, r_z], dtype=float)

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
    pose = np.array([pos[0], pos[1], pos[2], r_x, r_y, r_z], dtype=float)

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

    for r_x in np.linspace(60, 0, 3):
        for r_z in np.linspace(0, 300, 6):
            camera_data = get_camera_data(10, r_x, 0, r_z, scene_data)
            scene_data.cameras_data[camera_data.name] = camera_data

    light_data = get_light_data(scene_data)
    scene_data.lights_data[light_data.name] = light_data

    return scene_data


def get_render_data(name, args) -> lib.data.render_data.RenderData:

    with open(args.preds_path, 'r') as f:
        preds = json.load(f)
    os.makedirs(args.preds_dir, exist_ok=True)

    valid_indices = list(range(1, 10 * 29 * 18, 5)) + list(range(3, 10 * 29 * 18, 5))
    data = lib.dataset.BCData(args.dataset_path, valid_indices, torchvision.transforms.ToTensor())
    postfix = collections.OrderedDict()
    data_tqdm = tqdm.tqdm(enumerate(data), total=len(data), leave=False)
    for i_train, (curr_image, next_image, subgoal, action, curr_pose, next_pose, camera_pose) in data_tqdm:
        plt.gcf().set_size_inches(12, 9)

        curr_image = curr_image.numpy()
        curr_image = np.transpose(curr_image, (1, 2, 0))
        next_image = next_image.numpy()
        next_image = np.transpose(next_image, (1, 2, 0))
        curr_image_subplot = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=1)
        curr_image_subplot.imshow(curr_image)
        curr_image_subplot.axis('off')
        next_image_subplot = plt.subplot2grid((2, 3), (0, 1), rowspan=1, colspan=1)
        next_image_subplot.imshow(next_image)
        next_image_subplot.axis('off')

        worldTcamera = lib.dataset.pose_euler_to_T(camera_pose)
        cameraTworld = lib.dataset.pose_euler_to_invT(camera_pose)
        worldTcurr = lib.dataset.pose_euler_to_T(curr_pose[:6])
        cameraTcurr = cameraTworld @ worldTcurr
        cameratnext = np.ones(4)
        cameratnext[:3] = action[:3] + cameraTcurr[:3, 3]
        worldtnext = worldTcamera @ cameratnext
        currqnext = action[3:7]
        currqnext[1:] = torch.transpose(cameraTcurr[:3, :3], 1, 0) @ currqnext[1:]
        currRnext = pytorch3d.transforms.quaternion_to_matrix(currqnext)
        worldRnext = worldTcurr[:3, :3] @ currRnext
        next_inferred_pose = torch.concat([worldtnext[:3], pytorch3d.transforms.matrix_to_euler_angles(worldRnext, 'XYZ')], 0)

        subgoal_item = subgoal.item()
        action_list = action.tolist()
        action_list = [round(a, 3) for a in action_list]
        curr_pose_list = curr_pose.tolist()
        curr_pose_list = [round(a, 3) for a in curr_pose_list]
        next_pose_list = next_pose.tolist()
        next_pose_list = [round(a, 3) for a in next_pose_list]
        next_inferred_pose = next_inferred_pose.tolist()
        next_inferred_pose = [round(a, 3) for a in next_inferred_pose]
        next_label_subplot = plt.subplot2grid((2, 3), (0, 2), rowspan=1, colspan=1)
        next_label_subplot.text(0, 1.0, f'subgoal: {subgoal_item}')
        next_label_subplot.text(0, 0.9, f'curr_pose1: {curr_pose_list[:7]}')
        next_label_subplot.text(0, 0.8, f'curr_pose2: {curr_pose_list[7:]}')
        next_label_subplot.text(0, 0.7, f'action1: {action_list[:7]}')
        next_label_subplot.text(0, 0.6, f'action2: {action_list[7:]}')
        next_label_subplot.text(0, 0.5, f'next_pose1: {next_pose_list[:6]}')
        next_label_subplot.text(0, 0.4, f'next_pose2: {next_pose_list[6:]}')
        next_label_subplot.text(0, 0.3, f'next_inferred_pose1: {next_inferred_pose}')
        next_label_subplot.axis('off')

        pred_image_subplot = plt.subplot2grid((2, 3), (1, 1), rowspan=1, colspan=1)
        pred_image_subplot.axis('off')

        subgoal_pred = preds['subgoal'][i_train]
        action_pred = preds['action'][i_train]
        action_pred = [round(a, 3) for a in action_pred]

        worldTcamera = lib.dataset.pose_euler_to_T(camera_pose)
        cameraTworld = lib.dataset.pose_euler_to_invT(camera_pose)
        worldTcurr = lib.dataset.pose_euler_to_T(curr_pose[:6])
        currTnext = cameraTworld @ worldTcurr
        cameraTnext = cameraTcurr @ currTnext
        cameratnext[:3] = torch.as_tensor(action_pred[:3]).double() + cameraTcurr[:3, 3]
        worldTnext = worldTcamera @ cameraTnext
        next_inferred_pose = torch.concat([worldTnext[:3], pytorch3d.transforms.matrix_to_euler_angles(worldTnext[:3, :3], 'XYZ')], 0)

        next_inferred_pose = next_inferred_pose.tolist()
        next_inferred_pose = [round(a, 3) for a in next_inferred_pose]
        pred_label_subplot = plt.subplot2grid((2, 3), (1, 2), rowspan=1, colspan=1)
        pred_label_subplot.text(0, 1.0, f'subgoal_pred: {subgoal_pred}')
        pred_label_subplot.text(0, 0.9, f'action_pred1: {action_pred[:7]}')
        pred_label_subplot.text(0, 0.8, f'action_pred2: {action_pred[7:]}')
        pred_label_subplot.text(0, 0.7, f'next_inferred_pose1: {next_inferred_pose}')
        pred_label_subplot.axis('off')

        plt.savefig(os.path.join(args.preds_dir, f'{i_train:02d}.png'), bbox_inches='tight')
        plt.close()
        # plt.show()

    render_data = None

    return render_data


def main():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset_path', type=str, default='data/caps_onlycap_small.h5')
    parser.add_argument('--preds_path', type=str, default='data/full_gt_onlycap_small_159.json')
    parser.add_argument('--preds_dir', type=str, default='preds/caps_onlycap_small')

    # scene
    parser.add_argument('--base_scene_blendfile', default=None)

    # object
    parser.add_argument('--properties_json', default='data/properties/cap_properties.json')
    parser.add_argument('--shape_dir', default='data/shapes')
    parser.add_argument('--material_dir', default='data/materials')

    # output
    parser.add_argument('--num_renders', default=10, type=int)
    parser.add_argument('--num_scenes', default=10, type=int)
    parser.add_argument('--output_dir', default='./output/caps_onlycap_small/')
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
