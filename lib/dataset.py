import io
import json
import random
import h5py
import numpy
import pytorch3d.transforms
import torch
import torchvision.transforms
from PIL import Image
import numpy as np


def pose_quat_to_T(pose_euler):
    t = pose_euler[:3, None]
    R = pytorch3d.transforms.quaternion_to_matrix(pose_euler[3:])
    T = torch.cat((R, t), dim=1)
    T = torch.cat((T, torch.from_numpy(np.array([[0., 0., 0., 1.]]))), dim=0)
    return T


def pose_quat_to_invT(pose_euler):
    t = pose_euler[:3, None]
    R = pytorch3d.transforms.quaternion_to_matrix(pose_euler[3:])
    T = torch.cat((torch.permute(R, (1, 0)), -torch.permute(R, (1, 0)) @ t), dim=1)
    T = torch.cat((T, torch.from_numpy(np.array([[0., 0., 0., 1.]]))), dim=0)
    return T


def pose_euler_to_T(pose_euler):
    t = pose_euler[:3, None]
    R = pytorch3d.transforms.euler_angles_to_matrix(pose_euler[3:], 'XYZ')
    T = torch.cat((R, t), dim=1)
    T = torch.cat((T, torch.from_numpy(np.array([[0., 0., 0., 1.]]))), dim=0)
    return T


def pose_euler_to_invT(pose_euler):
    t = pose_euler[:3, None]
    R = pytorch3d.transforms.euler_angles_to_matrix(pose_euler[3:], 'XYZ')
    T = torch.cat((torch.permute(R, (1, 0)), -torch.permute(R, (1, 0)) @ t), dim=1)
    T = torch.cat((T, torch.from_numpy(np.array([[0., 0., 0., 1.]]))), dim=0)
    return T


class BCData(torch.utils.data.Dataset):
    def __init__(self, path, indices, transform=torchvision.transforms.ToTensor()):
        super(BCData, self).__init__()
        self.path = path
        self.h5f = None

        self.renders_data = {}
        with h5py.File(path, 'r') as renders_h5_file:
            for render_name, render in renders_h5_file.items():
                self.renders_data[render_name] = json.loads(render['render_data'][()])
        self.keys = []
        for render_name, render_data in self.renders_data.items():
            for i_scene, (scene_name, scene_data) in enumerate(render_data['scenes_data'].items()):
                if i_scene == len(list(render_data['scenes_data'].items())) - 1:
                    break
                for i_camera, camera_name in enumerate(scene_data['cameras_data'].keys()):
                    self.keys.append((render_name, scene_name, camera_name))

        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.h5f is None:
            self.h5f = h5py.File(self.path, 'r')

        render_name, scene_name, camera_name = self.keys[self.indices[item]]
        next_scene_name = f'{int(scene_name)+1:06d}'

        curr_image = self.transform(Image.open(io.BytesIO(self.h5f[render_name][scene_name][camera_name]['rgba'][()])).convert('RGB'))
        next_image = self.transform(Image.open(io.BytesIO(self.h5f[render_name][next_scene_name][camera_name]['rgba'][()])).convert('RGB'))
        data = json.loads(self.h5f[render_name]['render_data'][()])['scenes_data']

        subgoal = torch.tensor((int(scene_name) + 1) // 10)

        camera_pose = torch.from_numpy(numpy.array(data[scene_name]['cameras_data'][camera_name]['pose']))
        cameraTworld = pose_euler_to_invT(camera_pose)
        action = []
        curr_poses = []
        next_poses = []
        for object_name in ['swell_cap_0', 'swell_bottle_0']:
            curr_pose = torch.from_numpy(numpy.array(data[scene_name]['objects_data'][object_name]['pose']))
            next_pose = torch.from_numpy(numpy.array(data[f'{int(scene_name) + 1:06d}']['objects_data'][object_name]['pose']))
            currTworld = pose_euler_to_invT(curr_pose)
            worldTcurr = pose_euler_to_T(curr_pose)
            worldTnext = pose_euler_to_T(next_pose)

            cameraTcurr = cameraTworld @ worldTcurr
            cameraTnext = cameraTworld @ worldTnext
            currTnext = currTworld @ worldTnext
            t = cameraTnext[:3, 3] - cameraTcurr[:3, 3]
            q = pytorch3d.transforms.matrix_to_quaternion(currTnext[:3, :3])
            q[1:] = cameraTcurr[:3, :3] @ q[1:]
            action.append(torch.concat((t, q)))
            curr_poses.append(curr_pose)
            next_poses.append(next_pose)
        action = torch.concat(action)
        curr_pose = torch.concat(curr_poses)
        next_pose = torch.concat(next_poses)

        return curr_image, next_image, subgoal, action, curr_pose, next_pose, camera_pose
