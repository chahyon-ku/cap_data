import lib.data
import lib.blend_nocs
import os
import numpy as np
import cv2
import OpenEXR

try:
    import bpy
    from mathutils import Vector
except ImportError as e:
    print(e)


def blend_render(render_data: lib.data.render_data.RenderData):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = render_data.width
    bpy.context.scene.render.resolution_y = render_data.height
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.tile_x = render_data.render_tile_size
    bpy.context.scene.render.tile_y = render_data.render_tile_size

    if render_data.device_type == 'CPU':
        bpy.context.scene.cycles.device = 'CPU'
    elif render_data.device_type == 'CUDA':
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.context.scene.cycles.device = 'GPU'
    elif render_data.device_type == 'OPTIX':
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'
        bpy.context.scene.cycles.device = 'GPU'

    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.transparent_min_bounces = render_data.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_data.render_max_bounces

    for mode in render_data.modes:
        if mode == 'rgba':
            bpy.context.scene.cycles.samples = render_data.render_num_samples
            bpy.context.scene.render.image_settings.file_format = 'PNG'
        elif mode == 'nocs':
            bpy.context.scene.cycles.samples = 1
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            lib.blend_nocs.blend_nocs()
        elif mode == 'depth':
            bpy.context.scene.cycles.samples = 1
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
            bpy.context.scene.render.image_settings.use_zbuffer = True

        for ob in bpy.context.scene.objects:
            if ob.type == "CAMERA":
                bpy.context.scene.camera = ob
                os.makedirs(render_data.dir, exist_ok=True)
                bpy.context.scene.render.filepath = f'{render_data.dir}/{render_data.name}_{ob.name}_{mode}'
                bpy.ops.render.render(write_still=True, use_viewport=True)

                if mode == 'depth':
                    f = OpenEXR.InputFile(bpy.context.scene.render.filepath + '.exr')
                    zs = np.frombuffer(f.channel('Z'), np.float32).reshape(render_data.height, render_data.width)
                    f.close()
                    os.remove(bpy.context.scene.render.filepath + '.exr')

                    r = np.ceil(np.log2(zs))
                    g = zs / np.power(r, 2)
                    b = g * 256 - np.trunc(g * 256)
                    a = b * 256 - np.trunc(b * 256)
                    rgba = np.stack([r + 128, g * 256, b * 256, a * 256], axis=-1).astype(np.uint8)

                    cv2.imwrite(bpy.context.scene.render.filepath + '.png', rgba)


def blend_object(object_data: lib.data.object_data.ObjectData):
    bpy.ops.wm.append(filename=object_data.shape_pair[1])
    bpy.data.objects[object_data.shape_pair[0]].name = object_data.name

    bpy.data.objects[object_data.name].rotation_euler = object_data.pose[3:] * np.pi / 180
    bpy.data.objects[object_data.name].location = object_data.pose[:3]
    bpy.data.objects[object_data.name].scale = np.array(bpy.data.objects[object_data.name].scale) \
                                               * object_data.scale_pair[1]

    if object_data.color_pair is None:
        material_name = object_data.material_pair[1]
    else:
        material_name = f'{object_data.material_pair[1]}_{object_data.color_pair[0]}'
    if material_name not in bpy.data.materials:
        material = bpy.data.materials.new(material_name)
        material.use_nodes = True
        group_node = material.node_tree.nodes.new('ShaderNodeGroup')
        group_node.node_tree = bpy.data.node_groups[object_data.material_pair[1]]
        group_node.inputs['Color'].default_value = object_data.color_pair[1]
        material.node_tree.links.new(group_node.outputs['Shader'],
                                     material.node_tree.nodes['Material Output'].inputs['Surface'])

    bpy.data.objects[object_data.name].data.materials.clear()
    bpy.data.objects[object_data.name].data.materials.append(bpy.data.materials[material_name])


def blend_camera(camera_data: lib.data.camera_data.CameraData):
    pass


def blend_scene(scene_data: lib.data.scene_data.SceneData):
    bpy.ops.wm.open_mainfile(filepath=scene_data.base_scene_blendfile)

    for dir_entry in os.scandir(scene_data.material_dir):
        if dir_entry.name.endswith('.blend'):
            material_path = os.path.join(dir_entry.path, 'NodeTree', os.path.splitext(dir_entry.name)[0])
            bpy.ops.wm.append(filename=material_path)

    for object_name, object_data in scene_data.objects_data.items():
        blend_object(object_data)

    for camera_name, camera_data in scene_data.cameras_data.items():
        blend_camera(camera_data)