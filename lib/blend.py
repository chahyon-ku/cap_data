import shutil

import numpy

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
    os.makedirs(f'{render_data.output_dir}/{render_data.name}', exist_ok=True)
    shutil.rmtree(f'{render_data.output_dir}/{render_data.name}')
    os.makedirs(f'{render_data.output_dir}/{render_data.name}', exist_ok=True)
    for scene_name, scene_data in render_data.scenes_data.items():
        blend_scene(scene_data)

        if render_data.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=f'{render_data.output_dir}/{render_data.name}/{scene_name}.blend')

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
                    os.makedirs(render_data.output_dir, exist_ok=True)
                    bpy.context.scene.render.filepath = f'{render_data.output_dir}/{render_data.name}/{scene_name}_{ob.name}_{mode}'
                    bpy.ops.render.render(write_still=True, use_viewport=True)

                    if mode == 'depth':
                        f = OpenEXR.InputFile(bpy.context.scene.render.filepath + '.exr')
                        zs = np.frombuffer(f.channel('Z'), np.float32).reshape(render_data.height, render_data.width)
                        f.close()
                        os.remove(bpy.context.scene.render.filepath + '.exr')

                        r = np.ceil(np.log2(zs))
                        g = zs / np.power(2, r)
                        b = g * 256 - np.trunc(g * 256)
                        a = b * 256 - np.trunc(b * 256)
                        rgba = np.stack([r + 128, g * 256, b * 256, a * 256], axis=-1).astype(np.uint8)

                        cv2.imwrite(bpy.context.scene.render.filepath + '.png', rgba)


def blend_object(object_data: lib.data.object_data.ObjectData):
    if object_data.shape_pair[0] == 'plane':
        bpy.ops.mesh.primitive_plane_add(size=1000.0)
        bpy.data.objects['Plane'].name = object_data.name
    else:
        bpy.ops.wm.append(filename=object_data.shape_pair[1])
        bpy.data.objects[object_data.shape_pair[0]].name = object_data.name
    obj = bpy.data.objects[object_data.name]

    if object_data.material_pair is not None:
        material_name = f'{object_data.material_pair[1]}_{object_data.color_pair[0]}'
        if material_name not in bpy.data.materials:
            if material_name.startswith('solid'):
                material = bpy.data.materials.new(material_name)
                material.use_nodes = True
            else:
                material = bpy.data.materials.new(material_name)
                material.use_nodes = True
                group_node = material.node_tree.nodes.new('ShaderNodeGroup')
                group_node.node_tree = bpy.data.node_groups[object_data.material_pair[1]]
                group_node.inputs['Color'].default_value = object_data.color_pair[1]
                material.node_tree.links.new(group_node.outputs['Shader'],
                                             material.node_tree.nodes['Material Output'].inputs['Surface'])

        obj.data.materials.clear()
        obj.data.materials.append(bpy.data.materials[material_name])

    obj.location = object_data.pose[:3]
    obj.rotation_euler = numpy.radians(object_data.pose[3:])
    obj.scale = np.array(obj.scale) * object_data.scale_pair[1]
    # min_z = 999999
    # for polygon_index, polygon in enumerate(obj.data.polygons):
    #     for vertex_index, vertex in enumerate(polygon.vertices):
    #         z = obj.data.vertices[vertex].co[2] * obj.scale[2]
    #         min_z = min(min_z, z)
    # obj.location[2] = obj.location[2] - min_z


def blend_camera(camera_data: lib.data.camera_data.CameraData):
    camera = bpy.data.cameras.new(camera_data.name)
    object = bpy.data.objects.new(camera_data.name, camera)
    object.location = camera_data.pose[:3]
    object.rotation_euler = numpy.radians(camera_data.pose[3:])
    bpy.context.scene.collection.objects.link(object)


def blend_light(light_data: lib.data.light_data.LightData):
    light = bpy.data.lights.new(light_data.name, light_data.type)
    light.energy = light_data.energy
    object = bpy.data.objects.new(light_data.name, light)
    object.location = light_data.pose[:3]
    object.rotation_euler = numpy.radians(light_data.pose[3:])
    bpy.context.scene.collection.objects.link(object)


def blend_scene(scene_data: lib.data.scene_data.SceneData):
    if scene_data.base_scene_blendfile is None:
        bpy.ops.wm.read_factory_settings()
        # bpy.data.scenes.new('scene')
        for object_name, object_value in bpy.data.objects.items():
            bpy.data.objects.remove(object_value)
    else:
        bpy.ops.wm.open_mainfile(filepath=scene_data.base_scene_blendfile)

    for dir_entry in os.scandir(scene_data.material_dir):
        if dir_entry.name.endswith('.blend'):
            material_path = os.path.join(dir_entry.path, 'NodeTree', os.path.splitext(dir_entry.name)[0])
            bpy.ops.wm.append(filename=material_path)

    for object_name, object_data in scene_data.objects_data.items():
        blend_object(object_data)

    for camera_name, camera_data in scene_data.cameras_data.items():
        blend_camera(camera_data)

    for light_name, light_data in scene_data.lights_data.items():
        blend_light(light_data)

