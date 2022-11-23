import matplotlib
import numpy

try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError as e:
    print(e)


def get_emission_material(color=(0, 0, 0)):
    emission_count = sum([1 for material_name in bpy.data.materials.keys() if material_name.startswith('emission_material')])
    emission_material = bpy.data.materials.new(f'emission_material_{emission_count}')
    emission_material.use_nodes = True

    shader_node_emission = emission_material.node_tree.nodes.new('ShaderNodeEmission')
    shader_node_emission.inputs['Color'].default_value = color
    material_output = emission_material.node_tree.nodes.get('Material Output')

    emission_material.node_tree.links.new(shader_node_emission.outputs['Emission'], material_output.inputs['Surface'])
    return emission_material


def blend_segmentation():
    colors = matplotlib.colors.to_rgba_array(matplotlib.colors.TABLEAU_COLORS)
    print(colors)

    for i_obj, obj in enumerate(bpy.data.objects):
        if obj.type == 'MESH':
            obj.data.materials.clear()

            emission_material = get_emission_material(colors[i_obj])
            obj.data.materials.append(emission_material)
            obj.active_material = emission_material
