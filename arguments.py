import argparse


def add_input_arguments(parser: argparse.ArgumentParser):
    # Input options
    parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
                        help="Base blender file on which all scenes are based; includes " +
                             "ground plane, lights, and camera.")
    parser.add_argument('--properties_json', default='data/properties/mug_properties.json',
                        help="JSON file defining objects, materials, sizes, and colors. " +
                             "The \"colors\" field maps from CLEVR color names to RGB values; " +
                             "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                             "rescale object models; the \"materials\" and \"shapes\" fields map " +
                             "from CLEVR material and shape names to .blend files in the " +
                             "--object_material_dir and --shape_dir directories respectively.")
    parser.add_argument('--shape_dir', default='data/shapes',
                        help="Directory where .blend files for object models are stored")
    parser.add_argument('--material_dir', default='data/materials',
                        help="Directory where .blend files for materials are stored")


def add_output_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--start_idx', default=0, type=int,
                        help="The index at which to start for numbering rendered images. Setting " +
                             "this to non-zero values allows you to distribute rendering across " +
                             "multiple machines and recombine the results later.")
    parser.add_argument('--num_scenes', default=1, type=int,
                        help="The number of scenes to render")
    parser.add_argument('--output_dir', default='./output/test/',
                        help="The directory where output images will be stored. It will be " +
                             "created if it does not exist.")


def add_render_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--device_type', default='OPTIX', type=str, choices=('CPU', 'CUDA', 'OPTIX'),
                        help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                             "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                             "to work.")
    parser.add_argument('--width', default=480, type=int,
                        help="The width (in pixels) for the rendered images")
    parser.add_argument('--height', default=320, type=int,
                        help="The height (in pixels) for the rendered images")
    parser.add_argument('--render_num_samples', default=512, type=int,
                        help="The number of samples to use when rendering. Larger values will " +
                             "result in nicer images but will cause rendering to take longer.")
    parser.add_argument('--render_min_bounces', default=8, type=int,
                        help="The minimum number of bounces to use for rendering.")
    parser.add_argument('--render_max_bounces', default=8, type=int,
                        help="The maximum number of bounces to use for rendering.")
    parser.add_argument('--render_tile_size', default=256, type=int,
                        help="The tile size to use for rendering. This should not affect the " +
                             "quality of the rendered image but may affect the speed; CPU-based " +
                             "rendering may achieve better performance using smaller tile sizes " +
                             "while larger tile sizes may be optimal for GPU-based rendering.")
    parser.add_argument('--modes', default=('rgba', 'nocs', 'depth'), type=int, nargs='+',)