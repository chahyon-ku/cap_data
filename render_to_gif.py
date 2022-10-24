import argparse
import os
import imageio
import numpy
from pygifsicle import pygifsicle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_dir', type=str, default='output/caps/000000')
    args = parser.parse_args()

    camera_mode_frames = {}
    for dir_entry in os.scandir(args.render_dir):
        if dir_entry.name.endswith('png'):
            i_frame, camera, mode = dir_entry.name.split('_')
            mode = mode.split('.')[0]
            camera_mode = f'{camera}_{mode}'
            if camera_mode not in camera_mode_frames:
                camera_mode_frames[camera_mode] = []

            if mode.startswith('depth'):
                bgra = numpy.array(imageio.v3.imread(dir_entry.path), dtype=float)
                depth = 2.0 ** (bgra[:, :, 2] - 128) * (bgra[:, :, 1] / 256 + bgra[:, :, 0] / 256 ** 2 + bgra[:, :, 3] / 256 ** 3)
                normalized_depth = numpy.clip((depth - numpy.min(depth)) / (numpy.max(depth) - numpy.min(depth)) * 255, 0, 255).astype(numpy.uint8)
                camera_mode_frames[camera_mode].append(normalized_depth)
            else:
                im = imageio.v3.imread(dir_entry.path)
                camera_mode_frames[camera_mode].append(im)

    all_frames = []
    for camera_mode, frames in sorted(camera_mode_frames.items(), key=lambda item: item[0]):
        for frame in frames:
            if len(frame.shape) < 3:
                all_frames.append(numpy.stack(4 * [frame], -1))
            else:
                all_frames.append(frame)

    frames = numpy.concatenate(all_frames)
    frames = numpy.reshape(frames, (3, 6, 3, 10, 320, 480, 4))
    frames = numpy.transpose(frames, (3, 2, 0, 1, 4, 5, 6))
    frames = numpy.reshape(frames, (10, 9, 6, 320, 480, 4))
    frames = numpy.transpose(frames, (0, 1, 3, 2, 4, 5))
    frames = numpy.reshape(frames, (10, 2880, 2880, 4))
    frames = [frame for frame_i, frame in enumerate(frames)]
    gif_path = os.path.join(args.render_dir, f'all.gif')
    imageio.mimsave(gif_path, frames, duration=0.25)
    # pygifsicle.optimize(gif_path)

    for camera_mode, frames in camera_mode_frames.items():
        gif_path = os.path.join(args.render_dir, f'{camera_mode}.gif')
        imageio.mimsave(gif_path, frames, duration=0.25)
        # pygifsicle.optimize(gif_path)
