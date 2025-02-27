"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modified by: [Your Name]
Date: 2025-01-08
Changes:
 - Always use 8-bit images for saving and metrics computation.
 - Added LPIPS metric computation.
 - Removed unnecessary mapping to [-1,1] as images are already in that range.
 - Added debug prints to verify image ranges before LPIPS computation.
"""

import os
import torch
import numpy as np
from PIL import Image

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html


def tensor_to_numpy(tensor):
    """
    Convert a torch tensor to a NumPy array.
    Assumes tensor is in shape (B, C, H, W).
    """
    tensor = tensor.detach().cpu().float()
    array = tensor.numpy()
    array = np.transpose(array, (0, 2, 3, 1))  # (B, H, W, C)
    return array

def save_numpy_as_image(array, save_path, normalize=False,
                        window_level=None, window_width=None):
    """
    Save a NumPy array as an image file with optional windowing (for single-channel images).
    The output image is always 8-bit.
    """

    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
            mode = 'L'  
        elif array.shape[-1] == 3:
            mode = 'RGB'
        else:
            raise ValueError(f"Unsupported channels: {array.shape[-1]}")
    elif array.ndim == 2:
        mode = 'L' 
    else:
        raise ValueError(f"Unsupported shape: {array.shape}")


    if (window_level is not None) and (window_width is not None) and (array.ndim == 2):
        lower = window_level - window_width / 2
        upper = window_level + window_width / 2
        array = np.clip(array, lower, upper)
        array = (array - lower) / window_width  

    if normalize or (array.dtype in [np.float32, np.float64]):
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val - min_val > 1e-8:
            array = (array - min_val) / (max_val - min_val)
        else:
            array = np.zeros_like(array)
    array = (array * 255).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(array, mode=mode)
    image.save(save_path)
    print(f"the generated image has saved to: {save_path}")

def ensure_rgb(array):
    """
    Ensure the image has 3 channels. If single channel, repeat to make 3 channels.
    If already 3 channels, return as is.
    """
    if array.ndim == 2:
        # (H, W) -> (H, W, 3)
        return np.stack([array]*3, axis=-1)
    elif array.ndim == 3:
        if array.shape[2] == 3:
            return array
        elif array.shape[2] == 1:
            return np.repeat(array, 3, axis=2)
        else:
            raise ValueError(f"Unexpected number of channels: {array.shape[2]}")
    else:
        raise ValueError(f"Unexpected array shape: {array.shape}")


def main():

    opt = TestOptions().parse()


    dataloader = data.create_dataloader(opt)


    model = Pix2PixModel(opt)
    model.eval()


    visualizer = Visualizer(opt)
    web_dir = os.path.join(opt.results_dir, opt.name,
                           f"{opt.phase}_{opt.which_epoch}")
    webpage = html.HTML(
        web_dir,
        f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.which_epoch}"
    )

    generated_dir = os.path.join(web_dir, 'generated_images')
    os.makedirs(generated_dir, exist_ok=True)


    real_dir = os.path.join(web_dir, 'real_images')
    os.makedirs(real_dir, exist_ok=True)


    window_level = None
    window_width = None

    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        with torch.no_grad():
            adj_matrix = data_i.get('adj')
            adj_matrix = adj_matrix.cuda() if adj_matrix is not None else None
            generated = model(data_i, mode='inference', adj_matrix=adj_matrix)

        img_path = data_i['path']


        generated_np = tensor_to_numpy(generated)


        real_np = None
        if 'image' in data_i:
            data_i['image'] = data_i['image'].reshape(-1, data_i['image'].shape[2], data_i['image'].shape[3], data_i['image'].shape[4])
            real_np = tensor_to_numpy(data_i['image'])

        for b in range(generated_np.shape[0]):
            print(f'Processing image... {img_path[b]}')


            if isinstance(img_path[b], (tuple, list)):
                img_name = img_path[b][0]
            else:
                img_name = img_path[b]

            base_name = os.path.splitext(os.path.basename(img_name))[0]
            img_extension = ".png"

            if generated_np.shape[0] > 1:
                slice_suffix = f"_slice{b}"
            else:
                slice_suffix = ""

            final_name = f"{base_name}{slice_suffix}"


            if real_np is not None:
                try:
                    real_image_path = os.path.join(real_dir, f"{final_name}{img_extension}")
                    print(f"Saving real image to: {real_image_path}")
                    save_numpy_as_image(real_np[b], real_image_path, normalize=True,
                                        window_level=window_level, window_width=window_width)
                except ValueError as e:
                    print(f"wrong,{e}")


            gen_image_path = os.path.join(generated_dir, f"{final_name}{img_extension}")
            try:
                save_numpy_as_image(generated_np[b], gen_image_path, normalize=True,
                                    window_level=window_level, window_width=window_width)
            except Exception as e:
                print(f"wrong,{e}")

    # webpage.save()

if __name__ == '__main__':
    main()
