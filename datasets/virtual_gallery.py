# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import cv2
import os
from glob import glob
import random
import numpy as np
import PIL.Image
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from networks import *
import networks

data_path = "/home/woody/iwnt/iwnt138h/virtual_gallery_dataset"
class VirtualGalleryDataset():

    def __init__(self, mode,  *args, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.mode = mode
        self.rgb_depth_pairs = self.get_rgb_depth_pairs(self.mode)
            

        
    def get_rgb_depth_pairs(self, mode):

        rgb_depth_pairs = []
        
        if mode == "training":
            rgb_path_template = os.path.join(self.data_path, "training/gallery_light{}_loop{}/frames/rgb/camera_{}/")
            depth_path_template = os.path.join(self.data_path, "training/gallery_light{}_loop{}/frames/depth/camera_{}/")
            light_range = range(1, 7)
            loop_range = range(1, 5)
            camera_range = range(0, 6) 

        elif mode == "testing":
            rgb_path_template = os.path.join(self.data_path, "testing/gallery_light{}_occlusion{}/frames/rgb/camera_0/")
            depth_path_template = os.path.join(self.data_path, "testing/gallery_light{}_occlusion{}/frames/depth/camera_0/")
            light_range = range(1,7) 
            occlusion_range = range(1, 4) 
            

        else:
            raise ValueError("Invalid mode! Use 'training' or 'testing'.")

        
        for light_num in light_range:
            if mode == "training":
                for loop_num in loop_range:
                    for cam_num in camera_range:
                        rgb_folder = rgb_path_template.format(light_num, loop_num, cam_num)
                        depth_folder = depth_path_template.format(light_num, loop_num, cam_num)

                        if not os.path.exists(rgb_folder) or not os.path.exists(depth_folder):
                            continue  # Skip if folder does not exist
                        
                        rgb_files = sorted(glob(os.path.join(rgb_folder, "rgb_*.jpg")))
                        depth_files = sorted(glob(os.path.join(depth_folder, "depth_*.png")))

                        # Ensure we have matching RGB and Depth images
                        for rgb_file, depth_file in zip(rgb_files, depth_files):
                            rgb_depth_pairs.append((rgb_file, depth_file))

            elif mode == "testing":
                for occlusion_num in occlusion_range:
                    rgb_folder = rgb_path_template.format(light_num, occlusion_num)
                    depth_folder = depth_path_template.format(light_num, occlusion_num)

                    if not os.path.exists(rgb_folder) or not os.path.exists(depth_folder):
                        continue 
                    
                    rgb_files = sorted(glob(os.path.join(rgb_folder, "rgb_*.jpg")))
                    depth_files = sorted(glob(os.path.join(depth_folder, "depth_*.png")))

                    # Ensure we have matching RGB and Depth images
                    for rgb_file, depth_file in zip(rgb_files, depth_files):
                        rgb_depth_pairs.append((rgb_file, depth_file))

        return rgb_depth_pairs

    def __len__(self):
        return len(self.rgb_depth_pairs)
    
    def __getitem__(self, idx):
        rgb_path , depth_path = self.rgb_depth_pairs[idx]

        rgb_image = Image.open(rgb_path).convert("RGB")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
              
        rgb_tensor = transforms.ToTensor()(rgb_image)
        depth_tensor = transforms.ToTensor()(depth_image)

        resize_rgb = transforms.Resize((352, 704), interpolation=InterpolationMode.BILINEAR, antialias=True)
        resize_depth = transforms.Resize((352, 704), interpolation=InterpolationMode.BILINEAR, antialias=True)
        rgb_input = resize_rgb(rgb_tensor)
        depth_input = resize_depth(depth_tensor)

        return {"rgb_image": rgb_input, "depth_image": depth_input}
 

