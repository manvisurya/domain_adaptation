import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import cv2

from datasets import *     
from networks import *  

class Visualization:
    def __init__(self, dataset):
        self.dataset = dataset.lower()

        if self.dataset == "nyu":
            self.data_path = "/home/woody/iwnt/iwnt138h/NYU/test"
        elif self.dataset == "virtualgallery":
            self.data_path = "/home/woody/iwnt/iwnt138h/virtual_gallery_dataset"
        elif self.dataset == "kitti":
            self.data_path = "/home/woody/iwnt/iwnt138h"
        else:
            raise ValueError("Unsupported dataset")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MAX_DEPTH = 10
        self.encoder, self.decoder = self.load_model(self.device)

        self.resize_rgb = transforms.Resize((352, 704), interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.resize_depth = transforms.Resize((352, 704), interpolation=InterpolationMode.BILINEAR, antialias=True)

    def load_model(self, device):
        sup_encoder = networks.trans_backbone('large07', 'models/swin_large_patch4_window7_224_22k.pth')
        sup_encoder = torch.nn.DataParallel(sup_encoder)
        sup_decoder = networks.NewCRFDepth(version='large07', max_depth=self.MAX_DEPTH)
        sup_decoder = torch.nn.DataParallel(sup_decoder)

        sup_encoder.to(device)
        sup_decoder.to(device)

        encoder_ckpt = torch.load('/home/hpc/iwnt/iwnt138h/ada-depth/adap_model/nyu/encoder.pth', map_location=self.device)
        decoder_ckpt = torch.load('/home/hpc/iwnt/iwnt138h/ada-depth/adap_model/nyu/depth.pth', map_location=self.device)

        sup_encoder.load_state_dict(encoder_ckpt, strict=False)
        sup_decoder.load_state_dict(decoder_ckpt, strict=False)

        sup_encoder.eval()
        sup_decoder.eval()

        return sup_encoder, sup_decoder

    def load_random_images(self, num_images):
        rgb_images = []
        depth_images = []

        if self.dataset == "virtualgallery":
            for _ in range(num_images):
                light = random.randint(1, 6)
                occlusion = random.randint(1, 3)
                image_idx = random.randint(0, 495)
                idx_str = f"{image_idx:03d}"

                rgb_path = os.path.join(self.data_path, f"testing/gallery_light{light}_occlusion{occlusion}/frames/rgb/camera_0/rgb_00{idx_str}.jpg")
                depth_path = os.path.join(self.data_path, f"testing/gallery_light{light}_occlusion{occlusion}/frames/depth/camera_0/depth_00{idx_str}.png")

                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    continue

                rgb_tensor = self.resize_rgb(transforms.ToTensor()(Image.open(rgb_path).convert("RGB")))
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    continue
                depth_tensor = self.resize_depth(torch.from_numpy(depth_img).float().unsqueeze(0))

                rgb_images.append(rgb_tensor)
                depth_images.append(depth_tensor)

        elif self.dataset == "nyu":
            image_pairs = []
            for root, _, files in os.walk(self.data_path):
                rgb_files = [f for f in files if f.startswith("rgb_") and f.endswith(".jpg")]
                for rgb_file in rgb_files:
                    idx = rgb_file.replace("rgb_", "").replace(".jpg", "")
                    depth_file = f"sync_depth_{idx}.png"
                    rgb_path = os.path.join(root, rgb_file)
                    depth_path = os.path.join(root, depth_file)
                    if os.path.exists(depth_path):
                        image_pairs.append((rgb_path, depth_path))

            sampled_pairs = random.sample(image_pairs, min(num_images, len(image_pairs)))

            for rgb_path, depth_path in sampled_pairs:
                rgb_tensor = self.resize_rgb(transforms.ToTensor()(Image.open(rgb_path).convert("RGB")))
                depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth_img is None:
                    continue
                depth_tensor = self.resize_depth(torch.from_numpy(depth_img).float().unsqueeze(0))

                rgb_images.append(rgb_tensor)
                depth_images.append(depth_tensor)

        return rgb_images, depth_images

    def visualize_depth_predictions(self, rgb_images, depth_images, predicted_depth_maps):
        num_images = len(rgb_images)
        save_dir = "/home/hpc/iwnt/iwnt138h/ada-depth/preddepthmaps"
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
        

        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(num_images):
            rgb_np = rgb_images[i].permute(1, 2, 0).cpu().numpy()
            rgb_np = np.clip(rgb_np, 0, 1)
            axes[i, 0].imshow(rgb_np)
            axes[i, 0].set_title("RGB Image")
            axes[i, 0].axis('off')

            gt_np = depth_images[i].squeeze().cpu().numpy()
            axes[i, 1].imshow(gt_np, cmap='viridis')
            axes[i, 1].set_title("Ground Truth Depth")
            axes[i, 1].axis('off')

            pred_np = predicted_depth_maps[i]
            axes[i, 2].imshow(pred_np, cmap='viridis')
            axes[i, 2].set_title("Predicted Depth")
            axes[i, 2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "sample_1.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def run_visualization(self, num_samples=3):
        rgb_imgs, depth_imgs = self.load_random_images(num_samples)
        pred_depths = []

        for rgb in rgb_imgs:
            with torch.no_grad():
                inp = rgb.unsqueeze(0).to(self.device)
                features = self.encoder(inp)
                pred = self.decoder(features)
                pred = pred.squeeze().cpu().numpy()
                pred = np.clip(pred, 0, self.MAX_DEPTH)
                pred_depths.append(pred)

        self.visualize_depth_predictions(rgb_imgs, depth_imgs, pred_depths)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to visualize: nyu / virtualgallery / kitti")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random samples to visualize")
    args = parser.parse_args()

    vis = Visualization(args.dataset)
    vis.run_visualization(num_samples=args.num_samples)
