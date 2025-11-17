import os
import copy
import time
import math
import numpy as np
import cv2
import torch
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import transform
import matplotlib.cm
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
import datasets
import networks
from layers import *
from utils import *
from options import MonodepthOptions
from tqdm import tqdm
import pdb
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import json
from datasets.nyu_dataset import ImageTransformer


options = MonodepthOptions()
opts = options.parse()

class Adapt:
    def __init__(self, options):
        self.opt = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transformer = ImageTransformer()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 10
        self.ema_decay = 0.99
        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_name = "nyu_adapt"  
        self.load_path = "/home/hpc/iwnt/iwnt138h/ada-depth/exp_logs/Virtual_gallery(1e-6)/models/weights_21"
        self.input_height, self.input_width = None, None
        if self.opt.dataset == "nyu":
            self.data_path = "/home/woody/iwnt/iwnt138h/NYU"
            self.gt_height, self.gt_width = 352, 704
            self.height, self.width = 352, 704
        else:
            raise ValueError(f"Dataset {self.opt.dataset} not supported yet.")

        self.log_path = os.path.join("./adapt_logs", self.model_name, run_id)
        self.save_path = os.path.join("./adap_model", "nyu" )
        # os.makedirs(self.save_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.device)

        self.ssim = SSIM().to(self.device)

        # models to train (student)
        self.models = {}
        self.parameters_to_train = []
        self.models["encoder"] = networks.trans_backbone(
            'large07', './models/swin_large_patch4_window7_224_22k.pth')
        self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
        self.models["depth"] = networks.NewCRFDepth(version = 'large07', max_depth = self.MAX_DEPTH)
        self.models["encoder"].to(self.device)
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        self.models["depth"].to(self.device)
       
        # Load pretrained weights
        encoder_ckpt = torch.load('/home/hpc/iwnt/iwnt138h/ada-depth/exp_logs/Virtual_gallery(1e-6)/models/weights_21/encoder.pth', map_location=self.device)
        decoder_ckpt = torch.load('/home/hpc/iwnt/iwnt138h/ada-depth/exp_logs/Virtual_gallery(1e-6)/models/weights_21/depth.pth', map_location=self.device)
        self.models["encoder"].load_state_dict(encoder_ckpt, strict=False)
        self.models["depth"].load_state_dict(decoder_ckpt, strict=False)

        # self.models["encoder"].eval()
        # self.models["depth"].eval()
        
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())
        
        load_model_list = ['encoder', 'depth']
        if self.opt.load_weights_folder is not None:
            self.load_model(self.opt.load_weights_folder, self.models, load_model_list)

        # ema models for pseudo-labeling (teacher)
        self.models_ema = copy.deepcopy(self.models)
        for m in self.models_ema.values():
            m.eval()

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.step = 0
        self.depth_metric_names = [
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
        ]
        

    def run_adapt(self):
        self.start_time = time.time()

        if self.opt.dataset == "nyu":
            train_dataset = datasets.NYUv2RawDataset(self.data_path, split='train')
        else:
            raise ValueError(f"Dataset {self.opt.dataset} not supported for adaptation.")

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.opt.batch_size,
            shuffle=False,
            num_workers=self.opt.num_workers
        )
        print(f"Train Items: {len(train_dataset)}")

        errors_tt = []

        for epoch in range(self.opt.num_epochs):  # Assuming self.opt.num_epochs exists
            print(f"\nEpoch {epoch + 1}/{self.opt.num_epochs}")
            pbar = tqdm(total=len(self.dataloader))
            
            for batch_idx, (rgb_img, depth_gt) in enumerate(self.dataloader):
                rgb_img = rgb_img.to(self.device)
                depth_gt = depth_gt.to(self.device)
                self.step += 1
                before_op_time = time.time()

                outputs, losses = self.process_batch(rgb_img, depth_gt)
                errors_tt.append(losses['depth_errors'])

                for i, metric in enumerate(self.depth_metric_names):
                    self.writer.add_scalar(f"Metrics/{metric}", losses['depth_errors'][i], self.step)
                self.writer.add_scalar("Loss/self_training", losses["self_training_loss"], self.step)

                if batch_idx % 10 == 0:
                    self.log(rgb_img, depth_gt, outputs, losses)

                mean_errors = np.array(errors_tt).mean(axis=0)
                pbar.set_description(" | ".join(f"{v:.4f}" for v in mean_errors))
                pbar.update(1)
            
            pbar.close()

            print("\nMean Errors on Test Set:")
            for v in mean_errors:
                print(f" {v:.4f}")
        
        self.save_model()
        print("\nAdaptation completed. Starting final evaluation on test set...")

        final_gt_all = []
        final_pred_all = []

        for model_name in self.models:
            self.models[model_name].eval()

        with torch.no_grad():
            test_dataset = datasets.NYUv2RawDataset(self.data_path, split='test')  # Changed to 'test'
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                num_workers=self.opt.num_workers
            )

            for eval_rgb_img, eval_depth_gt in test_dataloader:
                eval_rgb_img = eval_rgb_img.to(self.device)
                eval_depth_gt = eval_depth_gt.to(self.device)

                eval_features = self.models["encoder"](eval_rgb_img)
                eval_pred = self.models["depth"](eval_features)

                final_gt_all.append(eval_depth_gt.cpu().numpy())
                final_pred_all.append(eval_pred.cpu().numpy())

        final_gt_all_np = np.concatenate(final_gt_all, axis=0)
        final_pred_all_np = np.concatenate(final_pred_all, axis=0)

        print("\nComputing final evaluation metrics...")
        self.compute_and_log_metrics(final_gt_all_np, final_pred_all_np)

        self.writer.close()

    def process_batch(self, rgb_img, depth_gt): 
        rgb_img = rgb_img.to(self.device)
        I1, transform_params = self.image_transformer.transform_image(rgb_img)

        # Ensure student models are in training mode
        for model_name in self.models:
            self.models[model_name].train()

        features_student_orig = self.models["encoder"](rgb_img)
        depth_pred_orig = self.models["depth"](features_student_orig)

        features_student_transformed = self.models["encoder"](I1)
        depth_pred_transformed = self.models["depth"](features_student_transformed)

        depth_pred_transformed_inv = self.image_transformer.depth_inverse_transform(depth_pred_transformed, transform_params)

        # Clamp predictions to valid depth range (slightly inside range to avoid boundary issues)
        eps = 1e-6
        depth_pred_orig = depth_pred_orig.clamp(self.MIN_DEPTH + eps, self.MAX_DEPTH - eps)
        depth_pred_transformed = depth_pred_transformed.clamp(self.MIN_DEPTH + eps, self.MAX_DEPTH - eps)
        depth_pred_transformed_inv = depth_pred_transformed_inv.clamp(self.MIN_DEPTH + eps, self.MAX_DEPTH - eps)

        # Set EMA models to eval mode
        for model_name in self.models_ema:
            self.models_ema[model_name].eval()

        with torch.no_grad():
            features_teacher = self.models_ema["encoder"](rgb_img)
            depth_teacher = self.models_ema["depth"](features_teacher)
        depth_teacher = depth_teacher.clamp(self.MIN_DEPTH + eps, self.MAX_DEPTH - eps)

        # Create masks to exclude invalid or NaN values from loss computation
        mask_pred_orig = (depth_pred_orig > self.MIN_DEPTH) & (depth_pred_orig < self.MAX_DEPTH) & (~torch.isnan(depth_pred_orig))
        mask_pred_transformed_inv = (depth_pred_transformed_inv > self.MIN_DEPTH) & (depth_pred_transformed_inv < self.MAX_DEPTH) & (~torch.isnan(depth_pred_transformed_inv))
        mask_teacher = (depth_teacher > self.MIN_DEPTH) & (depth_teacher < self.MAX_DEPTH) & (~torch.isnan(depth_teacher))

        # Common masks for losses (only pixels valid in both)
        common_mask_consistency = mask_pred_orig & mask_pred_transformed_inv
        common_mask_teacher = mask_pred_orig & mask_teacher

        # Consistency loss calculation (L1)
        if common_mask_consistency.sum() > 0:
            loss_consistency = F.l1_loss(
                depth_pred_orig[common_mask_consistency], 
                depth_pred_transformed_inv[common_mask_consistency]
            )
        else:
            loss_consistency = torch.tensor(0.0, device=self.device, dtype=depth_pred_orig.dtype, requires_grad=True)

        # Teacher loss calculation (L1)
        if common_mask_teacher.sum() > 0:
            loss_teacher = F.l1_loss(
                depth_pred_orig[common_mask_teacher], 
                depth_teacher[common_mask_teacher]
            )
        else:
            loss_teacher = torch.tensor(0.0, device=self.device, dtype=depth_pred_orig.dtype, requires_grad=True)

        # Combine losses: early steps only consistency, later steps add teacher guidance
        if self.step < 1:  
            total_loss = loss_consistency
        else:
            total_loss =  0.8 * loss_consistency + 0.5 * loss_teacher
        # def print_tensor_min_max(name, tensor):
        #     print(f"{name} — min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}")   
        # print_tensor_min_max("RGB Image", rgb_img)
        # print_tensor_min_max("Transformed Image (I1)", I1)
        # print_tensor_min_max("Depth Prediction (Original)", depth_pred_orig)
        # print_tensor_min_max("Depth Prediction (Transformed)", depth_pred_transformed)
        # print_tensor_min_max("Depth Prediction (Transformed Inverse)", depth_pred_transformed_inv)
        # print_tensor_min_max("Depth Teacher", depth_teacher)

        
        # Optional visualization
        self.visualize_batch(
            rgb_img, I1, depth_pred_orig, depth_pred_transformed, 
            depth_pred_transformed_inv, depth_teacher, 
            step=None, 
            save_dir="/home/hpc/iwnt/iwnt138h/ada-depth/adapt_logs/images"
        )

        # Backpropagation
        self.model_optimizer.zero_grad()
        total_loss.backward()
        self.model_optimizer.step()

        # Update EMA
        update_ema_variables(self.models["encoder"], self.models_ema["encoder"], self.ema_decay, self.step)
        update_ema_variables(self.models["depth"], self.models_ema["depth"], self.ema_decay, self.step)

        # Set student models to eval mode after update
        for model_name in self.models:
            self.models[model_name].eval()

        depth_errors = [torch.tensor(0.0, device=self.device)] * len(self.depth_metric_names)

        with torch.no_grad():
            if depth_gt is not None:
                depth_errors = self.compute_depth_losses(depth_gt, depth_pred_orig)

        outputs = {
            "depth": depth_pred_orig.detach(),
            "pseudo_depth": depth_teacher.detach()
        }
        losses = {
            "self_training_loss": total_loss.item(),
            "consistency_loss": loss_consistency.item(),
            "teacher_loss": loss_teacher.item(),
            "depth_errors": [e.item() for e in depth_errors]
        }

        return outputs, losses


    
    def compute_and_log_metrics(self, gt_all, pred_all):
        gt_all_np = gt_all.cpu().numpy() if isinstance(gt_all, torch.Tensor) else gt_all
        pred_all_np = pred_all.cpu().numpy() if isinstance(pred_all, torch.Tensor) else pred_all

        errors = []
        
        for i in range(gt_all_np.shape[0]):
            gt = gt_all_np[i]
            pred = pred_all_np[i]

            gt = gt / 1000.0
            gt = np.clip(gt, self.MIN_DEPTH, 10.0)
            pred = np.clip(pred, self.MIN_DEPTH, 10.0)

            mask = np.logical_and(gt > self.MIN_DEPTH, gt < self.MAX_DEPTH)
            gt = gt[mask]
            pred = pred[mask]

            if gt.size == 0 or pred.size == 0:
                continue

            pred *= np.median(gt) / np.median(pred)
            pred = np.clip(pred, self.MIN_DEPTH, self.MAX_DEPTH)
            gt = np.clip(gt, self.MIN_DEPTH, self.MAX_DEPTH)

            raw_errors_for_sample = self.compute_depth_losses(gt, pred)
            converted_errors_for_sample = []
            for err_item in raw_errors_for_sample:
                if isinstance(err_item, torch.Tensor):
                    converted_errors_for_sample.append(err_item.cpu().numpy())
                else:
                    converted_errors_for_sample.append(err_item)
            
            errors.append(converted_errors_for_sample) 
            

        mean_errors = np.array(errors).mean(0) 

        print("\nEvaluation Results:")
        for i, metric in enumerate(self.depth_metric_names):
            print(f"{metric}: {mean_errors[i]:.4f}")
            self.writer.add_scalar(f"eval/{metric}", mean_errors[i])

        self.writer.close()
 
    def compute_depth_losses(self, gt, pred):
        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt)
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)

        gt = gt.to(torch.float32).to(self.device)
        pred = pred.to(torch.float32).to(self.device)

        mask = (gt > self.MIN_DEPTH) & (gt < self.MAX_DEPTH)

        gt_masked = gt[mask]
        pred_masked = pred[mask]

        if gt_masked.numel() == 0:
            return [torch.tensor(0.0, device=self.device)] * len(self.depth_metric_names)

        scale = torch.median(gt_masked) / torch.median(pred_masked)
        pred_scaled = pred_masked * scale
        
        pred_scaled = torch.clamp(pred_scaled, self.MIN_DEPTH, self.MAX_DEPTH)
        gt_masked = torch.clamp(gt_masked, self.MIN_DEPTH, self.MAX_DEPTH)

        gt_masked = gt_masked.to(self.device)
        pred_scaled = pred_scaled.to(self.device)

        abs_rel = torch.mean(torch.abs(gt_masked - pred_scaled) / gt_masked)
        sq_rel = torch.mean(torch.square(gt_masked - pred_scaled) / gt_masked)
        rmse = torch.sqrt(torch.mean(torch.square(gt_masked - pred_scaled)))
        rmse_log = torch.sqrt(torch.mean(torch.square(torch.log(gt_masked) - torch.log(pred_scaled))))

        thresh = torch.max((gt_masked / pred_scaled), (pred_scaled / gt_masked))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25 ** 2).float().mean()
        a3 = (thresh < 1.25 ** 3).float().mean()

        
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


    def log(self, rgb_img, depth_gt, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writer
        for l, v in losses.items():
            if isinstance(v, list):
                for i, val in enumerate(v):
                    writer.add_scalar(f"{l}/{self.depth_metric_names[i]}", val, self.step)
            else:
                writer.add_scalar(f"{l}", v, self.step)

        j = 0  # Assuming batch size of 1 for visualization
        cmap = matplotlib.cm.viridis
        cmap.set_bad('white', 1.)

        # Prediction from student
        pred_student = copy.deepcopy(outputs["depth"][j]).squeeze().detach().cpu().numpy()
        pred_student = (255 * cmap(pred_student / self.MAX_DEPTH)).astype('uint8')[:, :, :3]
        pred_student = np.transpose(pred_student, (2, 0, 1))
        writer.add_image("pred_student", pred_student, self.step)

        # Pseudo-label from EMA
        pseudo = copy.deepcopy(outputs["pseudo_depth"][j]).squeeze().detach().cpu().numpy()
        pseudo = (255 * cmap(pseudo / self.MAX_DEPTH)).astype('uint8')[:, :, :3]
        pseudo = np.transpose(pseudo, (2, 0, 1))
        writer.add_image("pseudo_depth", pseudo, self.step)

        writer.add_image("color_input", rgb_img[j], self.step)

        if depth_gt is not None:
            depth_gt_vis = copy.deepcopy(depth_gt[j]).squeeze().detach().cpu().numpy()
            depth_gt_vis = np.where(depth_gt_vis > 0, depth_gt_vis, np.nan)
            depth_gt_vis = (255 * cmap(depth_gt_vis / self.MAX_DEPTH)).astype('uint8')[:, :, :3]
            depth_gt_vis = np.transpose(depth_gt_vis, (2, 0, 1))
            writer.add_image("depth_gt", depth_gt_vis, self.step)
            

    def visualize_batch(self, rgb_img, I1, depth_pred_orig, depth_pred_transformed,
                        depth_pred_transformed_inv, depth_teacher, step=None,
                        save_dir="/home/hpc/iwnt/iwnt138h/ada-depth/adapt_logs/images"):
        """Visualizes and optionally saves batch-level visualizations using cv2 with centered bold titles."""

        def normalize_depth(depth):
            depth = np.clip(depth, self.MIN_DEPTH, self.MAX_DEPTH)
            if np.all(np.isnan(depth)):
                return np.zeros_like(depth)
            min_val = np.nanmin(depth)
            max_val = np.nanmax(depth)
            return ((depth - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8)

        def add_title_to_image(img, title, height=30, font_scale=0.8, thickness=2):
            """Adds a centered bold title above an image."""
            h, w = img.shape[:2]
            banner = np.ones((height, w, 3), dtype=np.uint8) * 255  # white banner

            # Get text size and center position
            text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = height - 10

            # Draw the text
            cv2.putText(banner, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
            return np.vstack([banner, img])

        # Convert tensors to NumPy
        rgb_np = rgb_img[0].cpu().permute(1, 2, 0).numpy()
        I1_np = I1[0].cpu().permute(1, 2, 0).numpy()
        
        d1_np = normalize_depth(depth_pred_orig[0].squeeze().detach().cpu().numpy())
        d2_np = normalize_depth(depth_pred_transformed[0].squeeze().detach().cpu().numpy())
        d2_inv_np = normalize_depth(depth_pred_transformed_inv[0].squeeze().detach().cpu().numpy())
        d_teacher_np = normalize_depth(depth_teacher[0].squeeze().detach().cpu().numpy())

        # Convert RGB images to BGR
        rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
        I1_np = (np.clip(I1_np, 0, 1) * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        I1_bgr = cv2.cvtColor(I1_np, cv2.COLOR_RGB2BGR)

        # Convert depth maps to color
        d1_color = cv2.applyColorMap(d1_np, cv2.COLORMAP_VIRIDIS)
        d2_color = cv2.applyColorMap(d2_np, cv2.COLORMAP_VIRIDIS)
        d2_inv_color = cv2.applyColorMap(d2_inv_np, cv2.COLORMAP_VIRIDIS)
        d_teacher_color = cv2.applyColorMap(d_teacher_np, cv2.COLORMAP_VIRIDIS)

        # Add centered bold titles
        rgb_bgr = add_title_to_image(rgb_bgr, "I1")
        I1_bgr = add_title_to_image(I1_bgr, "I2")
        d1_color = add_title_to_image(d1_color, "D1")
        d2_color = add_title_to_image(d2_color, "D2")
        d2_inv_color = add_title_to_image(d2_inv_color, "D2'")
        d_teacher_color = add_title_to_image(d_teacher_color, "Depth Teacher")

        # Stack images
        top_row = np.hstack([rgb_bgr, I1_bgr, d1_color])
        bottom_row = np.hstack([d2_color, d2_inv_color, d_teacher_color])
        full_image = np.vstack([top_row, bottom_row])

        # Save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"vis_step_{step:06d}.png" if step is not None else "vis.png"
            cv2.imwrite(os.path.join(save_dir, filename), full_image)
        else:
            cv2.imshow("Batch Visualization", full_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



    def save_model(self):
        save_folder = self.save_path
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            model_path = os.path.join(save_folder, f"{model_name}.pth")
            # Save the inner model weights (from DataParallel)
            torch.save(model.module.state_dict(), model_path)

        # Save optimizer (optional)
        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))


if __name__ == "__main__":
    adapt = Adapt(opts)
    adapt.run_adapt()
    
    
