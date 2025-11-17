# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

from torchvision.transforms import v2
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import json
import os

from utils import *
from kitti_utils import *
from layers import *
from datasets.virtual_gallery import *
import matplotlib.cm
import matplotlib.pyplot as plt

import datasets
import networks
from tqdm import tqdm


import copy
import pdb

data_path = "/home/woody/iwnt/iwnt138h/virtual_gallery_dataset"


class SimpleDepthDecoder(nn.Module):
    def __init__(self, in_channels=1536):  # Swin-Large final feature map channels
        super(SimpleDepthDecoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)  # output depth map
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[-1]
        # Upsample to match input resolution
        x = F.interpolate(x, size=(352, 704), mode='bilinear', align_corners=False)
        return self.decode(x)



class Trainer:
    def __init__(self, options): 
       
        self.data_path = data_path
        self.opt = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opt.model_name = "Virtual_gallery"
        self.opt.dataset = "Virtual_gallery"
        self.transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),  
            v2.RandomRotation(degrees=10),  
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),    
            v2.RandomResizedCrop(size=(352, 704), scale=(0.7, 1.0)),
            v2.GaussianBlur(kernel_size=3),
            v2.RandomGrayscale(p=0.2),
            v2.RandomPerspective(distortion_scale=0.5, p=0.5),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        self.max_depth = 10 #in meters
        self.dataset = datasets.VirtualGalleryDataset
        if self.opt.dataset == "dgp":
            self.gt_height, self.gt_width = 1920, 1152
        else:
            # kitti
            self.gt_height, self.gt_width = 352, 704
        self.log_path = os.path.join(self.opt.log_dir, 'Virtual_gallery(dr1e-6)')
        

       # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' mustbe a multiple of 32"

        # create a new log for every save
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.models = {}    
        self.parameters_to_train = []

    
        if self.opt.sup:
            self.opt.scales = [0]
        self.num_scales = len(self.opt.scales)

        self.num_pose_frames = 2 


        if self.opt.sup:
            self.models["encoder"] = networks.trans_backbone(
                'large07', './models/swin_large_patch4_window7_224_22k.pth')
            self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"])
            self.models["depth"] = networks.NewCRFDepth(version = 'large07', max_depth = 10)
            # self.models["depth"] = SimpleDepthDecoder()
        else:
            # resnet
            pass

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"])
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.opt.selfsup:
            pass
        else:
            self.prev_epoch = -1

        eps = 1e-3
        weight_decay = 1e-3
        self.model_optimizer = optim.Adam(self.parameters_to_train, lr=self.opt.learning_rate, eps=eps, weight_decay=weight_decay)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
                  

        img_ext = '.png' if self.opt.png else '.jpg'

        
        train_dataset = self.dataset("training",self.data_path, transform = self.transform)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle = True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset("testing",self.data_path)
        
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        if self.opt.selfsup:
            pass
        else:
            self.sup_train_loader = self.train_loader
            self.num_sup_samples = len(train_dataset)
            self.num_unsup_samples = 0

        if self.opt.sup:
            self.num_total_steps = self.num_sup_samples // self.opt.batch_size * self.opt.num_epochs
        else:
            self.num_total_steps = self.num_unsup_samples // self.opt.batch_size * self.opt.num_epochs

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        if not self.opt.resume:
            start_epoch = 0
            self.step = 0
        else:
            print('resuming from previous training')
            start_epoch = self.prev_epoch + 1
            if self.opt.sup:
                self.step = self.num_sup_samples // self.opt.batch_size * start_epoch
            else:
                self.step = self.num_unsup_samples // self.opt.batch_size * start_epoch

        self.start_time = time.time()
        for self.epoch in range(start_epoch, self.opt.num_epochs):
            if  self.opt.sup:
                self.train_loader = self.sup_train_loader
                print("training supervised")
            else:
                print("training self-supervised")
                self.train_loader = self.unsup_train_loader

            print("epoch "+str(self.epoch))
            self.run_epoch()
            print("running epoch")
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Training")
        self.set_train()
        total_train_error =[]
        total_epoch_loss = 0
        epoch_batch_count = 0
        
        pbar = tqdm(total = len(self.train_loader))
        for batch_idx, inputs in enumerate(self.train_loader):
            self.loss_nan = False
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            if self.loss_nan or torch.isnan(losses["loss"]).any():
                print("NaN loss detected, skipping batch")
                continue
            self.model_optimizer.zero_grad()
            losses["loss"].backward()

            for param_group in self.model_optimizer.param_groups:
                if self.epoch < self.opt.scheduler_step_size:
                    current_lr = self.opt.learning_rate
                # else:
                #     current_lr = self.opt.learning_rate * 0.1
                # current_lr = (self.opt.learning_rate - 0.1 * self.opt.learning_rate) * \
                # (1 - self.step / self.num_total_steps) ** 0.9 + 0.1 * self.opt.learning_rate
                param_group['lr'] = current_lr
            self.model_optimizer.step()

            total_epoch_loss += losses["loss"].item()
            epoch_batch_count += 1
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                duration = time.time() - before_op_time
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                for key, ipt in inputs.items():
                    inputs[key] = ipt.to(self.device)
                if "depth_image" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                # self.val()

            self.step += 1
            avg_train_loss = total_epoch_loss / epoch_batch_count
            
            pbar.update(1)
            pbar.set_description("loss: {:.4f}".format(losses["loss"].item()))

        pbar.close()

        duration = time.time() - before_op_time
        self.log_time(batch_idx, duration, losses["loss"].cpu().data)
        print(f"\n[Training - Epoch {self.epoch}] Average Loss: {avg_train_loss:.4f}")
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        if "depth_image" in inputs:
            training_errors = self.compute_depth_losses(inputs, outputs, losses)
            total_train_error.append(training_errors)
            mean_errors = np.mean(total_train_error, axis=0)
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors
                       
            print(f"\n[Training Results - Full Dataset]")
            print(f"abs_rel: {abs_rel:.3f} | sq_rel: {sq_rel:.3f} | rmse: {rmse:.3f} | rmse_log: {rmse_log:.3f}")
            print(f"a1: {a1:.3f} | a2: {a2:.3f} | a3: {a3:.3f}\n")
            
        self.log("train", inputs, outputs, losses)
        if (self.epoch + 1) % 5 == 0:
        
            self.val()


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        rgb_image = inputs["rgb_image"].to(self.device)
        depth_image = inputs["depth_image"].to(self.device)
       
        outputs = {}

        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features = self.models["encoder"](rgb_image)
        # print(f"self.opt.sup: {self.opt.sup}")
        if self.opt.sup:
            single_depth = self.models["depth"](features)
            outputs["depth"] = {scale: F.interpolate(single_depth, scale_factor=1/(2**scale), mode="bilinear", align_corners=False)
                                for scale in self.opt.scales}

            losses = self.compute_losses_supervised(inputs, outputs)
        elif self.opt.selfsup:
            pass
        else:
            print("Please specify which training is required supervised or self supervised")   
          
        return outputs, losses
        
        
    def val(self):
        """Validate the model on the entire validation set"""
        self.set_eval()
        total_depth_errors = []

        for inputs in self.val_loader:
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)

                if "depth_image" in inputs:
                    depth_errors = self.compute_depth_losses(inputs, outputs, losses)
                    total_depth_errors.append(depth_errors)
                    
                self.log("val", inputs, outputs, losses)
                del inputs, outputs, losses
        if total_depth_errors:
            mean_errors = np.mean(total_depth_errors, axis=0)
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors
            print(f"\n[Validation - Full Dataset]")
            print(f"abs_rel: {abs_rel:.3f} | sq_rel: {sq_rel:.3f} | rmse: {rmse:.3f} | rmse_log: {rmse_log:.3f}")
            print(f"a1: {a1:.3f} | a2: {a2:.3f} | a3: {a3:.3f}\n")

        self.set_train()

    

    def compute_losses_supervised(self, inputs, outputs):
        """Compute the supervision loss without masking"""

        losses = {}
        total_loss = 0

        # Define full-resolution ground truth once
        depth_gt_full = torch.clamp(inputs["depth_image"].to(self.device).to(torch.float32), min=1e-3, max=1000) / 1000
        gt_height, gt_width = depth_gt_full.shape[2], depth_gt_full.shape[3]

        # Generate downscaled GT for each scale
        gt = {
            scale: F.interpolate(depth_gt_full, scale_factor=1 / (2 ** scale), mode="bilinear", align_corners=False)
            for scale in self.opt.scales
        }

        for scale in self.opt.scales:
            loss = 0

            depth_pred = outputs["depth"][scale] if isinstance(outputs["depth"], dict) else outputs["depth"]
            depth_gt = gt[scale]  # Get scaled ground truth

            # Clamp prediction and scale to 0-1
            depth_pred = torch.clamp(depth_pred, min=1e-3, max=10) / 10

            # Compute SILog loss
            d = torch.log(depth_pred + 1e-7) - torch.log(depth_gt + 1e-7)
            variance_focus = 0.85
            sil_loss = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

            total_loss += sil_loss
            losses[f"loss/{scale}"] = sil_loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        depth_gt = inputs["depth_image"].to(self.device).to(torch.float32)
        gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]

        # Mask out invalid depths
        mask = depth_gt > 1.0

        depth_pred = outputs["depth"][0].to(torch.float32)
        depth_pred = F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        if self.opt.selfsup:
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_gt = torch.clamp(depth_gt, min=1e-3, max=1000) / 1000
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=10) / 10

        depth_errors = self.compute_depth_errors(depth_gt.detach().cpu().numpy(), depth_pred.detach().cpu().numpy())


        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = depth_errors[i]

        return depth_errors

            
    def compute_depth_errors(self, gt, pred):

        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()

        a2 = (thresh < 1.25 ** 2).mean()

        a3 = (thresh < 1.25 ** 3).mean()


        rmse = np.sqrt(((gt - pred) ** 2).mean())
        rmse_log = np.sqrt((np.log(gt) - np.log(pred)) ** 2).mean()
        abs_rel = np.mean(np.abs(gt - pred) / gt)
        sq_rel = np.mean(((gt - pred) ** 2) / gt)


        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        rgb_image = inputs["rgb_image"].squeeze(0)           # shape: (B, 3, H, W)
        depth_image = inputs["depth_image"].squeeze(0)       # shape: (B, 1, H, W) or (B, H, W)
        depth_preds = outputs["depth"][0].squeeze(0)         # shape: (B, 1, H, W) or (B, H, W)

        writer = self.writers[mode]

        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maximum of four images
            # ---- Ground truth depth ----
            depth_gt = depth_image[j].to(torch.float32)  # (1, H, W) or (H, W)
            if depth_gt.dim() == 2:
                depth_gt = depth_gt.unsqueeze(0)  # (1, H, W)
            depth_gt = depth_gt.repeat(3, 1, 1)  # (3, H, W)
            writer.add_image(
                "depth_gt/{}".format(j),
                normalize_image(depth_gt),
                self.step
            )

            # ---- Predicted depth ----
            depth_pred = depth_preds[j].to(torch.float32)  # (1, H, W) or (H, W)
            if depth_pred.dim() == 2:
                depth_pred = depth_pred.unsqueeze(0)  # (1, H, W)
            depth_pred = depth_pred.repeat(3, 1, 1)  # (3, H, W)
            writer.add_image(
                "depth_pred/{}".format(j),
                normalize_image(depth_pred),
                self.step
            )

            # ---- RGB image ----
            rgb_input = rgb_image[j].to(torch.float32)
            if rgb_input.dim() == 2:
                rgb_input = rgb_input.unsqueeze(0)# (3, H, W)
            writer.add_image(
                "rgb_input/{}".format(j),
                normalize_image(rgb_input),
                self.step
            )


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            try:
                self.model_optimizer.load_state_dict(optimizer_dict)
            except ValueError:
                print("Cannot find Adam weights so Adam is randomly initialized")
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


