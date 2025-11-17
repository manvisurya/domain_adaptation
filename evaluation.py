import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.cm
from tensorboardX import SummaryWriter
import datasets
import networks
from layers import SSIM
from utils import *
from options import MonodepthOptions
from tqdm import tqdm


data_root = "/home/woody/iwnt/iwnt138h/NYU"
class evaluate:
    def __init__(self, options):
        self.opt = options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 10
        self.dataset = self.opt.dataset
        self.opt.model_name = "nyu_adapt"

        self.input_height, self.input_width = 352, 704
        self.log_dir = os.path.join("./eval_result", self.opt.model_name)
        self.writer = SummaryWriter(self.log_dir)

        self.depth_metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        self.num_scales = len(self.opt.scales)

        self.ssim = SSIM().to(self.device)

        self.models = {}
        self.models["encoder"] = networks.trans_backbone('large07', './models/swin_large_patch4_window7_224_22k.pth')
        self.models["encoder"] = torch.nn.DataParallel(self.models["encoder"]).to(self.device)
        self.models["depth"] = networks.NewCRFDepth(version='large07', max_depth=self.MAX_DEPTH)
        self.models["depth"] = torch.nn.DataParallel(self.models["depth"]).to(self.device)

        # Load pretrained weights
        encoder_path = '/home/hpc/iwnt/iwnt138h/ada-depth/exp_logs/kitti_unsup/models/weights_24/encoder.pth'
        decoder_path = '/home/hpc/iwnt/iwnt138h/ada-depth/exp_logs/kitti_unsup/models/weights_24/depth.pth'
        self.models["encoder"].module.load_state_dict(torch.load(encoder_path, map_location=self.device), strict=False)
        self.models["depth"].module.load_state_dict(torch.load(decoder_path, map_location=self.device), strict=False)

        self.models["encoder"].eval()
        self.models["depth"].eval()

        print(f"Setting up the Model for evaluation on {self.dataset}")

    def run_infer(self, load_weights_folder=None):
        print("Models and tensorboard events files are saved to:\n  ", self.log_dir)
        print("Evaluation is using:  ", self.device)

        if self.dataset == "nyu":
            test_dataset = datasets.NYUv2RawDataset(data_root, split='test')
            train_dataset = datasets.NYUv2RawDataset(data_root, split ='train')
            
        else:
            raise ValueError(f"Dataset {self.dataset} not supported for evaluation.")

        self.dataloader = DataLoader(test_dataset, batch_size=self.opt.batch_size,
                                     shuffle=False, num_workers=self.opt.num_workers)
        self.dataloader = DataLoader(train_dataset, batch_size=self.opt.batch_size, shuffle=False, num_workers=self.opt.num_workers)

        pbar = tqdm(total=len(self.dataloader))

        all_depth_preds = []
        all_depth_gts = []

        with torch.no_grad():
            for batch_idx, (rgb_img, depth_gt) in enumerate(self.dataloader):
                rgb_img = rgb_img.to(self.device)
                depth_gt = depth_gt.to(self.device)

                features = self.models["encoder"](rgb_img)
                depth_pred = self.models["depth"](features)
                
                depth_pred = torch.clamp(F.interpolate(
                    depth_pred, [depth_gt.shape[2], depth_gt.shape[3]],
                    mode="bilinear", align_corners=False), self.MIN_DEPTH, self.MAX_DEPTH)

                all_depth_preds.append(depth_pred.cpu().numpy())
                all_depth_gts.append(depth_gt.cpu().numpy())

                pbar.update(1)

        pbar.close()

        self.compute_and_log_metrics(np.concatenate(all_depth_gts), np.concatenate(all_depth_preds))

    def compute_and_log_metrics(self, gt_all, pred_all):
        errors = []
        for i in range(gt_all.shape[0]):
            gt = gt_all[i]
            pred = pred_all[i]

            gt = gt /1000.0
            gt = np.clip(gt, self.MIN_DEPTH, 10.0)
            pred = np.clip(pred, self.MIN_DEPTH, 10.0)
            
            print("GT min/max (meters, clamped):", gt.min(), gt.max())
            print("Pred min/max (meters, clamped):", pred.min(), pred.max())
            
            mask = np.logical_and(gt > self.MIN_DEPTH, gt < self.MAX_DEPTH)
            gt = gt[mask]
            pred = pred[mask]

            if gt.size == 0 or pred.size == 0:
                continue

            pred *= np.median(gt) / np.median(pred)
            pred = np.clip(pred, self.MIN_DEPTH, self.MAX_DEPTH)
            gt = np.clip(gt, self.MIN_DEPTH, self.MAX_DEPTH)

            errors.append(self.compute_depth_errors(gt, pred))

        mean_errors = np.array(errors).mean(0)

        print("\nEvaluation Results:")
        for i, metric in enumerate(self.depth_metric_names):
            print(f"{metric}: {mean_errors[i]:.4f}")
            self.writer.add_scalar(f"eval/{metric}", mean_errors[i])

        self.writer.close()
    
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



if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()  # Adjust if you're using argparse/config loading differently
    evaluator = evaluate(opts)
    evaluator.run_infer()
