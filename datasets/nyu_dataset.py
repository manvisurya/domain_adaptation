import os
import cv2 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF

data_root = "/home/woody/iwnt/iwnt138h/NYU"

class NYUv2RawDataset(Dataset):

    def __init__(self, data_root, split='train'):

        self.data_root = data_root
        self.split = split
        self.rgb_files = []
        self.depth_files = [] 
        self._load_files()

    def _load_files(self):
        split_path = os.path.join(self.data_root, self.split)

        for root, _, files in os.walk(split_path):
            for file in files:
                if 'rgb_' in file:
                    self.rgb_files.append(os.path.join(root, file))
                elif 'depth_' in file:
                    self.depth_files.append(os.path.join(root, file))

        self.rgb_files.sort()
        self.depth_files.sort()

        #Ensure rgb and depth files are the same length.
        if len(self.rgb_files) != len(self.depth_files):
          raise ValueError(f"Number of RGB and depth files do not match in {split_path}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = self.depth_files[idx]

        rgb_img = cv2.imread(rgb_path)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_img is None or depth_img is None:
            raise RuntimeError(f"Could not read image or depth map at {rgb_path} or {depth_path}")

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
     
        rgb_img = transforms.ToTensor()(rgb_img)
        depth_img = transforms.ToTensor()(depth_img)

        resize_rgb = transforms.Resize((352, 704), interpolation=InterpolationMode.BILINEAR, antialias=True)
        resize_depth = transforms.Resize((352, 704), interpolation=InterpolationMode.BILINEAR, antialias=True)
        rgb_img = resize_rgb(rgb_img)
        depth_img = resize_depth(depth_img)
        
        return rgb_img, depth_img

# train_dataset = NYUv2RawDataset(data_root, split='train')
# test_dataset = NYUv2RawDataset(data_root, split='test')

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # Example iteration:
# for batch_rgb, batch_depth in train_loader:
#     print("RGB Batch shape:", batch_rgb.shape)
#     print("Depth Batch shape:", batch_depth.shape)
#     print("Train items:", len(train_dataset))
#     print("Test items:", len(test_dataset))
#     break


# def data_attributes():
#     maximum_depth = []
        
#     for _, depth_img in test_dataset:
#         depth_map = cv2.imread(depth_img)
#         max_depth = depth_map.max()
#         maximum_depth.append(max_depth)
            
#     print(f"Maximum Depth Value is {max(maximum_depth)}")
    
# data_attributes()



# class ImageTransformer:
#     def __init__(self, min_depth=0.0, max_depth=10.0):
#         self.MIN_DEPTH = 1e-8
#         self.MAX_DEPTH = 10

#     def rotate_90(self, img, k):
#         # Rotate image by 90*k degrees clockwise without interpolation
#         # k in {0,1,2,3}
#         if k == 0:
#             return img
#         elif k == 1:
#             # 90 deg clockwise = transpose + horizontal flip
#             return torch.flip(img.transpose(-2, -1), dims=[-1])
#         elif k == 2:
#             # 180 deg clockwise = vertical flip + horizontal flip
#             return torch.flip(torch.flip(img, dims=[-2]), dims=[-1])
#         elif k == 3:
#             # 270 deg clockwise = transpose + vertical flip
#             return torch.flip(img.transpose(-2, -1), dims=[-2])
#         else:
#             raise ValueError("Rotation k must be in {0,1,2,3}")

#     def inverse_rotate_90(self, img, k):
#         # Inverse rotation by 90*k degrees clockwise is rotation by 90*(4 - k) mod 4
#         return self.rotate_90(img, (4 - k) % 4)

#     def translate(self, img, tx, ty):
#         # Integer pixel translation (tx, ty) with zero padding, no interpolation
#         # img shape: C,H,W
#         C, H, W = img.shape
#         shifted = torch.zeros_like(img)

#         src_x_start = max(0, -tx)
#         src_x_end = W - max(0, tx)
#         tgt_x_start = max(0, tx)
#         tgt_x_end = W - max(0, -tx)

#         src_y_start = max(0, -ty)
#         src_y_end = H - max(0, ty)
#         tgt_y_start = max(0, ty)
#         tgt_y_end = H - max(0, -ty)

#         shifted[:, tgt_y_start:tgt_y_end, tgt_x_start:tgt_x_end] = img[:, src_y_start:src_y_end, src_x_start:src_x_end]

#         return shifted

#     def transform_image(self, input_img):
#         B, C, H, W = input_img.shape
#         assert H == 352 and W == 704, "This transform assumes image size 352x704"

#         # Random rotation by multiples of 90 degrees (0, 90, 180, 270)
#         k = torch.randint(0, 4, (1,)).item()

#         flip_horizontal = torch.rand(1).item() > 0.5
#         flip_vertical = torch.rand(1).item() > 0.5

#         max_dx = int(0.10 * W)
#         max_dy = int(0.10 * H)
#         translate_x = torch.randint(-max_dx, max_dx + 1, (1,)).item()
#         translate_y = torch.randint(-max_dy, max_dy + 1, (1,)).item()

#         transformed = []
#         for i in range(B):
#             img = input_img[i]

#             # Apply rotation
#             img = self.rotate_90(img, k)

#             # Apply flips
#             if flip_horizontal:
#                 img = torch.flip(img, dims=[-1])  # horizontal flip (width axis)
#             if flip_vertical:
#                 img = torch.flip(img, dims=[-2])  # vertical flip (height axis)

#             # Apply translation with zero padding
#             img = self.translate(img, translate_x, translate_y)

#             transformed.append(img)

#         transformed = torch.stack(transformed)

#         # Get the new height and width after rotation (H_rot, W_rot)
#         # This is the crucial part that needs to be updated.
#         _, _, H_rot, W_rot = transformed.shape

#         transform_params = {
#             "rotation_k": k,
#             "flip_horizontal": flip_horizontal,
#             "flip_vertical": flip_vertical,
#             "translate_x": translate_x,
#             "translate_y": translate_y,
#             "orig_size": (H_rot, W_rot) # Store the size AFTER rotation
#         }

#         return transformed, transform_params

#     def depth_inverse_transform(self, depth, transform_params):
#         B, C, H, W = depth.shape
#         # The assertion now correctly checks against the size after the initial rotation
#         assert (H, W) == transform_params["orig_size"], f"Depth size mismatch. Expected {transform_params['orig_size']}, got {(H,W)}"

#         k = transform_params["rotation_k"]
#         flip_horizontal = transform_params["flip_horizontal"]
#         flip_vertical = transform_params["flip_vertical"]
#         translate_x = transform_params["translate_x"]
#         translate_y = transform_params["translate_y"]

#         restored = []
#         for i in range(B):
#             d = depth[i]

#             # Inverse translation
#             d = self.translate(d, -translate_x, -translate_y)

#             # Inverse flips
#             if flip_vertical:
#                 d = torch.flip(d, dims=[-2])
#             if flip_horizontal:
#                 d = torch.flip(d, dims=[-1])

#             # Inverse rotation
#             d = self.inverse_rotate_90(d, k)

#             d = torch.clamp(d, self.MIN_DEPTH, self.MAX_DEPTH)

#             restored.append(d)

#         return torch.stack(restored)

class ImageTransformer:
    def __init__(self, min_depth=0.0, max_depth=10.0):
        self.MIN_DEPTH = 1e-8
        self.MAX_DEPTH = 10
    def transform_image(self, input_img):
        B, C, H, W = input_img.shape

        flip_horizontal = torch.rand(1).item() > 0.5
        flip_vertical = torch.rand(1).item() > 0.5

        transformed = []
        for i in range(B):
            img = input_img[i]

            if flip_horizontal:
                img = torch.flip(img, dims=[-1])  # horizontal flip
            if flip_vertical:
                img = torch.flip(img, dims=[-2])  # vertical flip

            transformed.append(img)

        transformed = torch.stack(transformed)

        transform_params = {
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "orig_size": (H, W)
        }

        return transformed, transform_params
    def depth_inverse_transform(self, depth, transform_params):
        B, C, H, W = depth.shape
        assert (H, W) == transform_params["orig_size"], \
            f"Depth size mismatch. Expected {transform_params['orig_size']}, got {(H, W)}"

        flip_horizontal = transform_params["flip_horizontal"]
        flip_vertical = transform_params["flip_vertical"]

        restored = []
        for i in range(B):
            d = depth[i]

            if flip_vertical:
                d = torch.flip(d, dims=[-2])
            if flip_horizontal:
                d = torch.flip(d, dims=[-1])

            d = torch.clamp(d, self.MIN_DEPTH, self.MAX_DEPTH)

            restored.append(d)

        return torch.stack(restored)
