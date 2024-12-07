import glob
import os
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from pycocotools import mask as mask_utils

import json
import numpy as np
from tqdm import tqdm
import random
import joblib

input_transforms = transforms.Compose([
    transforms.Resize((160, 256)),
    transforms.ToTensor(),
])

target_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 256)),
])

def construct_input(image, target):
    def get_random_points(target, num_points=16):
        """从所有掩码中随机采样 num_points 个前景点"""
        points = []
        mask_y, mask_x = torch.where(target.sum(dim=0) > 0)  # 聚合所有掩码的前景区域
        idx = torch.randint(0, len(mask_y), (num_points,))
        points = [[mask_x[i].item(), mask_y[i].item()] for i in idx]
        return torch.tensor(points, dtype=torch.float32)

    def get_random_boxes(target, num_boxes=16):
        """生成 num_boxes 个加噪边界框，基于每个掩码的独立区域"""
        boxes = []
        num_masks = target.shape[0]  # 获取掩码数量

        for _ in range(num_boxes):
            # 随机选择一个掩码
            mask_idx = torch.randint(0, num_masks, (1,)).item()
            mask = target[mask_idx]

            # 获取该掩码的前景区域
            mask_y, mask_x = torch.where(mask > 0)
            if len(mask_y) == 0:
                boxes.append([0, 0, 0, 0])  # 没有前景时返回空框
                continue

            x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
            width, height = x2 - x1, y2 - y1

            # 添加噪声
            dx1, dy1 = torch.normal(0, 0.1 * width), torch.normal(0, 0.1 * height)
            dx2, dy2 = torch.normal(0, 0.1 * width), torch.normal(0, 0.1 * height)

            dx1, dy1, dx2, dy2 = torch.clamp(torch.tensor([dx1, dy1, dx2, dy2]), -20, 20)
            noisy_box = [
                max(0, x1 + dx1), max(0, y1 + dy1),
                min(target.shape[2], x2 + dx2), min(target.shape[1], y2 + dy2)
            ]
            boxes.append(noisy_box)

        return torch.tensor(boxes, dtype=torch.float32)

    def generate_gt_masks(target, prompts, num_masks=3, prompt_type="point"):
        """为每个点或边界框生成 3xHxW 的 ground truth 掩码"""
        gt_masks = []
        for prompt in prompts:
            if prompt_type == "point":
                x, y = prompt
                intersecting_masks = target[:, int(y), int(x)] > 0
            elif prompt_type == "box":
                x1, y1, x2, y2 = map(int, prompt)
                intersecting_masks = target[:, y1:y2, x1:x2].sum(dim=(1, 2)) > 0

            intersecting_indices = torch.where(intersecting_masks)[0]
            if len(intersecting_indices) == 0:
                gt_mask = torch.zeros((num_masks, *target.shape[1:]), dtype=target.dtype)
            else:
                sizes = [target[idx].sum().item() for idx in intersecting_indices]
                sorted_indices = [intersecting_indices[i] for i in torch.argsort(torch.tensor(sizes), descending=True)]
                gt_mask = torch.zeros((num_masks, *target.shape[1:]), dtype=target.dtype)
                for i, idx in enumerate(sorted_indices[:num_masks]):
                    gt_mask[i] = target[idx]
            gt_masks.append(gt_mask)
        return torch.stack(gt_masks, dim=0)

    # 生成随机点提示
    points = get_random_points(target, num_points=16)
    gt_masks_points = generate_gt_masks(target, points, prompt_type="point")

    # 生成随机边界框
    boxes = get_random_boxes(target, num_boxes=16)
    gt_masks_boxes = generate_gt_masks(target, boxes, prompt_type="box")

    return image, gt_masks_points, points, gt_masks_boxes, boxes


class SA1B_Dataset(torchvision.datasets.ImageFolder):
    """A data loader for the SA-1B Dataset from "Segment Anything" (SAM)
    This class inherits from :class:`~torchvision.datasets.ImageFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples

    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.imgs[index] # discard automatic subfolder labels
        image = self.loader(path)
        masks = json.load(open(f'{path[:-3]}json'))['annotations'] # load json masks
        target = []
        mask_count = 0

        for m in masks:
            # decode masks from COCO RLE format
            target.append(mask_utils.decode(m['segmentation']))
            mask_count+=1
            if mask_count>=17:
                break
        target = np.stack(target, axis=-1)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target[target > 0] = 1 # convert to binary masks

        return image, target
        


    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    Save_path = './train_data'
    os.makedirs(Save_path, exist_ok=True)

    path = './sa1b'
    SA1Bdataset = SA1B_Dataset(path, transform=input_transforms, target_transform=target_transforms)

    # 遍历数据集并生成 .pkl 文件
    for idx in tqdm(range(len(SA1Bdataset)), desc="Generating PKL Files"):
        try:
            # 加载数据
            image, target = SA1Bdataset[idx]
            # 构建输入
            img, gt_masks_points, points, gt_masks_boxes, boxes = construct_input(image, target)
            # 保存到文件
            joblib.dump(
                [img.numpy(), gt_masks_points.numpy(), points.numpy(), gt_masks_boxes.numpy(), boxes.numpy()],
                f'{Save_path}/sa1b{idx:07d}.pkl'
            )
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
