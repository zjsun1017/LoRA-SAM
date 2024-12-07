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
    def get_sorted_masks(target):
        """按掩码大小从大到小排序"""
        sizes = [mask.sum().item() for mask in target]
        sorted_indices = torch.argsort(torch.tensor(sizes), descending=True)
        return target[sorted_indices]

    def get_random_points(target, num_points=16):
        """从前 16 个最大掩码中采样 num_points 个点"""
        points = []
        sorted_target = get_sorted_masks(target)[:16]  # 获取前 16 个最大掩码

        for mask in sorted_target:
            mask_y, mask_x = torch.where(mask > 0)
            if len(mask_y) == 0:  # 跳过无前景的掩码
                continue
            for _ in range(1):  # 每个掩码选一个点
                idx = torch.randint(0, len(mask_y), (1,)).item()
                points.append([mask_x[idx].item(), mask_y[idx].item()])
            if len(points) == num_points:
                break

        # 如果点数不足，重复已有点
        while len(points) < num_points:
            points.append(points[len(points) % len(points)])
        return torch.tensor(points, dtype=torch.float32)

    def get_random_boxes(target, num_boxes=16):
        """从前 16 个最大掩码中生成 num_boxes 个边界框"""
        boxes = []
        sorted_target = get_sorted_masks(target)[:16]  # 获取前 16 个最大掩码

        for mask in sorted_target:
            mask_y, mask_x = torch.where(mask > 0)
            if len(mask_y) == 0:  # 跳过无前景的掩码
                continue

            x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
            width, height = x2 - x1, y2 - y1

            # 为当前掩码生成一个加噪的边界框
            dx1, dy1 = torch.normal(0, 0.1 * width), torch.normal(0, 0.1 * height)
            dx2, dy2 = torch.normal(0, 0.1 * width), torch.normal(0, 0.1 * height)

            dx1, dy1, dx2, dy2 = torch.clamp(torch.tensor([dx1, dy1, dx2, dy2]), -20, 20)
            noisy_box = [
                max(0, x1 + dx1), max(0, y1 + dy1),
                min(target.shape[2], x2 + dx2), min(target.shape[1], y2 + dy2)
            ]
            boxes.append(noisy_box)
            if len(boxes) == num_boxes:
                break

        # 如果边界框数量不足，重复已有边界框
        while len(boxes) < num_boxes:
            boxes.append(boxes[len(boxes) % len(boxes)])
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
                sorted_indices = torch.argsort(torch.tensor(sizes), descending=True)

                # 获取相交掩码
                intersecting_masks = [target[intersecting_indices[idx]] for idx in sorted_indices]

                # 如果掩码不足 3 个，重复最大的掩码
                while len(intersecting_masks) < num_masks:
                    intersecting_masks.append(intersecting_masks[0])  # 重复最大的掩码

                # 只保留前 num_masks 个掩码
                intersecting_masks = intersecting_masks[:num_masks]

                # 构建 ground truth 掩码
                gt_mask = torch.stack(intersecting_masks, dim=0)

            gt_masks.append(gt_mask)

        return torch.stack(gt_masks, dim=0)

    # 按掩码大小排序 target
    target = get_sorted_masks(target)

    # 生成随机点提示
    points = get_random_points(target, num_points=16)
    gt_masks_points = generate_gt_masks(target, points, prompt_type="point")

    # 生成随机边界框
    boxes = get_random_boxes(target, num_boxes=16)
    gt_masks_boxes = generate_gt_masks(target, boxes, prompt_type="box")

    return image, gt_masks_points, points, gt_masks_boxes, boxes

    # 按掩码大小排序 target
    target = get_sorted_masks(target)

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
