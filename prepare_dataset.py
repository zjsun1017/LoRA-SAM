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


def get_sorted_masks(target):
    """按掩码大小从大到小排序"""
    sizes = [mask.sum().item() for mask in target]
    sorted_indices = torch.argsort(torch.tensor(sizes), descending=True)
    return target[sorted_indices]

def pad_masks(masks, target_num):
    """
    填补 mask 数量不足的情况，用最大的 mask 填充
    :param masks: 当前的 mask 列表
    :param target_num: 目标数量
    :return: 填充后的 mask 列表
    """
    while len(masks) < target_num:
        masks.append(masks[len(masks) % len(masks)])
    return masks[:target_num]

def get_center_points(target, num_points=16):
    """
    获取每个掩码的中心点，最多选取 num_points 个
    :param target: 输入的目标掩码
    :return: 中心点和对应的 mask
    """
    points = []
    masks = []
    sorted_target = get_sorted_masks(target)

    for mask in sorted_target:
        mask_y, mask_x = torch.where(mask > 0)
        if len(mask_y) == 0:
            continue
        # 计算中心点
        center_x = (mask_x.min() + mask_x.max()) // 2
        center_y = (mask_y.min() + mask_y.max()) // 2
        points.append([center_x.item(), center_y.item()])
        masks.append(mask)
        if len(points) == num_points:
            break

    # 填补不足的点和 mask
    masks = pad_masks(masks, num_points)
    while len(points) < num_points:
        points.append(points[len(points) % len(points)])

    return torch.tensor(points, dtype=torch.float32), torch.stack(masks, dim=0)

def get_random_boxes(target, num_boxes=16):
    """
    从前 num_boxes 个最大掩码中生成边界框，并添加扰动
    :param target: 输入的目标掩码
    :return: 边界框和对应的 mask
    """
    boxes = []
    masks = []
    sorted_target = get_sorted_masks(target)

    for mask in sorted_target:
        mask_y, mask_x = torch.where(mask > 0)
        if len(mask_y) == 0:
            continue

        x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
        width, height = x2 - x1, y2 - y1

        # 加扰动
        dx1, dy1 = torch.normal(0, 0.1 * width), torch.normal(0, 0.1 * height)
        dx2, dy2 = torch.normal(0, 0.1 * width), torch.normal(0, 0.1 * height)

        dx1, dy1, dx2, dy2 = torch.clamp(torch.tensor([dx1, dy1, dx2, dy2]), -20, 20)
        noisy_box = [
            max(0, x1 + dx1.item()),
            max(0, y1 + dy1.item()),
            min(target.shape[2], x2 + dx2.item()),
            min(target.shape[1], y2 + dy2.item()),
        ]
        boxes.append(noisy_box)
        masks.append(mask)
        if len(boxes) == num_boxes:
            break

    # 填补不足的边界框和 mask
    masks = pad_masks(masks, num_boxes)
    while len(boxes) < num_boxes:
        boxes.append(boxes[len(boxes) % len(boxes)])

    return torch.tensor(boxes, dtype=torch.float32), torch.stack(masks, dim=0)

def construct_input(image, target):
    """
    构造输入，包括点、边界框和对应的掩码
    :param image: 输入图像
    :param target: 输入目标掩码
    :return: 构造好的输入
    """
    # 获取点提示
    points, gt_masks_points = get_center_points(target)

    # 获取边界框
    boxes, gt_masks_boxes = get_random_boxes(target)

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
