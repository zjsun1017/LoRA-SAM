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
        """随机生成 num_points 个不同的点提示"""
        points = []
        for _ in range(num_points):
            valid = False
            while not valid:
                y, x = torch.randint(0, target.shape[1], (1,)).item(), torch.randint(0, target.shape[2], (1,)).item()
                if target[:, y, x].sum() > 0:  # 至少有一个掩码覆盖该点
                    valid = True
                    points.append([x, y])
        points = torch.tensor(points, dtype=torch.float32)
        return points

    def generate_gt_masks(target, points, num_masks=3):
        """为每个点提示生成一个 `3xHxW` 的 ground truth 掩码"""
        gt_masks = []
        for x, y in points:
            intersecting_masks = target[:, int(y), int(x)] > 0
            intersecting_indices = torch.where(intersecting_masks)[0]

            if len(intersecting_indices) == 0:  # 没有相交掩码
                gt_mask = torch.zeros((num_masks, *target.shape[1:]), dtype=target.dtype)
            else:
                # 按掩码大小排序
                sizes = [target[idx].sum().item() for idx in intersecting_indices]
                sorted_indices = [intersecting_indices[i] for i in torch.argsort(torch.tensor(sizes), descending=True)]

                # 构建 ground truth mask
                gt_mask = torch.zeros((num_masks, *target.shape[1:]), dtype=target.dtype)
                for i, idx in enumerate(sorted_indices[:num_masks]):
                    gt_mask[i] = target[idx]
            gt_masks.append(gt_mask)
        return torch.stack(gt_masks, dim=0)

    # 生成随机点提示
    points = get_random_points(target, num_points=16)
    # 为每个点提示生成 ground truth 掩码
    gt_masks = generate_gt_masks(target, points)
    return image, gt_masks, points

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
    import os
    import joblib
    import torch
    from tqdm import tqdm

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
            img, gt_masks, points = construct_input(image, target)
            # 保存到文件
            joblib.dump(
                [img.numpy(), gt_masks.numpy(), points.numpy()],
                '%s/sa1b%07i.pkl' % (Save_path, idx)
            )
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
