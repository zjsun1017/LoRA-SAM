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



# def construct_input(image, target):
#     # Function to select a specific number of masks from the target
#     def load_num_masks(target, NUM_MASK_PER_IMG):
#         num_masks = target.shape[0]
#         selected = []
#         if num_masks >= NUM_MASK_PER_IMG:
#             # Sort masks by size and select top NUM_MASK_PER_IMG masks
#             mask_size = torch.count_nonzero(target, dim=(-2, -1))
#             select_mask_indices = torch.argsort(mask_size, descending=True)[:NUM_MASK_PER_IMG].numpy()
#         else:
#             # If fewer masks than needed, repeat indices for missing masks
#             select_mask_indices = np.arange(num_masks)
#             for _ in range(NUM_MASK_PER_IMG - num_masks):
#                 select_mask_indices = np.append(select_mask_indices, select_mask_indices[-1])
#
#         # Select masks based on the calculated indices
#         for ind in select_mask_indices:
#             m = target[ind]
#             selected.append(m)
#
#         target = torch.stack(selected, dim=0)
#         return target
#
#     # Function to generate bounding boxes from the masks in the target
#     def get_bbox_from_target(target):
#         bbox = []
#         for mask in target:
#             mask_y, mask_x = torch.where(mask > 0)
#             x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
#
#             center_x = (x1 + x2) / 2
#             center_y = (y1 + y2) / 2
#             w = (x2 - x1)
#             h = (y2 - y1)
#             delta_w = min(random.random() * 0.2 * w, 20)
#             delta_h = min(random.random() * 0.2 * h, 20)
#
#             x1, y1, x2, y2 = center_x - (w + delta_w) / 2, center_y - (h + delta_h) / 2, \
#                              center_x + (w + delta_w) / 2, center_y + (h + delta_h) / 2
#             bbox.append(torch.tensor([x1, y1, x2, y2]))
#
#         bbox = torch.stack(bbox, dim=0)
#         return bbox
#
#     # Function to generate random points from the target masks
#     def get_random_points(target, num_points=16):
#         """Randomly generates a specified number of valid points."""
#         points = []
#         for _ in range(num_points):
#             valid = False
#             while not valid:
#                 y, x = torch.randint(0, target.shape[1], (1,)).item(), torch.randint(0, target.shape[2], (1,)).item()
#                 if target[:, y, x].sum() > 0:  # Ensure at least one mask covers the point
#                     valid = True
#                     points.append([x, y])
#         points = torch.tensor(points, dtype=torch.float32)
#         return points
#
#     # Function to generate ground truth masks for given points
#     def generate_gt_masks(target, points, num_masks=3):
#         """Generates ground truth masks for each point with specified number of masks."""
#         gt_masks = []
#         for x, y in points:
#             intersecting_masks = target[:, int(y), int(x)] > 0
#             intersecting_indices = torch.where(intersecting_masks)[0]
#
#             if len(intersecting_indices) == 0:  # No intersecting masks
#                 gt_mask = torch.zeros((num_masks, *target.shape[1:]), dtype=target.dtype)
#             else:
#                 # Sort masks by size and select top masks
#                 sizes = [target[idx].sum().item() for idx in intersecting_indices]
#                 sorted_indices = [intersecting_indices[i] for i in torch.argsort(torch.tensor(sizes), descending=True)]
#
#                 # Construct ground truth mask
#                 gt_mask = torch.zeros((num_masks, *target.shape[1:]), dtype=target.dtype)
#                 for i, idx in enumerate(sorted_indices[:num_masks]):
#                     gt_mask[i] = target[idx]
#             gt_masks.append(gt_mask)
#         return torch.stack(gt_masks, dim=0)
#
#     # Generate random points
#     points = get_random_points(target, num_points=16)
#     # Generate ground truth masks for the random points
#     gt_masks = generate_gt_masks(target, points)
#     bbox = None  # Bounding boxes can be generated using get_bbox_from_target if needed
#
#     return image, gt_masks, bbox, points

def get_sorted_masks(target):
    """
    Sort masks by their size in descending order.
    :param target: List of input masks.
    :return: Masks sorted by size.
    """
    sizes = [mask.sum().item() for mask in target]
    sorted_indices = torch.argsort(torch.tensor(sizes), descending=True)
    return target[sorted_indices]

def pad_masks(masks, target_num):
    """
    Pad the mask list to the target number of masks by repeating the largest mask.
    :param masks: Current list of masks.
    :param target_num: Target number of masks.
    :return: Padded list of masks.
    """
    while len(masks) < target_num:
        masks.append(masks[len(masks) % len(masks)])
    return masks[:target_num]

def get_center_points(target, num_points=16):
    """
    Get the center points for each mask, up to a specified number of points.
    :param target: Input target masks.
    :param num_points: Maximum number of points to select.
    :return: Center points and their corresponding masks.
    """
    points = []
    masks = []
    sorted_target = get_sorted_masks(target)

    for mask in sorted_target:
        mask_y, mask_x = torch.where(mask > 0)
        if len(mask_y) == 0:
            continue
        # Calculate center point
        center_x = (mask_x.min() + mask_x.max()) // 2
        center_y = (mask_y.min() + mask_y.max()) // 2
        points.append([center_x.item(), center_y.item()])
        masks.append(mask)
        if len(points) == num_points:
            break

    # Pad missing points and masks
    masks = pad_masks(masks, num_points)
    while len(points) < num_points:
        points.append(points[len(points) % len(points)])

    return torch.tensor(points, dtype=torch.float32), torch.stack(masks, dim=0)

def get_random_boxes(target, num_boxes=16):
    """
    Generate bounding boxes from the top masks with random perturbations.
    :param target: Input target masks.
    :param num_boxes: Number of bounding boxes to generate.
    :return: Bounding boxes and their corresponding masks.
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

        # Add noise to the bounding box
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

    # Pad missing boxes and masks
    masks = pad_masks(masks, num_boxes)
    while len(boxes) < num_boxes:
        boxes.append(boxes[len(boxes) % len(boxes)])

    return torch.tensor(boxes, dtype=torch.float32), torch.stack(masks, dim=0)

def construct_input(image, target):
    """
    Construct input, including points, bounding boxes, and their corresponding masks.
    :param image: Input image.
    :param target: Input target masks.
    :return: Constructed input data.
    """
    # Get point prompts
    points, gt_masks_points = get_center_points(target)

    # Get bounding boxes
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


if __name__=="__main__":
    Save_path = 'E:\data\lora_sam'
    import pdb

    path = './sa1b'
    SA1Bdataset = SA1B_Dataset(path, transform=input_transforms, target_transform=target_transforms)
    for idx in range(len(SA1Bdataset)):
        image, target = SA1Bdataset[idx]
        image, gt_masks_points, points, gt_masks_boxes, boxes = construct_input(image, target)
        joblib.dump([image.numpy(), gt_masks_points.numpy(), points.numpy(), gt_masks_boxes.numpy(), boxes.numpy()],'%s/sa1b%07i.pkl' % (Save_path, idx))
        print('Preparing data ',idx , 'out of', len(SA1Bdataset))
    
