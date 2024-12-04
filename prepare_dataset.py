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
        # Lists input
        def load_num_masks(target, NUM_MASK_PER_IMG):
            num_masks = target.shape[0]
            selected = []
            if num_masks >= NUM_MASK_PER_IMG :
                all_mask_index = np.arange(num_masks)
                np.random.shuffle(all_mask_index)
                select_mask_indices = all_mask_index[:NUM_MASK_PER_IMG]
            else:
                select_mask_indices = np.arange(num_masks)
                for _ in range(NUM_MASK_PER_IMG-num_masks):
                  select_mask_indices = np.append(select_mask_indices, select_mask_indices[-1])

            # Select only 
            for ind in select_mask_indices:
                m = target[ind]
                # decode masks from COCO RLE format
                selected.append(m)

            target = torch.stack(selected, dim=0)
            return target

        def get_bbox_from_target(target):
            bbox = []
            for mask in target:
                mask_y, mask_x = torch.where(mask > 0)
                x1, y1, x2, y2 = mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                w = (x2 - x1)
                h = (y2 - y1)
                delta_w = min(random.random() * 0.2 * w, 20)
                delta_h = min(random.random() * 0.2 * h, 20)

                x1, y1, x2, y2  = center_x - (w + delta_w) / 2, center_y - (h + delta_h) / 2, \
                                    center_x + (w + delta_w) / 2, center_y + (h + delta_h) / 2
                bbox.append(torch.tensor([x1, y1, x2, y2]))

            bbox = torch.stack(bbox, dim=0)
            return bbox

        new_target = load_num_masks(target,16)
        
        bbox = get_bbox_from_target(new_target)

        return image, new_target, bbox

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
    Save_path = './train_data'
    import pdb

    path = './sa1b'
    SA1Bdataset = SA1B_Dataset(path, transform=input_transforms, target_transform=target_transforms)
    for idx in range(len(SA1Bdataset)):
        image, target = SA1Bdataset[idx]
        img, ntg, bbox = construct_input(image, target)
        joblib.dump([img.numpy(), ntg.numpy(), bbox.numpy()],'%s/sa1b%07i.pkl' % (Save_path, idx))
        print('Preparing data ',idx , 'out of', len(SA1Bdataset))
    
