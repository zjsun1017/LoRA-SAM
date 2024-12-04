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

input_transforms = transforms.Compose([
    transforms.Resize((160, 256)),
    transforms.ToTensor(),
])

target_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((160, 256)),
])

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

        for m in masks:
            # decode masks from COCO RLE format
            target.append(mask_utils.decode(m['segmentation']))
        target = np.stack(target, axis=-1)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target[target > 0] = 1 # convert to binary masks

        return image, target

    def __len__(self):
        return len(self.imgs)


input_reverse_transforms = transforms.Compose([
    transforms.ToPILImage(),
])

import matplotlib.pyplot as plt
def show_image(image, target, row=12, col=12):
    # image: numpy image
    # target: mask [N, H, W]
    fig, axs = plt.subplots(row, col, figsize=(20, 12))
    for i in range(row):
        for j in range(col):
            if i*row+j < target.shape[0]:
                axs[i, j].imshow(image)
                axs[i, j].imshow(target[i*row+j], alpha=0.5)
            else:
                axs[i, j].imshow(image)
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()


from segment_anything.modeling.sam import Sam
import pytorch_lightning as pl
import random
import pdb
from loralib.lora import LoRA_injection


class MyFastSAM(pl.LightningModule):
    def __init__(self, orig_sam: Sam, lora_rank: int, lora_scale: float):
        super().__init__()
        self.lora_sam = orig_sam
        LoRA_injection(self.lora_sam,['linear','conv'], lora_rank, lora_scale)
        self.configure_optimizers()
        

    def forward(self, *args, **kwargs):
        """
        comments imported from original SAM code

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        return self.lora_sam(*args, **kwargs)

    def configure_optimizers(self):
        lora_parameters = [param for param in self.parameters() if param.requires_grad]
        # make sure original sam don't requires_grad
        self.optimizer = torch.optim.AdamW(lora_parameters, lr=1e-5)

    @staticmethod
    def mask_dice_loss(prediction, targets):

        pred_mask = torch.sigmoid(prediction)
        dice_loss = 2 * torch.sum(pred_mask * targets, dim=(-1, -2) ) / (torch.sum( pred_mask ** 2, dim=(-1, -2)) + torch.sum( targets ** 2, dim=(-1, -2) ) + 1e-5)
        dice_loss = (1 - dice_loss)
        dice_loss = torch.mean(dice_loss)
        return dice_loss

    @staticmethod
    def mask_focal_loss(prediction, targets, alpha, gamma):

        pred_mask = torch.sigmoid(prediction)
        fl_1 = -alpha * ( (1 - pred_mask[targets > .5]) ** gamma ) * \
            torch.log(pred_mask[targets > .5] + 1e-6)
        
        fl_2 = -(1-alpha) * ( (pred_mask[targets < .5]) ** gamma ) * \
            torch.log(1 - pred_mask[targets < .5] + 1e-6)
        
        return (torch.mean(fl_1) + torch.mean(fl_2))

    @staticmethod
    def iou_token_loss(iou_prediction, prediction, targets):
        pass

    def training_step(self, batch):
        self.lora_sam.train()
        batched_input = self.construct_batched_input(batch)
        device = self.device
        # 1a. single point prompt training
        # 1b. iterative point prompt training up to 3 iteration
        # 2. box prompt training, only 1 iteration
        self.optimizer.zero_grad()
        predictions = self(batched_input, multimask_output = False
        ) 
        pred = [predictions[j]['masks'] for j in range(len(predictions))]
        pred = torch.stack(pred, dim=0)
        pred = pred.squeeze(2)
        target = [batched_input[j]['target'] for j in range(len(batched_input))]
        target = torch.stack(target, dim=0)
        loss = 0.01 * self.mask_dice_loss(pred, target) + self.mask_focal_loss(pred,target,0.25,2)
        loss.backward()

        self.optimizer.step()
        # self.log('train_loss', loss.item(), prog_bar=True)

        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return loss.item()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        ...
        loss = ...
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)

    def construct_batched_input(self, batch):
        image, target = batch # lists
        device = self.device

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
        
        image = torch.stack(image,dim=0)

        new_target = [load_num_masks(tg,16) for tg in target]
        new_target = torch.stack(new_target,dim=0)
        
        bbox = [get_bbox_from_target(tg) for tg in new_target]
        bbox = torch.stack(bbox,dim=0)

        assert bbox.shape[0] == image.shape[0]
        
        batch_input = [{
            'image':image[j].to(device),
            'original_size':(160, 256),
            'boxes':bbox[j].to(device),
            'target': new_target[j].to(device)
        } for j in range(image.shape[0])] 

        return batch_input

from segment_anything import sam_model_registry
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device)
import copy
import torch.nn as nn
def downsample_inject(model):
    downsampled_sam = copy.deepcopy(sam)
    for param in downsampled_sam.parameters():
        param.requires_grad_(False)

    _, H, W, _ = sam.image_encoder.pos_embed.shape
    downsampled_sam.image_encoder.pos_embed = nn.Parameter(sam.image_encoder.pos_embed.data[:, :H//4, :W//4, :])
    downsampled_sam.image_encoder.img_size = 256
    downsampled_sam.prompt_encoder.factor = 4
    downsampled_sam.prompt_encoder.image_embedding_size = (16, 16)
    
    return downsampled_sam
    


def collate_fn(batches):
    batch_data = []
    targets = []

    for b in batches:
        image, target = b
        batch_data.append(image)

        targets.append(target)

    return batch_data, targets

def print_params(model):
  model_parameters = filter(lambda p: True, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print("total params: ", params)
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print("training params: ", params)

from tensorboardX import SummaryWriter
import argparse
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--data_dir', type=str, default='/home/rex/Desktop/github/EventMonoPose/baseline/Eventhpe/data')
parser.add_argument('--result_dir', type=str, default='./exp')
parser.add_argument('--save_dir', type=str, default='./model')
parser.add_argument('--log_dir', type=str, default='optimize')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args("")

os.makedirs(args.save_dir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
dtype = torch.float32

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))
save_dir = '%s/%s/%s' % (args.result_dir, args.log_dir, start_time)

path = './sa1b'
SA1Bdataset = SA1B_Dataset(path, transform=input_transforms, target_transform=target_transforms)
train_loader = DataLoader(SA1Bdataset,batch_size=16,shuffle=True,collate_fn=collate_fn)

sam_downsampled = downsample_inject(sam)


model = MyFastSAM(sam_downsampled, 4, 1.0).to(device)
print_params(model)

if args.train:

    pbar = tqdm(range(args.epoch))
    total_step = len(train_loader)
    for i in pbar:
        for iter, data in enumerate(train_loader):
            loss = model.training_step(data)
            if iter%1==0:

                writer.add_scalar('train/loss',loss,global_step=i*total_step+iter)
            #     writer.add_images('train/pred', pred, epoch*total_step+iter, dataformats='HWC')
            #     writer.add_images('train/gt', gt, epoch*total_step+iter, dataformats='HWC')

        torch.save({
                'model_state_dict': model.state_dict(),
            }, '%s/model.pkl' % save_dir)