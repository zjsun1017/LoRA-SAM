import glob
import os
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from pycocotools import mask as mask_utils
import joblib

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

from torch.utils.data import Dataset
class SA1B_Dataset(Dataset):
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
    def __init__(self, path) -> None:
        super().__init__()
        self.data_dir = path
        self.files = glob.glob(os.path.join(path,'*.pkl'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        image, target, bbox = joblib.load('%s/sa1b%07i.pkl' % (self.data_dir, index))

        return image, target, bbox

    def __len__(self):
        return len(self.files)


input_reverse_transforms = transforms.Compose([
    transforms.ToPILImage(),
])

class PKL_Dataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(f"{data_dir}/*.pkl"))

    def __getitem__(self, index):
        # 加载预处理好的数据
        image, gt_masks_points, points, gt_masks_boxes, boxes = joblib.load(self.files[index])
        return (
            torch.tensor(image, dtype=torch.float32),  # 图像
            torch.tensor(gt_masks_points, dtype=torch.float32),  # 点提示的掩码
            torch.tensor(points, dtype=torch.float32),  # 点提示
            torch.tensor(gt_masks_boxes, dtype=torch.float32),  # 边界框的掩码
            torch.tensor(boxes, dtype=torch.float32)  # 边界框
        )

    def __len__(self):
        return len(self.files)

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
        LoRA_injection(self.lora_sam,['linear','conv','convT'], lora_rank, lora_scale)
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
        # targets = targets>0.5
        dice_loss = 2 * torch.sum(pred_mask * targets, dim=(-1, -2) ) / (torch.sum( pred_mask ** 2, dim=(-1, -2)) + torch.sum( targets ** 2, dim=(-1, -2) ) + 1e-5)
        dice_loss = (1.0 - dice_loss)
        dice_loss = torch.mean(dice_loss)
        return dice_loss

    @staticmethod
    def mask_focal_loss(prediction, targets, alpha, gamma):

        pred_mask = torch.sigmoid(prediction)
        # pred_mask = prediction
        fl_1 = -alpha * ( (1 - pred_mask[targets > .5]) ** gamma ) * \
            torch.log(pred_mask[targets > .5] + 1e-6)
        
        fl_2 = -(1-alpha) * ( (pred_mask[targets < .5]) ** gamma ) * \
            torch.log(1 - pred_mask[targets < .5] + 1e-6)
        
        return (torch.mean(fl_1) + torch.mean(fl_2))

    @staticmethod
    def iou_token_loss(iou_prediction, prediction, targets):
        def calculateIoU(pred, gt):
          intersect = (pred * gt).sum(dim=(-1, -2))
          union = pred.sum(dim=(-1, -2)) + gt.sum(dim=(-1, -2)) - intersect
          ious = intersect.div(union)
          return ious

        # pred_mask = prediction
        pred = prediction>0
        
        iou = calculateIoU(pred, targets)

        return (iou_prediction - iou) ** 2, iou

    def training_step(self, batch):
        self.lora_sam.train()

        # 从 batch 中加载数据
        images, gt_masks_points, points, gt_masks_boxes, boxes = batch  # DataLoader 提供的数据
        device = self.device

        # 随机选择点提示或边界框提示
        if random.random() < 0.5:  # 点提示
            prompts = points
            gt_masks = gt_masks_points
            prompt_type = "point"
        else:  # 边界框提示
            prompts = boxes
            gt_masks = gt_masks_boxes
            prompt_type = "box"

        # 构建 batched_input
        batched_input = []
        for j in range(len(images)):
            if prompt_type == "point":
                batched_input.append({
                    'image': images[j].to(device),
                    'original_size': images[j].shape[1:],
                    'point_coords': prompts[j].unsqueeze(1).to(device),
                    'point_labels': torch.ones(prompts[j].shape[0], 1).to(device),
                    'target': gt_masks[j].to(device),
                })
            elif prompt_type == "box":
                batched_input.append({
                    'image': images[j].to(device),
                    'original_size': images[j].shape[1:],
                    'boxes': prompts[j].to(device),
                    'target': gt_masks[j].to(device),
                })

        # 前向传播
        predictions = self(batched_input, multimask_output=False)
        pred_masks = torch.stack([pred['masks'] for pred in predictions], dim=0)  # [B, N_prompts, H, W]

        # 计算损失
        dice_loss = self.mask_dice_loss(pred_masks, gt_masks)
        focal_loss = self.mask_focal_loss(pred_masks, gt_masks, alpha=0.25, gamma=2)
        total_loss = dice_loss + focal_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        # 记录损失
        self.log("train_dice_loss", dice_loss.item(), prog_bar=True)
        self.log("train_focal_loss", focal_loss.item(), prog_bar=True)
        self.log("train_total_loss", total_loss.item(), prog_bar=True)

        return total_loss

    def iterative_training_step(self, batch):
        images, gt_masks, points, _, _ = batch
        device = self.device

        predictions = []
        for iteration in range(3):
            if iteration == 0:
                prompts = points
            else:
                # 从误差区域采样新点
                error_region = (gt_masks - predictions[-1]).abs()
                error_y, error_x = torch.where(error_region > 0)
                idx = torch.randint(0, len(error_y), (1,)).item()
                prompts = torch.tensor([[error_x[idx], error_y[idx]]], dtype=torch.float32).unsqueeze(1)

            batched_input = [{
                'image': images[j].to(device),
                'point_coords': prompts.to(device),
                'point_labels': torch.ones(prompts.shape[0], 1).to(device),
                'mask_inputs': predictions[-1] if iteration > 0 else None,
            } for j in range(len(images))]

            pred = self(batched_input, multimask_output=False)
            predictions.append(pred)

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        ...
        loss = ...
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)

    def construct_batched_input(self, batch):
        image, target, bbox = batch # Tensor
        device = self.device

        def get_point_from_target(target):
            points = []
            for mask in target:
                mask_y, mask_x = torch.where(mask > 0)
                selection = random.randint(0, mask_y.shape[0]-1)
                points.append(torch.tensor([mask_x[selection], mask_y[selection]]))

            points = torch.stack(points, dim=0)
            return points

        points = [get_point_from_target(tg) for tg in target]
        
        # pdb.set_trace()
        batch_input = [{
            'image':image[j].to(device)*255,
            'original_size':(160, 256),
            'boxes':bbox[j].to(device),
            'target': target[j].to(device)
        } for j in range(image.shape[0])] 

        batch_input2 = [{
            'image':image[j].to(device)*255,
            'original_size':(160, 256),
            'point_coords':points[j].to(device).to(torch.float32).unsqueeze(1),
            'point_labels' : torch.ones(points[j].shape[0]).to(device).to(torch.int).unsqueeze(1),
            'target': target[j].to(device)
        } for j in range(image.shape[0])] 
        # pdb.set_trace()

        return batch_input, batch_input2

from segment_anything import sam_model_registry
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device)
import copy
import torch.nn as nn
def downsample_inject(model):
    downsampled_sam = copy.deepcopy(model)
    

    _, H, W, _ = sam.image_encoder.pos_embed.shape
    downsampled_sam.image_encoder.pos_embed = nn.Parameter(sam.image_encoder.pos_embed.data[:, :H//4, :W//4, :])
    downsampled_sam.image_encoder.img_size = 256
    downsampled_sam.prompt_encoder.factor = 4
    downsampled_sam.prompt_encoder.image_embedding_size = (16, 16)
    downsampled_sam.prompt_encoder.input_image_size = (256,256)
    for param in downsampled_sam.parameters():
        param.requires_grad_(False)
    
    return downsampled_sam
    
# def collate_fn(batches):
#     batch_data = []
#     targets = []

#     for b in batches:
#         image, target = b
#         batch_data.append(image)

#         targets.append(target)

#     return batch_data, targets

def print_params(model):
    model_parameters = filter(lambda p: True, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total params: ", params)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("training params: ", params)

def visualize_batch(images, pred, target_points, target_boxes, gt_masks_points, gt_masks_boxes):
    def show_mask(mask, ax, random_color=False):
        color = np.random.random(3) if random_color else np.array([30/255, 144/255, 255/255])
        ax.imshow(mask, alpha=0.6, cmap='cool')

    def show_box(box, ax):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', linewidth=2, facecolor='none')
        ax.add_patch(rect)

    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))  # 3 columns

    for i in range(batch_size):
        # Image + Point Masks
        axes[i, 0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        for mask in gt_masks_points[i]:
            show_mask(mask.cpu().numpy(), axes[i, 0])
        axes[i, 0].set_title("GT Point Masks")

        # Image + Box Masks
        axes[i, 1].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        for mask in gt_masks_boxes[i]:
            show_mask(mask.cpu().numpy(), axes[i, 1])
        axes[i, 1].set_title("GT Box Masks")

        # Image + Bounding Boxes
        axes[i, 2].imshow(images[i].permute(1, 2, 0).cpu().numpy())
        for box in target_boxes[i]:
            show_box(box.cpu().numpy(), axes[i, 2])
        axes[i, 2].set_title("Bounding Boxes")

    plt.tight_layout()
    plt.show()

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

path = './train_data'

train_dataset = PKL_Dataset(path)
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)#,collate_fn=collate_fn)

sam_downsampled = downsample_inject(sam)
# pdb.set_trace()

model = MyFastSAM(sam_downsampled, 4, 1.0).to(device)
print_params(model)

if args.train:

    pbar = tqdm(range(args.epoch))
    total_step = len(train_loader)
    for i in pbar:
        for iter, data in enumerate(train_loader):
            fl, dl, iou_loss, iou, batch = model.training_step(data)
            if iter%200==0:

                writer.add_scalar('train/fl',fl,global_step=i*total_step+iter)
                writer.add_scalar('train/dl',dl,global_step=i*total_step+iter)
                writer.add_scalar('train/tl',iou_loss,global_step=i*total_step+iter)
                writer.add_scalar('train/iou',iou,global_step=i*total_step+iter)
                canvas = visualize_batch(batch['images'], batch['pred']>0, batch['target'], batch['prompt'])
                writer.add_images('train/viz', canvas, i*total_step+iter, dataformats='HWC')
            #     writer.add_images('train/pred', pred, epoch*total_step+iter, dataformats='HWC')
            #     writer.add_images('train/gt', gt, epoch*total_step+iter, dataformats='HWC')

        torch.save({
                'model_state_dict': model.state_dict(),
            }, '%s/model.pkl' % save_dir)