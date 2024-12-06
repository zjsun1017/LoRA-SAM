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

def construct_input(batch):
        image, target = batch # lists
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
            # pdb.set_trace()
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

        return image, new_target, bbox

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
        return iou


    def training_step(self, batch):
        self.lora_sam.train()
        images = batch[0]
        bbox = batch[2]
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

        # pdb.set_trace()
        # loss = 0.1 * self.mask_dice_loss(pred, target) + self.mask_focal_loss(pred,target,0.25,2)
        # loss.backward()
        dl = self.mask_dice_loss(pred, target)
        fl = self.mask_focal_loss(pred,target,0.25,2)
        loss = fl + 0.1*dl
        loss.backward()
        self.optimizer.step()

        iou_loss = self.iou_token_loss(None, pred, target)

        batch_prediction = {
            'images':images.cpu(),
            'target':target.cpu(),
            'bbox':bbox.cpu(),
            'pred':pred.detach().cpu(),
        }
        # self.log('train_loss', loss.item(), prog_bar=True)

        # During training, we backprop only the minimum loss over the 3 output masks.
        # sam paper main text Section 3
        return fl.item(), dl.item(), torch.mean(iou_loss).item(), batch_prediction

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        ...
        loss = ...
        # use same procedure as training, monitor the loss
        self.log('val_loss', loss, prog_bar=True)

    def construct_batched_input(self, batch):
        image, target, bbox = batch # Tensor
        device = self.device
        
        # pdb.set_trace()
        batch_input = [{
            'image':image[j].to(device),
            'original_size':(160, 256),
            'boxes':bbox[j].to(device),
            'target': target[j].to(device)
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
    downsampled_sam = copy.deepcopy(model)
    for param in downsampled_sam.parameters():
        param.requires_grad_(False)

    _, H, W, _ = sam.image_encoder.pos_embed.shape
    downsampled_sam.image_encoder.pos_embed = nn.Parameter(sam.image_encoder.pos_embed.data[:, :H//4, :W//4, :])
    downsampled_sam.image_encoder.img_size = 256
    downsampled_sam.prompt_encoder.factor = 4
    downsampled_sam.prompt_encoder.image_embedding_size = (16, 16)
    downsampled_sam.prompt_encoder.input_image_size = (256,256)
    
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

def visualize_batch(images, pred, target, bbox):
    def show_mask(mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import numpy as np

    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))  # 3 columns: image+GT, image+pred, image+bbox

    for i in range(batch_size):
        # Plot image + ground truth
        ax_gt = axes[i, 0] if batch_size > 1 else axes[0]
        ax_gt.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Assuming images are [C, H, W]
        ax_gt.set_title(f"Image + GT (Sample {i})")
        for mask in target[i]:
            show_mask(mask.cpu().numpy(), ax_gt)
        # ax_gt.axis('off')

        # Plot image + prediction
        ax_pred = axes[i, 1] if batch_size > 1 else axes[1]
        # ax_pred.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        ax_pred.set_title(f"Image + Pred (Sample {i})")
        for mask in pred[i]:
            show_mask(mask.cpu().numpy(), ax_pred)
        # ax_pred.axis('off')

        # Plot image + bounding boxes
        ax_bbox = axes[i, 2] if batch_size > 1 else axes[2]
        ax_bbox.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        ax_bbox.set_title(f"Image + BBox (Sample {i})")
        for box in bbox[i]:
            x1, y1, x2, y2 = box.cpu().numpy()
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='green', facecolor='none')
            ax_bbox.add_patch(rect)
        # ax_bbox.axis('off')

    plt.tight_layout()

    # Convert the Matplotlib figure to a NumPy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    array = array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the figure to free memory
    return array




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
SA1Bdataset = SA1B_Dataset(path)
train_loader = DataLoader(SA1Bdataset,batch_size=16,shuffle=True)#,collate_fn=collate_fn)

sam_downsampled = downsample_inject(sam)
# pdb.set_trace()

model = MyFastSAM(sam_downsampled, 4, 1.0).to(device)
print_params(model)

if args.train:

    pbar = tqdm(range(args.epoch))
    total_step = len(train_loader)
    for i in pbar:
        for iter, data in enumerate(train_loader):
            fl, dl, iou, batch = model.training_step(data)
            if iter%100==0:

                writer.add_scalar('train/fl',fl,global_step=i*total_step+iter)
                writer.add_scalar('train/dl',dl,global_step=i*total_step+iter)
                writer.add_scalar('train/iou',iou,global_step=i*total_step+iter)
                canvas = visualize_batch(batch['images'], batch['pred']>0, batch['target'], batch['bbox'])
                writer.add_images('train/viz', canvas, i*total_step+iter, dataformats='HWC')
            #     writer.add_images('train/pred', pred, epoch*total_step+iter, dataformats='HWC')
            #     writer.add_images('train/gt', gt, epoch*total_step+iter, dataformats='HWC')

        torch.save({
                'model_state_dict': model.state_dict(),
            }, '%s/model.pkl' % save_dir)