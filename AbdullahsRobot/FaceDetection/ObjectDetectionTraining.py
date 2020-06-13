"""Object Detection Training Program

    This is a module for managing object detection training.

    @author : Muhammad Sakti Alvissalim (alvissalim@gmail.com)

    Copyright (C) 2020  Muhammad Sakti Alvissalim 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from ObjectDetectionModel import ObjectDetectionModel
from ObjectDetectionDataset import ObjectDetectionDataset, object_dataset_collate_fn, BoundingBox
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from typing import Type, List, NamedTuple
import math
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from datetime import datetime
import os

import torchvision.transforms.functional as TF
from torchvision.transforms import RandomApply
import random

from  torchvision.transforms import ColorJitter
import torchvision.transforms as transforms

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import List, Set, Dict, Tuple, Optional, NamedTuple, Type

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]


# Transforms which applies only to image
preprocessing = transforms.Compose([
    RandomApply([ColorJitter(0.25, 0.25, 0.25, 0.25)], 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std)
])

def np_image_from_tensor(tensor_image : Type[torch.FloatTensor]) -> Type[np.array]:
    """Convert normalized tensor to numpy array

    Args:
        tensor_image (Type[tensor.FloatTensor]): [description]

    Returns:
        Type[np.array]: [description]
    """
    inverse_mean = [-m/s for m, s in zip(image_mean, image_std)]
    inverse_std = [1/s for s in image_std]
    unnormalized_img = TF.normalize(tensor_image, inverse_mean, inverse_std)
    pil_image = TF.to_pil_image(unnormalized_img.cpu())
    np_img = np.array(pil_image)

    return np_img

def worker_init_fn(worker_id):     
    """Init function to ensure different seed for every run """                                                     
    np.random.seed()

def image_bb_transforms(image : Type[Image.Image], bboxes : List[BoundingBox]) -> Tuple[Type[torch.FloatTensor], List[BoundingBox]]:
    """Transform image and bboxes

    Args:
        image (Type[Image]): Pillow image
        bboxes (List[BoundingBox]): List of BoundingBox NamedTuple objects for the image

    Returns:
        Type[torch.FloatTensor], List[BoundingBox] : Transformed data
    """
    # No transform as default
    transformed_bboxes = bboxes
    transformed_image = image

    # Only distort the image half of the time
    if random.random() > 0.5:
        # Get a random translation
        translate = np.random.uniform(low=-0.4, high=0.4, size = 2)
        # Get random scaling factor
        scale = np.random.uniform(low=0.25, high=1.0, size=1)
        
        transformed_bboxes = []

        for bb in bboxes:
            # Implicit convertion from Namedtuple to 
            scaled_bb = np.array(list(bb))

            # Scale the bb
            center_offset = ( scaled_bb - 0.5 )
            scaled_offset = center_offset * scale
            scaled_bb = 0.5 + scaled_offset

            # Translate the bb
            scaled_bb[[0,2]] = scaled_bb[[0,2]] + translate[0]
            scaled_bb[[1,3]] = scaled_bb[[1,3]] + translate[1]

            # Clip the bb
            scaled_bb = np.clip(scaled_bb, 0.0, 1.0).reshape(-1)

            transformed_bboxes.append(BoundingBox(*scaled_bb.tolist()))

        # Translate to pixel coordinate
        translate[0] *= image.width
        translate[1] *= image.height

        # Random color to fill the border
        random_color = np.random.randint(0, 255, size=3).tolist()

        transformed_image = TF.affine(image, 0, translate.tolist(), scale, 0, resample=Image.BILINEAR, fillcolor=tuple(random_color))

    # Apply the preprocessing
    transformed_image = preprocessing(transformed_image)

    # TODO : Implement more transforms

    return transformed_image, transformed_bboxes

class ObjectDetectionTrainer:
    """Object detection training manager.
    """
    def __init__(self, max_epoch : int, dataset_train_path : str, dataset_validation_path : str, output_grid_size : int, anchor_boxes : List[BoundingBox], target_device : str, label_map : Dict[str,int], target_image_size : Tuple[int,int] = (320,320), n_classes : int = 1):
        """Create a new ObjectDetectionTrainer

        Args:
            max_epoch (int): Maximum number of epoch to run the training.
            dataset_path (str): Path to the folder containing the json files.
            output_grid_size (int): Number of grids for the prediction. The model will output (output_grid_size, output_grid_size) cells.
            anchor_boxes (List[BoundingBox]): List of bounding box to be used as anchor.
            target_device (str): The target device of that Pytorch will use. Example ("cuda:0", "cpu:0")
            target_image_size (Tuple[int,int], optional): The expected size of the network input (width, height). Defaults to (320,320).
        """
        self.n_classes = n_classes
        self.max_epoch = max_epoch
        self.target_image_size = target_image_size
        self.dataset_train = ObjectDetectionDataset(dataset_train_path, target_image_size, transform=image_bb_transforms, label_map = label_map)
        self.dataset_validation = ObjectDetectionDataset(dataset_validation_path, target_image_size, transform=image_bb_transforms, label_map = label_map)
        self.boxes_per_cell = len(anchor_boxes)
        self.anchor_boxes = anchor_boxes
        self.output_grid_size = output_grid_size
        self.model = ObjectDetectionModel(n_classes, self.output_grid_size, self.boxes_per_cell)
        self.target_device = target_device
        self.model.to(self.target_device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.label_map = label_map

        # Timestamp to assign unique id to logs
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")

        self.run_id = date_time

        self.writer = SummaryWriter("logs/%s"%(self.run_id))
        self.scheduler = ReduceLROnPlateau(self.optimizer , 'min', verbose=True, patience=10)

    def calculate_iou(self, b1 : BoundingBox, b2 : BoundingBox) -> float:
        """Calculate intersection over union

        Args:
            b1 (BoundingBox): First bounding box
            b2 (BoundingBox): Second bounding box

        Returns:
            float: iou score between the boxes
        """
        x_a = max(b1.min_x, b2.min_x)
        y_a = max(b1.min_y, b2.min_y)

        x_b = min(b1.max_x, b2.max_x)
        y_b = min(b1.max_y, b2.max_y)

        intersection = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        area_b1 = (b1.max_x - b1.min_x + 1) * (b1.max_y - b1.min_y + 1)
        area_b2 = (b2.max_x - b2.min_x + 1) * (b2.max_y - b2.min_y + 1)

        iou = intersection / float(area_b1 + area_b2 - intersection)

        return iou

    # assume output is (batch_size*n_cells, cell_tensor_size)
    def parse_the_outputs(self, output : Type[torch.FloatTensor], boxes : List[List[BoundingBox]], classes : List[List[int]]):
        """Takes the output of the neural network and the target bounding boxes

        Args:
            output (torch.FloatTensor): The output from the neural network, reshaped as (batch_size*n_cells, cell_tensor_size)
            boxes (List[List[BoundingBox]]): List of list of target bounding boxes for each image
            clasess (List[int]): List of list of target classes for each image

        Returns:
            [type]: [description]
        """
        # Each cell has [ self.boxes_per_cell * 5 + n_classes ] entries, extract each relevant part
        bbox_slices = output[:, :self.boxes_per_cell * 5].reshape(-1, 5)
        class_slices = output[:, self.boxes_per_cell * 5: ]

        positive_bbox_outputs = []
        positive_bbox_targets = []
        negative_bbox_outputs = []

        positive_classprob_indices = []
        positive_classprob_target = []
        positive_classprob_output = []

        anchor_ids = []
        positive_bbox_indices_list = []

        # total number of cells in the batch output
        n_cells = output.shape[0]
        
        # gather positive cell indices and corresponding anchor boxes
        for image_id, bb_list in enumerate(boxes):
            # Process all bbs in the image
            for bb_id, bb in enumerate(bb_list):

                # Center in output grid coordinate
                center_x_grid  = self.output_grid_size * (bb.min_x + bb.max_x) / 2
                center_y_grid  = self.output_grid_size * (bb.min_y + bb.max_y) / 2

                # Get cell index in the batch
                cell_index = math.floor(center_y_grid) * self.output_grid_size + math.floor(center_x_grid)
                cell_index += image_id * self.model.n_cells

                # Calc bounding box offset within the grid
                offset_x = center_x_grid - math.floor(center_x_grid) 
                offset_y = center_y_grid - math.floor(center_y_grid)

                # Calc bb size
                width = (bb.max_x - bb.min_x) 
                height = (bb.max_y - bb.min_y)

                matching_anchors = []

                max_anchor_id = -1

                anchor_offset_x = (bb.min_x + bb.max_x) / 2
                anchor_offset_y = (bb.min_y + bb.max_y) / 2

                # Find matching matching boxes
                for anchor_id, anchor in enumerate(self.anchor_boxes):
                    adjusted_anchor = BoundingBox(min_x = anchor.min_x + anchor_offset_x, min_y = anchor.min_y + anchor_offset_y, max_x = anchor.max_x + anchor_offset_x, max_y= anchor.max_y + anchor_offset_y)
                    iou = self.calculate_iou(bb, adjusted_anchor)
                    if iou > 0.5:
                        matching_anchors.append((iou,anchor_id))

                # Find the maximum matching anchor
                if matching_anchors:
                    max_anchor_id = max(matching_anchors, key = lambda t: t[0])[1]

                    anchor = self.anchor_boxes[max_anchor_id]

                    anchor_w = anchor.max_x - anchor.min_x
                    anchor_h = anchor.max_y - anchor.min_y
                    
                    # calc bbox index
                    bbox_index = cell_index * self.model.boxes_per_cell + max_anchor_id

                    # Calc target tensor
                    target_tensor = torch.FloatTensor([ offset_x, offset_y, width, height, 1.0])
                    
                    positive_bbox_targets.append(target_tensor)

                    positive_bbox_indices_list.append(bbox_index)

                    # class prob target
                    #class_target = torch.zeros(self.n_classes)
                    #class_target[classes[image_id][bb_id]] = 1.0

                    positive_classprob_output.append(class_slices[cell_index])
                    positive_classprob_target.append(classes[image_id][bb_id])

                    anchor_ids.append(max_anchor_id)
        
        # Make a set for quick membership lookup
        positive_bbox_indices_set = set(positive_bbox_indices_list)
        # Other than positive indices, combine as negative indices
        negative_bbox_indices_list = torch.LongTensor([ s for s in range(bbox_slices.shape[0]) if s not in positive_bbox_indices_set ])

        # Get positive bbox and negative bbox
        positive_bbox_outputs = bbox_slices[positive_bbox_indices_list,:]
        negative_bbox_outputs = bbox_slices[negative_bbox_indices_list,:]

        positive_classprob_output = torch.stack(positive_classprob_output)
        positive_classprob_target = torch.LongTensor(positive_classprob_target).to(self.target_device)

        # Stack together bbox target tensors
        bbox_targets = torch.stack(positive_bbox_targets, 0).to(self.target_device)

        # Keep track of anchor sizes. This will be used as scaling factor for bbox size
        anchor_sizes = [ [self.anchor_boxes[s].max_x - self.anchor_boxes[s].min_x, self.anchor_boxes[s].max_y - self.anchor_boxes[s].min_y] for s in anchor_ids ]
        anchor_sizes = np.array(anchor_sizes)

        return positive_bbox_outputs, negative_bbox_outputs, bbox_targets, positive_classprob_output, positive_classprob_target, anchor_sizes

    def preview_generate(self, input_img : Type[torch.FloatTensor], output : Type[torch.FloatTensor], target_box : List[BoundingBox]):
        """Detection results preview generator

        Args:
            input_img (Type[torch.FloatTensor]): Input image (C,H,W)
            output (Type[torch.FloatTensor]) : Network output (-1,CELL_SIZE)
            target_box (List[BoundingBox]): List of the boxes in image[0]
        """
        np_image = np_image_from_tensor(input_img)

        img_height = np_image.shape[0]
        img_width = np_image.shape[1]
        

        # Reshape cells to 2D
        bb_outputs = output.view(self.model.output_grid_size, self.model.output_grid_size, -1)[:,:,:self.boxes_per_cell * 5]
        bb_outputs = bb_outputs.reshape(self.model.output_grid_size, self.model.output_grid_size, self.model.boxes_per_cell, -1)

        # Get objectness outputs
        objectness_output = torch.sigmoid(bb_outputs[:,:,:,-1])

        # Only select if objectness > 0.5, will return list of index tuples
        candidate_cell_index = torch.nonzero(objectness_output>0.1).detach().cpu().numpy().tolist()
        
        # Bounding box outputs of candidate boxes
        candidates_bb = []

        # Cell sizes in normalized coordinate
        cell_size_x = 1.0 / self.model.output_grid_size
        cell_size_y = 1.0 / self.model.output_grid_size
        
        # Bounding boxes showing candidate cells
        cell_bboxes = []

        for idx in candidate_cell_index:
            # Get output of current candidate cell
            bb = bb_outputs[idx[0], idx[1], idx[2]][:4]

            # idx[2] corresponds to the anchor idx
            anchor_w = self.anchor_boxes[idx[2]].max_x - self.anchor_boxes[idx[2]].min_x
            anchor_h = self.anchor_boxes[idx[2]].max_y - self.anchor_boxes[idx[2]].min_y
            
            # Compute offset and sizes
            bb[:2] = torch.sigmoid(bb[:2])
            bb[2:4] = torch.exp(bb[2:4])
            bb[2] *= anchor_w
            bb[3] *= anchor_h

            # Compute the origin of the cell coordinate in pixel coordinate
            base_x = cell_size_x * (idx[1] ) * img_width
            base_y = cell_size_y * (idx[0] ) * img_height

            preview_bb = bb.detach()

            preview_bb[0] = base_x + preview_bb[0] * cell_size_x * img_width  - preview_bb[2] * img_width / 2.0
            preview_bb[1] = base_y + preview_bb[1] * cell_size_y * img_height - preview_bb[3] * img_height / 2.0

            preview_bb[2] = preview_bb[0] + preview_bb[2] * img_width
            preview_bb[3] = preview_bb[1] + preview_bb[3] * img_height

            preview_bb[[0,2]] = torch.clamp_min(preview_bb[[0,2]], 0.0)
            preview_bb[[0,2]] = torch.clamp_max(preview_bb[[0,2]], img_width-1)
            preview_bb[[1,3]] = torch.clamp_min(preview_bb[[1,3]], 0.0)
            preview_bb[[1,3]] = torch.clamp_max(preview_bb[[1,3]], img_height-1)
            
            candidates_bb.append(preview_bb)

            cell_bbox = np.array([base_x, base_y, base_x + cell_size_x * img_width, base_y + cell_size_y*img_height])
            cell_bboxes.append(cell_bbox)


        target_boxes = []
        for target in target_box:
            bb_ref = list(target)
            bb_ref = np.array(bb_ref)

            # Convert to pixel coordinate
            bb_ref[[0,2]] *= img_width
            bb_ref[[1,3]] *= img_height
            
            target_boxes.append(bb_ref)

        if len(cell_bboxes) > 0:
            cell_bboxes = np.stack(cell_bboxes)
        else:
            cell_bboxes = torch.FloatTensor([])
            #self.writer.add_image_with_boxes("Preview/cell", np.array(image), cell_bboxes, step, dataformats='HWC')
        #else:
        #    self.writer.add_image("Preview/cell", np.array(image), step, dataformats='HWC')

        if len(target_boxes) > 0:
            target_boxes = np.stack(target_boxes)
        else:
            target_boxes = torch.FloatTensor([])
            #self.writer.add_image_with_boxes("Preview/target", np.array(image), target_boxes, step, dataformats='HWC')

        if len(candidates_bb) > 0:
            candidates_bb = torch.stack(candidates_bb, dim=0)
        else:
            candidates_bb = torch.FloatTensor([])
            #self.writer.add_image_with_boxes("Preview/detection", np.array(image), candidates_bb, step, dataformats='HWC')
        #else:
        #    self.writer.add_image("Preview/detection", np.array(image), step, dataformats='HWC')

        preview_info = {
            "image" : np_image,
            "target_bb" : target_boxes,
            "candidate_bb" : candidates_bb,
            "cell_bb" : cell_bboxes
        }

        return preview_info

    def run_batch(self, batch_data : Type[torch.Tensor], epoch : int):
        """Train batch

        Args:
            batch_data (Type[torch.Tensor]): Batch of data
            epoch (int): epoch number

        Returns:
            Dict: Dict of losses
        """
        images = batch_data["images"]
        boxes = batch_data["bboxes"]
        classes = batch_data["labels"]

        images = images.float().to(self.target_device)

        self.optimizer.zero_grad()

        # run network
        output = self.model(images)

        # Reshape to cell outputs
        cell_outputs = output.view(-1, self.model.cell_tensor_size)
        
        # Parse the cell outputs
        positive_bbox, negative_bbox, positive_bbox_targets, positive_classprob_output, positive_classprob_target, anchor_sizes = self.parse_the_outputs(cell_outputs, boxes, classes)

        # Cell structure, ( box1, box2, box3, ..., objectness, class_1, class_2, ... )
        positive_objectness_output = positive_bbox[:,-1].flatten()
        negative_objectness_output = negative_bbox[:,-1].flatten()
        objectness_output = torch.cat([positive_objectness_output, negative_objectness_output], dim=0)

        # generate target objectness tensor
        objectness_target_positive = torch.ones(positive_objectness_output.shape[0]).to(self.target_device)
        objectness_target_negative = torch.zeros(negative_objectness_output.shape[0]).to(self.target_device)

        objectness_positive_loss = nn.functional.mse_loss(input=torch.sigmoid(positive_objectness_output), target = objectness_target_positive, reduction="sum")
        objectness_negative_loss = nn.functional.mse_loss(input=torch.sigmoid(negative_objectness_output), target = objectness_target_negative, reduction="sum")
        
        # Apply weight to the losses to handle class imbalance
        objectness_loss = objectness_positive_loss + 0.5 * objectness_negative_loss
        
        # Update offset to sigmoid
        positive_bbox[:,:2] = torch.sigmoid(positive_bbox[:,:2])
        # width = exp(w) * anchor_w
        # height = exp(h) * anchor_h
        positive_bbox[:,2:4] = torch.exp(positive_bbox[:,2:4])
        positive_bbox[:,2:4] *= torch.from_numpy(anchor_sizes).to(self.target_device)
        
        # Calc bb loss
        bb_loss_offset = nn.functional.mse_loss(positive_bbox[:,:2], positive_bbox_targets[:,:2], reduction='sum')
        bb_loss_dims = nn.functional.mse_loss(torch.sqrt(positive_bbox[:,2:4]) , torch.sqrt(positive_bbox_targets[:,2:4]), reduction='sum')

        bb_loss = bb_loss_offset + bb_loss_dims
        
        # Calc classes prob loss
        classes_loss = nn.functional.nll_loss(input=nn.functional.log_softmax(positive_classprob_output), target=positive_classprob_target, reduction = 'sum')

        # Make bb_loss stronger to counter gradient from negative objectness
        total_loss = 1 * objectness_loss + 5 * bb_loss + classes_loss
        
        total_loss.backward()

        self.optimizer.step()

        preview_info = self.preview_generate(images[0], output[0], boxes[0])

        loss_info = {
            "objectness" : objectness_loss.detach().item(),
            "bb" : bb_loss.detach().item(),
            "class" : classes_loss.detach().item(),
            "total" : total_loss.detach().item()
        }

        return loss_info, preview_info
        
        
    def start_training(self):
        # Create dataloader to batch the dataset
        dataloader_train = DataLoader(self.dataset_train, 32, True, collate_fn = object_dataset_collate_fn, num_workers=6, worker_init_fn=worker_init_fn)
        sampler =  torch.utils.data.RandomSampler(self.dataset_validation, replacement= True, num_samples = 32)
        dataloader_validation = DataLoader(self.dataset_validation, 1, False, collate_fn = object_dataset_collate_fn, num_workers=1, worker_init_fn=worker_init_fn, sampler=sampler)

        for ep in range(self.max_epoch):
            print("epoch : {}".format(ep))
            objectness_loss = []
            bb_loss = []
            total_loss = []
            classes_loss = []

            # Train batches
            for idx, data in enumerate(dataloader_train):
                losses, preview_images = self.run_batch(data, idx)
                objectness_loss.append(losses["objectness"])
                bb_loss.append(losses["bb"])
                total_loss.append(losses["total"])
                classes_loss.append(losses["class"])
            
            mean_objectness_loss = np.mean(objectness_loss)
            mean_bb_loss = np.mean(bb_loss)
            mean_total_loss = np.mean(total_loss)
            mean_classes_loss = np.mean(classes_loss)
            
            self.writer.add_scalar("training/mean_objectness_loss", mean_objectness_loss, ep)
            self.writer.add_scalar("training/mean_bb_loss", mean_bb_loss, ep)
            self.writer.add_scalar("training/mean_classes_loss", mean_classes_loss, ep)
            self.writer.add_scalar("training/mean_total_loss", mean_total_loss, ep)

            self.writer.add_image_with_boxes("training/target", preview_images["image"], preview_images["target_bb"], ep, dataformats='HWC')
            self.writer.add_image_with_boxes("training/cell", preview_images["image"], preview_images["cell_bb"], ep, dataformats='HWC')
            self.writer.add_image_with_boxes("training/detection", preview_images["image"], preview_images["candidate_bb"], ep, dataformats='HWC')
            
            if ep % 20 == 0:
                val_objectness_losses = []
                val_bb_losses = []
                val_class_losses = []
                val_total_losses = []

                # Eval mode for validation
                self.model.eval()

                # Validation
                for idx, data in enumerate(dataloader_validation):
                    losses, preview_images = self.run_batch(data, idx)
                    val_objectness_losses.append(losses["objectness"])
                    val_bb_losses.append(losses["bb"])
                    val_total_losses.append(losses["total"])
                    val_class_losses.append(losses["class"])

                # Return to train mode
                self.model.train()

                # Update learning rate scheduler step
                self.scheduler.step(mean_total_loss)
                
                val_mean_objectness_loss = np.mean(val_objectness_losses)
                val_mean_bb_loss = np.mean(val_objectness_losses)
                val_mean_total_loss = np.mean(val_total_losses)
                val_mean_classes_loss = np.mean(val_class_losses)

                self.writer.add_scalar("validation/mean_objectness_loss", val_mean_objectness_loss, ep)
                self.writer.add_scalar("validation/mean_bb_loss", val_mean_bb_loss, ep)
                self.writer.add_scalar("validation/mean_classes_loss", val_mean_classes_loss, ep)
                self.writer.add_scalar("validation/mean_total_loss", val_mean_total_loss, ep)

                self.writer.add_image_with_boxes("validation/target", preview_images["image"], preview_images["target_bb"], ep, dataformats='HWC')
                self.writer.add_image_with_boxes("validation/cell", preview_images["image"], preview_images["cell_bb"], ep, dataformats='HWC')
                self.writer.add_image_with_boxes("validation/detection", preview_images["image"], preview_images["candidate_bb"], ep, dataformats='HWC')
                
                checkpoint_dir = "checkpoints/{}".format(self.run_id)
                if (not os.path.exists(checkpoint_dir)):
                    os.makedirs(checkpoint_dir)

                torch.save(self.model.state_dict(), "{}/{}_{}.pth".format(checkpoint_dir, ep,  mean_total_loss))
            
if __name__ == "__main__":
    anchor_boxes = [    BoundingBox(min_x = -0.15, min_y = -0.15, max_x = 0.15, max_y = 0.15),
                        BoundingBox(min_x = -0.25, min_y = -0.25, max_x = 0.25, max_y = 0.25),
                        BoundingBox(min_x = -0.35, min_y = -0.35, max_x = 0.35, max_y = 0.35),
                        BoundingBox(min_x = -0.45, min_y = -0.45, max_x = 0.45, max_y = 0.45)
                     ]

    label_map = {
        "head" : 0
    }

    trainer = ObjectDetectionTrainer(10000000, "/workspace/face_detector/dataset/train", "/workspace/face_detector/dataset/val", 7, anchor_boxes=anchor_boxes, target_device="cuda:0", label_map=label_map)
    trainer.start_training()