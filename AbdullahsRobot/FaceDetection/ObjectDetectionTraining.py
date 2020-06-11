from ObjectDetectionModel import ObjectDetectionModel
from ObjectDetectionDataset import ObjectDetectionDataset, face_dataset_collate_fn, BoundingBox
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from typing import Type, List, NamedTuple
import math
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

import torchvision.transforms.functional as TF
from torchvision.transforms import RandomApply
import random

from  torchvision.transforms import ColorJitter

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

preview_counter = 0

def worker_init_fn(worker_id):                                                          
    np.random.seed()

def image_bb_transforms(image, bboxes : List[BoundingBox]):

    transformed_bboxes = bboxes

    if random.random() > 0.5:
        transformed_bboxes = bboxes
        
        translate = np.random.uniform(low=-0.4, high=0.4, size = 2)
        scale = np.random.uniform(low=0.25, high=1.0, size=1)
        
        transformed_bboxes = []

        for bb in bboxes:
            scaled_bb = np.array(list(bb))
            center_offset = ( scaled_bb - 0.5 )
            scaled_offset = center_offset * scale
            scaled_bb = 0.5 + scaled_offset

            scaled_bb[[0,2]] = scaled_bb[[0,2]] + translate[0]
            scaled_bb[[1,3]] = scaled_bb[[1,3]] + translate[1]

            scaled_bb = np.clip(scaled_bb, 0.0, 1.0).reshape(-1)

            transformed_bboxes.append(BoundingBox(*scaled_bb.tolist()))

        translate[0] *= image.width
        translate[1] *= image.height
        random_color = np.random.randint(0, 255, size=3).tolist()

        image = TF.affine(image, 0, translate.tolist(), scale, 0, resample=Image.BILINEAR, fillcolor=tuple(random_color))

    jitter = ColorJitter(0.25, 0.25, 0.25, 0.25)

    image = RandomApply([jitter], 0.5)(image)

        
    # more transforms ...
    return image, transformed_bboxes

class FaceDetectionTrainer:
    def __init__(self, max_epoch : int, dataset_path : str, output_grid_size : int, anchor_boxes : List[BoundingBox], target_device : str):
        self.max_epoch = max_epoch
        self.dataset = ObjectDetectionDataset(dataset_path, (512,512), transform=image_bb_transforms)
        self.boxes_per_cell = len(anchor_boxes)
        self.anchor_boxes = anchor_boxes
        self.output_grid_size = output_grid_size
        self.model = ObjectDetectionModel(1, self.output_grid_size, self.boxes_per_cell)
        self.target_device = target_device
        self.model.to(self.target_device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.writer = SummaryWriter("logs")
        self.scheduler = ReduceLROnPlateau(self.optimizer , 'min', verbose=True, patience=50)

    def calculate_iou(self, b1 : BoundingBox, b2 : BoundingBox):
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
    def split_positive_negative_cells(self, output, boxes):
        bbox_slices = output[:, :self.boxes_per_cell * 5].reshape(-1, 5)
        class_slices = output[:, self.boxes_per_cell * 5 : ]

        positive_bbox_outputs = []
        positive_bbox_targets = []
        negative_bbox_outputs = []
        anchor_ids = []

        positive_bbox_indices_list = []

        n_cells = output.shape[0]
        
        # gather positive cell indices
        for image_id, bb_list in enumerate(boxes):
            for bb in bb_list:
                center_x_grid = self.output_grid_size * (bb.min_x + bb.max_x) / 2
                center_y_grid  = self.output_grid_size * (bb.min_y + bb.max_y) / 2

                cell_index = math.floor(center_y_grid) * self.output_grid_size + math.floor(center_x_grid)
                cell_index += image_id * self.model.n_cells

                offset_x = center_x_grid - math.floor(center_x_grid) 
                offset_y = center_y_grid - math.floor(center_y_grid)

                # cell size
                cell_size_grid_x = 1.0
                cell_size_grid_y = 1.0
                

                # Bound to 0~1
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

                if matching_anchors:
                    max_anchor_id = max(matching_anchors, key = lambda t: t[0])[1]

                    anchor = self.anchor_boxes[max_anchor_id]

                    #print(anchor)

                    anchor_w = anchor.max_x - anchor.min_x
                    anchor_h = anchor.max_y - anchor.min_y

                    #print("{}, {}".format(anchor_w, anchor_h))

                    
                    # calc bbox index
                    bbox_index = cell_index * self.model.boxes_per_cell + max_anchor_id

                    target_tensor = torch.FloatTensor([ offset_x / cell_size_grid_x, offset_y / cell_size_grid_y, width, height, 1.0])
                    
                    positive_bbox_targets.append(target_tensor)

                    positive_bbox_indices_list.append(bbox_index)

                    anchor_ids.append(max_anchor_id)
                
        positive_bbox_indices_set = set(positive_bbox_indices_list)
        negative_bbox_indices_list = torch.LongTensor([ s for s in range(bbox_slices.shape[0]) if s not in positive_bbox_indices_set ])

        positive_bbox_outputs = bbox_slices[positive_bbox_indices_list,:]
        negative_bbox_outputs = bbox_slices[negative_bbox_indices_list,:]
        bbox_targets = torch.stack(positive_bbox_targets, 0).to(self.target_device)

        anchor_sizes = [ [self.anchor_boxes[s].max_x - self.anchor_boxes[s].min_x, self.anchor_boxes[s].max_y - self.anchor_boxes[s].min_y] for s in anchor_ids ]
        anchor_sizes = np.array(anchor_sizes)

        return positive_bbox_outputs, negative_bbox_outputs, bbox_targets, anchor_sizes

    def preview_res(self, sample_data, step, target_box):
        image = (sample_data[0] + 0.5) * 255

        image = image.to(torch.uint8).detach().cpu().numpy()
        #print(image)
        #print(image.shape)
        image = Image.fromarray(image)
        #print(image.size)
        sample_data = sample_data.float().to(self.target_device)

        sample_data = sample_data.permute(0, 3, 1, 2)

        output = self.model(sample_data)
        bb_outputs = output.view(self.model.output_grid_size, self.model.output_grid_size, -1)[:,:,:self.boxes_per_cell * 5]
        bb_outputs = bb_outputs.reshape(self.model.output_grid_size, self.model.output_grid_size, self.model.boxes_per_cell, -1)


        objectness_output = torch.sigmoid(bb_outputs[:,:,:,-1])

        candidate_cell_index = torch.nonzero(objectness_output>0.5).detach().cpu().numpy().tolist()

        #draw = 
        candidates_bb = []

        cell_size_x = 1.0 / self.model.output_grid_size
        cell_size_y = 1.0 / self.model.output_grid_size

        used_anchor_boxes = []

        cell_bboxes = []

        anchor_w = 1
        anchor_h = 1
        for idx in candidate_cell_index:
            bbs = bb_outputs[idx[0], idx[1], idx[2]]

            anchor_w = self.anchor_boxes[idx[2]].max_x - self.anchor_boxes[idx[2]].min_x
            anchor_h = self.anchor_boxes[idx[2]].max_y - self.anchor_boxes[idx[2]].min_y

            

            #bbox_slices = cell_output

            #bbs = bbox_slices[:, : 4]

            bb=bbs

            bb[:2] = torch.sigmoid(bbs[:2])
            bb[2:4] = torch.exp(bbs[2:4])
            

            bb[2] *= anchor_w
            bb[3] *= anchor_h
             
            #width = image.shape[2]
            #height = image.shape[1]

            base_x = cell_size_x * (idx[1] ) * image.width
            base_y = cell_size_y * (idx[0] ) * image.height

            #print("base x {}, base y {}".format(base_x, base_y))
            
            #print("image.shape {}".format(image.shape))

            #for bb in bbs:

            bb[0] = base_x + bb[0] * cell_size_x * image.width  - bb[2] * image.width / 2.0
            bb[1] = base_y + bb[1] * cell_size_y * image.height - bb[3] * image.height / 2.0

            bb[2] = bb[0] + bb[2] * image.width
            bb[3] = bb[1] + bb[3] * image.height

            bb[[0,2]] = torch.clamp_min(bb[[0,2]], 0.0)
            bb[[0,2]] = torch.clamp_max(bb[[0,2]], image.width-1)
            bb[[1,3]] = torch.clamp_min(bb[[1,3]], 0.0)
            bb[[1,3]] = torch.clamp_max(bb[[1,3]], image.height-1)
            

            candidates_bb.append(bb)

            anchor_box = np.array(list(self.anchor_boxes[idx[2]]))

            cell_bbox = np.array([base_x, base_y, base_x + cell_size_x * image.width, base_y + cell_size_y*image.height])
            cell_bboxes.append(cell_bbox)

            used_anchor_boxes.append(anchor_box)

        bb_ref = list(target_box[0])
        
        #print(bb_ref)
        bb_ref = np.array(bb_ref)

        bb_ref[[0,2]] *= image.width
        bb_ref[[1,3]] *= image.height
        
        bb_ref= bb_ref.reshape(-1,4)

        bb_ref_midx = np.mean(bb_ref[:,[0,2]], axis=1) 
        bb_ref_midy = np.mean(bb_ref[:,[1,3]], axis=1) 

        #used_anchor_boxes = np.stack(used_anchor_boxes)

        #used_anchor_boxes[[0,2]] *= image.width
        #used_anchor_boxes[[1,3]] *= image.height

        #used_anchor_boxes[[0,2]] += bb_ref_midx
        #used_anchor_boxes[[1,3]] += bb_ref_midy
        if cell_bboxes:
            cell_bboxes = np.stack(cell_bboxes)
            
            self.writer.add_image_with_boxes("Preview/cell", np.array(image), cell_bboxes, step, dataformats='HWC')

        else:
            self.writer.add_image("Preview/cell", np.array(image), step, dataformats='HWC')


        #print(used_anchor_boxes)

        #print(bb_ref)
        self.writer.add_image_with_boxes("Preview/target", np.array(image), bb_ref, step, dataformats='HWC')

        
        #print(candidates_bb)
        if len(candidates_bb) > 0:
            candidates_bb = torch.stack(candidates_bb, dim=0)
            self.writer.add_image_with_boxes("Preview/detection", np.array(image), candidates_bb, step, dataformats='HWC')
        else:
            self.writer.add_image("Preview/detection", np.array(image), step, dataformats='HWC')

        
    def train_batch(self, batch_data : Type[torch.Tensor], epoch):
        images = batch_data["images"]
        boxes = batch_data["bboxes"]

        images = images.float().to(self.target_device)

        images = images.permute(0, 3, 1, 2)

        #print(images.shape)

        self.optimizer.zero_grad()

        # run network
        output = self.model(images)

        output = output.view(-1, self.model.cell_tensor_size)
        
        positive_bbox, negative_bbox, positive_bbox_targets, anchor_sizes = self.split_positive_negative_cells(output, boxes)

        #print(positive_bbox_targets)
        # Compute objectness loss
        
        # Cell structure, ( objectness, box1, box2, box3, ... )
        positive_objectness_output = positive_bbox[:,-1].flatten()
        negative_objectness_output = negative_bbox[:,-1].flatten()
        objectness_output = torch.cat([positive_objectness_output, negative_objectness_output], dim=0)

        # generate target objectness tensor
        objectness_target_positive = torch.ones(positive_objectness_output.shape[0]).to(self.target_device)
        objectness_target_negative = torch.zeros(negative_objectness_output.shape[0]).to(self.target_device)

        weight = torch.cat( [ 1 * torch.ones(positive_objectness_output.shape[0]), 0.5 * torch.ones(negative_objectness_output.shape[0]) ])

        weight = weight.to(self.target_device)

        objectness_target = torch.cat([objectness_target_positive, objectness_target_negative], dim=0)
        
        objectness_loss = nn.functional.mse_loss(input=torch.sigmoid(objectness_output), target = objectness_target, reduction="none") * weight
        objectness_loss = torch.sum(objectness_loss)
        
        positive_bbox_output = positive_bbox

        positive_bbox_output[:,:2] = torch.sigmoid(positive_bbox_output[:,:2])
        positive_bbox_output[:,2:4] = torch.exp(positive_bbox_output[:,2:4])
        positive_bbox_output[:,2:4] *= torch.from_numpy(anchor_sizes).to(self.target_device)
        
        bb_loss_center = nn.functional.mse_loss(positive_bbox_output[:,:2], positive_bbox_targets[:,:2], reduction='sum')
        bb_loss_dims = nn.functional.mse_loss(torch.sqrt(positive_bbox_output[:,2:4]) , torch.sqrt(positive_bbox_targets[:,2:4]), reduction='sum')

        bb_loss = bb_loss_center + bb_loss_dims
        #classes_targets = torch.cat([torch.ones(classes_positives.shape[0]), torch.ones(classes_negatives.shape[0])], dim=0).to(self.target_device)
        #classes_outputs = torch.cat([classes_positives, classes_negatives], dim=0)
        #classes_loss = nn.functional.mse_loss(torch.sigmoid(classes_outputs), classes_targets)

        #print(bbox_targets)

        #if (epoch > 100):
        total_loss = 1 * objectness_loss + 5 * bb_loss #+ classes_loss
            
        #else:
        #    total_loss = objectness_loss
        
        total_loss.backward()

        self.optimizer.step()
        
        # bounding boxes losses

        #print(objectness_loss.detach())

        return objectness_loss.detach().item(), bb_loss.detach().item(), total_loss.detach().item()#, classes_loss.detach().item()
        
        
    def start_training(self):
        dataloader = DataLoader(self.dataset, 16, True, collate_fn = face_dataset_collate_fn, num_workers=6, worker_init_fn=worker_init_fn)
        for ep in range(self.max_epoch):
            print("epoch : {}".format(ep))
            objectness_loss = []
            bb_loss = []
            total_loss = []
            classes_loss = []

            for idx, data in enumerate(dataloader):
                losses = self.train_batch(data, idx)
                objectness_loss.append(losses[0])
                bb_loss.append(losses[1])
                total_loss.append(losses[2])
                #classes_loss.append(losses[3])
                
            images = data["images"]
            boxes = data["bboxes"]
            

            #print(objectness_loss)
            #objectness_loss = torch.cat(objectness_loss)
            #bb_loss = torch.cat(bb_loss)
            #total_loss = torch.cat(total_loss)

            
            mean_objectness_loss = np.mean(objectness_loss)
            mean_bb_loss = np.mean(bb_loss)
            mean_total_loss = np.mean(total_loss)
            #mean_classes_loss = np.mean(classes_loss)

            self.scheduler.step(mean_total_loss)
            self.model.eval()
            self.preview_res(images[[0]], ep, boxes[0])
            self.model.train()
            self.writer.add_scalar("training/objectness_loss", mean_objectness_loss, ep)
            self.writer.add_scalar("training/mean_bb_loss", mean_bb_loss, ep)
            self.writer.add_scalar("training/total_loss", mean_total_loss, ep)
            #self.writer.add_scalar("training/classes_loss", mean_classes_loss, ep)
            if ep % 300 == 0:
                torch.save(self.model.state_dict(), "checkpoints/{}_{}.pth".format(ep, mean_total_loss))
            
if __name__ == "__main__":
    anchor_boxes = [    BoundingBox(min_x = -0.15, min_y = -0.15, max_x = 0.15, max_y = 0.15),
                        BoundingBox(min_x = -0.25, min_y = -0.25, max_x = 0.25, max_y = 0.25),
                        BoundingBox(min_x = -0.35, min_y = -0.35, max_x = 0.35, max_y = 0.35),
                        BoundingBox(min_x = -0.45, min_y = -0.45, max_x = 0.45, max_y = 0.45)
                     ]
    trainer = FaceDetectionTrainer(10000000, "/workspace/face_detector/dataset", 7, anchor_boxes, "cuda:0")
    trainer.start_training()