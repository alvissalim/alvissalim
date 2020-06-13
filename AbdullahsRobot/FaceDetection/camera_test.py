"""Object Detection Tester Program

    This is a module for testing object detection model.

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

import cv2
from ObjectDetectionModel import ObjectDetectionModel
from ObjectDetectionDataset import BoundingBox
import torch
from torchvision.ops import nms
from torchvision.transforms.functional import normalize, to_tensor
from PIL import Image, ImageDraw
import numpy as np

CHECKPOINT_PATH = "checkpoints/06_13_2020_08_48_39/200_8.117394844690958.pth"
INPUT_VIDEO_PATH = "face.mp4"
OUTPUT_VIDEO_PATH = "output.avi"

anchor_boxes = [    BoundingBox(min_x = -0.15, min_y = -0.15, max_x = 0.15, max_y = 0.15),
                        BoundingBox(min_x = -0.25, min_y = -0.25, max_x = 0.25, max_y = 0.25),
                        BoundingBox(min_x = -0.35, min_y = -0.35, max_x = 0.35, max_y = 0.35),
                        BoundingBox(min_x = -0.45, min_y = -0.45, max_x = 0.45, max_y = 0.45)
                     ]

def load_model(check_point_path):
    model = ObjectDetectionModel(n_classes=1, output_grid_size=7, boxes_per_cell=4)
    model.load_state_dict(torch.load(check_point_path))
    return model

def detect_face_in_frame(model, image):
    
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    image_tensor = to_tensor(image)
    image_tensor = normalize(image_tensor, mean=image_mean, std=image_std)
    

    #image_tensor = torch.from_numpy(image).float() / 255.0 - 0.5
    #image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)

    image = Image.fromarray(image)

    image_tensor = image_tensor.to("cpu:0")

    output = model(image_tensor)

    bb_outputs = output.view(model.output_grid_size, model.output_grid_size, -1)[:,:,:model.boxes_per_cell * 5]
    bb_outputs = bb_outputs.reshape(model.output_grid_size, model.output_grid_size, model.boxes_per_cell, -1)


    objectness_output = torch.sigmoid(bb_outputs[:,:,:,-1])

    candidate_cell_index = torch.nonzero(objectness_output > 0.25).detach().cpu().numpy().tolist()

    bb_scores = []
    candidates_bb = []

    cell_size_x = image.width / model.output_grid_size
    cell_size_y = image.height / model.output_grid_size
    
    for idx in candidate_cell_index:
        bb = bb_outputs[idx[0], idx[1], idx[2]]

        anchor_w = anchor_boxes[idx[2]].max_x - anchor_boxes[idx[2]].min_x
        anchor_h = anchor_boxes[idx[2]].max_y - anchor_boxes[idx[2]].min_y

        bb[:2] = torch.sigmoid(bb[:2])
        bb[2:4] = torch.exp(bb[2:4])
        
        bb[2] *= anchor_w
        bb[3] *= anchor_h

        base_x = cell_size_x * idx[1]
        base_y = cell_size_y * idx[0]

        #for bb in bbs:
        bb[0] = base_x + bb[0] * cell_size_x - bb[2] * image.width / 2.0
        bb[1] = base_y + bb[1] * cell_size_y - bb[3] * image.height / 2.0

        bb[2] = bb[0] + bb[2] * image.width
        bb[3] = bb[1] + bb[3] * image.height

        candidates_bb.append(bb[:4])
        bb_scores.append(bb[4])

    
    if candidates_bb:
        candidates_bb = torch.stack(candidates_bb, dim=0).view(-1,4)
        #print(candidates_bb)
        bb_scores = torch.stack(bb_scores, dim=0).view(-1)
        #print(bb_scores)
        filtered_bb = nms(candidates_bb, bb_scores, 0.5)
        #print(filtered_bb)
        for bb in candidates_bb[filtered_bb]:
            #print(bb)
            draw = ImageDraw.Draw(image)
            draw.rectangle(bb.detach().cpu().numpy(), outline=(255,0,0))

    return np.array(image)



if __name__ == "__main__":
    model = load_model(CHECKPOINT_PATH).to("cpu:0").eval()

    # load video file
    vid = cv2.VideoCapture(INPUT_VIDEO_PATH)

    out_vid = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (512,300))

    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()

        if not ret:
            break

        # get image height, width
        (h, w) = frame.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2) 

        M = cv2.getRotationMatrix2D(center, 90, 1)
        frame = cv2.warpAffine(frame, M, (w, h)) 

        

        frame = cv2.resize(frame, (512,300))

        frame = detect_face_in_frame(model, frame)

        out_vid.write(frame)

    vid.release()
    out_vid.release()