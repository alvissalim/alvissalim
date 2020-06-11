import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np

from PIL import Image, ImageDraw
from io import StringIO, BytesIO

import base64

import json

from typing import List, Set, Dict, Tuple, Optional, NamedTuple

class BoundingBox(NamedTuple):
    min_x : float
    min_y : float
    max_x : float
    max_y : float

class ObjectDetectionDataset(Dataset):
    def __init__(self, dataset_root_path : str, target_image_size : Tuple[int,int], transform=None, **kwargs):
        super(ObjectDetectionDataset, self).__init__(**kwargs)
        self.target_image_size = target_image_size
        self.transform = transform
        # Find all npz files
        g = Path(dataset_root_path).rglob('*.json')
        
        self.json_list = list(g)
        
    def __getitem__(self, idx : int):
        data_path = self.json_list[idx]

        json_data = None

        with open(data_path, "r") as f:
            json_data = json.load(f)

        image_str = json_data["imageData"]

        tempBuff = BytesIO()
        tempBuff.write(base64.b64decode(image_str))
        tempBuff.seek(0)

        image = Image.open(tempBuff)
        resized_img = image.resize((self.target_image_size))

        x_scale = resized_img.size[0] / image.size[0]
        y_scale = resized_img.size[1] / image.size[1]

        shapes = json_data["shapes"]

        bboxes = []
        labels = []

        for s in shapes:
            if s["shape_type"] == "rectangle":
                bbox = np.array(s["points"]).reshape(-1)
                bbox[[0,2]] = bbox[[0,2]] * x_scale / self.target_image_size[0]
                bbox[[1,3]] = bbox[[1,3]] * y_scale / self.target_image_size[1]
                
                bbox = BoundingBox(min_x=bbox[0],min_y=bbox[1],max_x=bbox[2], max_y=bbox[3])
                bboxes.append(bbox)

                labels.append(s["label"])

        resized_bb = bboxes

        if self.transform:
            resized_img, resized_bb = self.transform(resized_img, bboxes)

        

        #print(resized_bb)
        resized_img = np.array(resized_img)
        data_item = {
            "bboxes" : resized_bb,
            "image" : resized_img,
            "labels" : labels
        }
        return data_item


    def __len__(self):
        return len(self.json_list)


def face_dataset_collate_fn(batch):
    colorImages = [ torch.from_numpy(s["image"]) / 255.0 - 0.5 for s in batch ]
    colorImages = torch.stack(colorImages)

    boundingBoxes = [ s["bboxes"] for s in batch ]

    labels = [ s["labels"] for s in batch ]
    
    res = {
        "images" : colorImages,
        "bboxes" : boundingBoxes,
        "labels" : labels
    }
    return res

if  __name__=="__main__":
    dataset = ObjectDetectionDataset("/workspace/face_detector/dataset", (320,320))
    
    for d in dataset:
        im = Image.fromarray(d["image"])

        bb = d["bboxes"][0]

        rect = np.array(list(bb))

        rect[[0,2]] = rect[[0,2]] * im.width
        rect[[1,3]] = rect[[1,3]] * im.height

        rect = rect.astype(np.uint32)
        #print(im.size)
        #print(rect)
        
        draw = ImageDraw.Draw(im)
        draw.rectangle(rect.tolist())
        im.save("temp.png")
