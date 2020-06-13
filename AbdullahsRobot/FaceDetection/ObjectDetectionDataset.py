"""Object Detection Dataset

This is a module for managing object detection dataset.

"""
import os
from io import StringIO, BytesIO
import base64
import json
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional, NamedTuple, Type

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
from torch.utils.data import Dataset

class BoundingBox(NamedTuple):
    """A named tuple class to represent bounding box.

    Attributes:
        min_x (float) : minimum x coordinate
        min_y (float) : minimum y coordinate
        max_x (float) : maximum x coordinate
        max_y (float) : maximum y coordinate
    """
    min_x : float
    min_y : float
    max_x : float
    max_y : float

class ObjectDetectionDataset(Dataset):
    """Pytorch dataset for loading LabelMe json annotation files."""

    def __init__(self, dataset_root_path : str, target_image_size : Tuple[int,int], transform = None, **kwargs):
        """Construct a new dataset

        Args:
            dataset_root_path (str): The path to folder which contains LabelMe .json files
            target_image_size (Tuple[int,int]): A tuple of integer (size_x, size_y) to resize the image to
            transform (optional): A transform function. Defaults to None.
        """
        super(ObjectDetectionDataset, self).__init__(**kwargs)
        self.target_image_size = target_image_size
        
        self.transform = transform
        # Find all json files
        discovered_json = Path(dataset_root_path).rglob('*.json')
        
        self.json_list = list(discovered_json)
        
    def __getitem__(self, idx : int) -> Dict:
        """Get data at certain index

        Args:
            idx (int): the index

        Returns:
            Dict: A dictionary of the data
        """

        data_path = self.json_list[idx]

        try:
            with open(data_path, "r") as f:
                json_data = json.load(f)
        except:
            print("Failed to open json file %s!" %(data_path))
            raise
        
        image_str = json_data["imageData"] # Get base64 string representation of the image data

        # Instead of reading the image from a file, we will create a BytesIO from the image data
        tempBuff = BytesIO() 
        # Decode the base64 string and write to buffer
        tempBuff.write(base64.b64decode(image_str))
        # After writing the buffer will be in the end position, reset to starting position for reading
        tempBuff.seek(0) 
        
        # Open the image as Pillow Image from the buffer
        image = Image.open(tempBuff)
        # Resize to target
        # TODO : mantain aspect ratio
        resized_img = image.resize((self.target_image_size))
        
        # Computing the x and y scale so we can transform the bounding boxes with the same scale
        x_scale = resized_img.size[0] / image.size[0]
        y_scale = resized_img.size[1] / image.size[1]

        # Get list of shapes
        shapes = json_data["shapes"]

        bboxes = []
        labels = []

        for s in shapes:
            # Only process rectangle annotation
            if s["shape_type"] == "rectangle":
                # The rectangle is described by 2 points, convert to np array and flatten it
                bbox = np.array(s["points"]).reshape(-1)

                # Scale the x coordinates
                bbox[[0,2]] = bbox[[0,2]] * x_scale / self.target_image_size[0]
                # Scale the y coordinates
                bbox[[1,3]] = bbox[[1,3]] * y_scale / self.target_image_size[1]
                
                # Create new bounding box
                bbox = BoundingBox(min_x=bbox[0],min_y=bbox[1],max_x=bbox[2], max_y=bbox[3])
                # Add bounding box to list
                bboxes.append(bbox)

                # Add label to list of labels
                labels.append(s["label"])

        # If no transform is given output the same        
        resized_bb = bboxes

        # Apply transform if given
        if self.transform:
            resized_img, resized_bb = self.transform(resized_img, bboxes)

        # Construct the data dict
        data_item = {
            "bboxes" : resized_bb,
            "image" : resized_img,
            "labels" : labels
        }

        return data_item


    def __len__(self):
        return len(self.json_list)


def object_dataset_collate_fn(batch : List[Dict]) -> Dict:
    """Collate function to be passed to dataloader, to organize list of data into a batch of data.

    Args:
        batch (List[Dict]): List of data

    Returns:
        Dict: Batch of data
    """

    if isinstance(batch["image"], np.ndarray):
        colorImages = np.stack(batch["image"])
    
    if isinstance(batch["image"], torch.FloatTensor):
        colorImages = torch.stack(batch["image"])

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
