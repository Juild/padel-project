from typing import Dict, List, Tuple, Union
from torch import Tensor
from torchvision import transforms
import torch
import cv2
import os
import json
import numpy as np

def import_data(annotations_path: str, images_path: str) -> Union[Tensor, Tensor, Tuple, Tuple]:
    # List containing each image in a Tensor form (W, H, R, G, B)
    images: List[Tensor] = []
    # List containing tuples where each tuple represents a bbox for 1 image (x0, y0, x1, y0)
    bboxes: List[Tuple] = []
    
    annotations_files: List[str] = os.listdir(annotations_path)
    s = 0
    for json_file_name in annotations_files:
        file_path = annotations_path + json_file_name
        with open(file_path, 'r') as f:
            if s == 60:
                break
            s +=1
            
            annotation: Dict = json.load(f)[0] # as it is a list with one element the dict
            #Images
            print(images_path + annotation['image'])
            image = cv2.imread(images_path + annotation["image"])
            # Convert the image to HSV format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #  (Y: 1080, X: 1920, RGB) (W,H,HSV) 
            image = torch.tensor(image, dtype=float)
            (h, w) = image.shape[:2]
            image = (image * 2)/255. - 1 # normalization [-1, 1]
            images.append(image)
            
            # Bounding boxes
            # [0] becasue "annoations" object is a list and we there's only one bounding box
            bbox = annotation['annotations'][0]["coordinates"] # {x, y, width, height}

            # Normalize bounding boxes too
            x0 = ( bbox['x'] - bbox['width']/2. )/ w
            y0 = ( bbox['y'] - bbox['height']/2. ) / h 
            x1 = ( bbox['x'] + bbox['width']/2. ) / w
            y1 = ( bbox['y'] + bbox['height']/2. ) / h
            bboxes.append((x0, y0, x1, y1))

    images = torch.stack(images)
    means = []
    stds = []
    # three channels rgb
    for channel in range(3):
        channel_stack: torch.Tensor = images[:, :, :, channel].reshape(images.shape[0] * images.shape[1] * images.shape[2])
        means.append(
            float(channel_stack.mean())
            )
        stds.append(
            float(channel_stack.std())
            )
        # https://pytorch.org/docs/stable/generated/torch.stack.html#torch-stack
        # Documentation on torch.stack, concatenates a sequence of tensors along a new dimension
    print(means, stds)
    return images, torch.tensor(bboxes), tuple(means), tuple(stds)

class Transformations(transforms.Compose):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, img):
        return 
