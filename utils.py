from typing import Dict, List, Tuple
from torch import Tensor
import torch
import cv2
import os
import json
import numpy as np

def import_data(annotations_path, images_path, channels):
    # List containing each image in a Tensor form (W, H, R, G, B)
    images: List[Tensor] = []
    # List containing tuples where each tuple represents a bbox for 1 image (x0, y0, x1, y0)
    bboxes: List[Tuple] = []
    

    annotations_files = os.listdir(annotations_path)
    for json_file_name in annotations_files:
        file_path = annotations_path + json_file_name
        with open(file_path, 'r') as f:
            annotation: Dict = json.load(f)[0] # as it is a list with one element the dict

            #Images
            image = cv2.imread(images_path + annotation["image"])
            image = torch.tensor(image, dtype=float).permute(1,0,2) # (X, Y, RGB) (W,H,RGB) we do this to match with the bboxes coordinates (x,y)
            print(image.shape[:2])
            (w, h) = image.shape[:2]
            image = (image * 2)/255. - 1 # normalization [-1, 1]
            images.append(image)

            # Bounding boxes
            bbox = annotation['annotations'][0]["coordinates"] # {x, y, width, height}
            # Normalize bounding boxes too
            x0 = bbox['x'] / w
            y0 = bbox['y'] / h 
            x1 = x0 + bbox['width'] / w
            y1 = y0 + bbox['height'] / h
            bboxes.append((x0, y0, x1, y1))

        # Compute the mean and std of the dataset to perform standarization
        # Using tuple because list wasn't working
        stack = torch.stack(tuple(images))
        means = []
        stds = []
        for channel in range(channels):
            channel_stack: torch.Tensor = stack[:, :, :, channel].reshape(stack.shape[0] * stack.shape[1] * stack.shape[2])
            means.append(
                float(channel_stack.mean())
                )
            stds.append(
                float(channel_stack.std())
                )
        return torch.tensor(images), torch.tensor(bboxes), tuple(means), tuple(stds)