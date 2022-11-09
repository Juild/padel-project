
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch
import cv2
import os
import json
from typing import Dict, List
from torch import Tensor
class ImageDataset(Dataset):
    def __init__(self, annotations_path, images_path, channels, transforms=None) -> None:
        self.transforms: Compose = transforms
        self.images: List[Tensor] = []
        self.image_bboxes: List[Dict] = []
        
        annotations_files = os.listdir(annotations_path)
        for json_file_name in annotations_files:
            file_path = annotations_path + json_file_name
            f = open(file_path, 'r')
            annotation: Dict = json.load(f)[0] # as it is a list with one element the dict
            coordinates: Dict = {
                "boxes": []
            }
            
            for bbox in annotation["annotations"]:
                coordinates["boxes"].append(bbox["coordinates"])

            self.image_bboxes.append(coordinates)
            image = cv2.imread(images_path + annotation["image"])
            image = torch.tensor(image, dtype=float).permute(1,0,2) # (X, Y, RGB) (W,H,RGB) we do this to match with the bboxes coordinates (x,y)
            image = (image * 2)/255. - 1 # normalization [-1, 1]
            self.images.append(image)
            f.close()
        # Compute the mean and std of the dataset to perform standarization
        stack = torch.stack(tuple(self.images))
        self.means = []
        self.stds = []
        for channel in range(channels):
            channel_stack: torch.Tensor = stack[:, :, :, channel].reshape(stack.shape[0] * stack.shape[1] * stack.shape[2])
            self.means.append(
                float(channel_stack.mean())
                )
            self.stds.append(
                float(channel_stack.std())
                )

    def __getitem__(self, idx): 
        if self.transforms:
            return (self.transforms(self.images[idx]) , self.image_bboxes[idx])
        else:
            return (self.images[idx], self.image_bboxes[idx])
    def __len__(self):
        return len(self.images)

