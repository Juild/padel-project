
from typing import Dict, List, Tuple
from torch import Tensor
import torch
from torch.utils.data import Dataset
from numpy import array

class ImageDataset(Dataset):
    def __init__(self, images_ball: List[Tensor], images_no_ball: Tensor, transforms=None, target_transforms=None) -> None:
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.images_ball: Tensor = images_ball
        self.images_no_ball: Tensor = images_no_ball
        self.images: Tensor = torch.concat()

    def __getitem__(self, idx): 
        image = self.images[idx]
        bbox = self.bboxes[idx]

        #  All Pytorch models need their input to be "channel first"
        # i.e. TensorSize(C, H, W)
        image = image.permute(2, 0, 1)
        if self.transforms:
            # Transform image
            self.transforms(image)
        if self.target_transforms:
            self.target_transforms()

        return (image, bbox)

    def __len__(self):
        # return the size of the first index in the stack of image tensors, i.e. the number of images tensors
        # that is, the number of images in the dataset
        return self.images.size(0)

