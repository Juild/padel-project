
from typing import Dict, List, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from numpy import array

class ImageDataset(Dataset):
    def __init__(self, images: Tensor, bboxes: Tensor, transforms=None, target_transforms=None) -> None:
        self.transforms = transforms
        self.target_transforms = target_transforms
        # Datapoint index is the first one
        # i.e. tensor(image_0: tensor, image_1: tensor,... , image_n: tensor)
        # where n is the number of images (datapoints)
        self.images: Tensor[Tensor] = images
        # Datapoint index is the first one
        # i.e. tensor(bbox_0 : tensor, bbox_1: tensor,... , bbox_n: tensor)
        # where n is the number of images (datapoints) each bbox is associated respectively
        # with the image in the same index in self.images tensor.
        self.bboxes: Tuple[Tuple] = bboxes

    def __getitem__(self, idx): 
        image = self.images[idx]
        bbox = self.bboxes[idx]
        if self.transforms:
            # Transform image
            self.transforms(image)
        if self.target_transforms:
            self.target_transforms()

        return (image, bbox)

    def __len__(self):
        return len(self.images)

