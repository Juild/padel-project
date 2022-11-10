
from typing import Dict, List, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from numpy import array

class ImageDataset(Dataset):
    def __init__(self, images: List[Tensor], bboxes: Tensor, transforms=None, target_transforms=None) -> None:
        self.transforms = transforms
        self.target_transforms = target_transforms
        # Datapoint index is the first one
        # i.e. tensor(image_0: tensor, image_1: tensor,... , image_n: tensor)
        # where n is the number of images (datapoints)
        self.images: Tensor = images
        # Datapoint index is the first one
        # i.e. tensor(bbox_0 : tensor, bbox_1: tensor,... , bbox_n: tensor)
        # where n is the number of images (datapoints) each bbox is associated respectively
        # with the image in the same index in self.images tensor.
        self.bboxes: Tensor = bboxes

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

