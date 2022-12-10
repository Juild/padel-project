
from typing import Dict, List, Tuple
from torch import Tensor
import torch
from torch.utils.data import Dataset
from numpy import array

class ImageDataset(Dataset):
    def __init__(self, images_ball: List[Tensor], images_no_ball: List[Tensor], transforms=None, target_transforms=None) -> None:
        self.transforms = transforms
        self.target_transforms = target_transforms
        print(images_ball.shape)
        assert images_ball.shape[1:] == (60, 60, 3), "Image shape not correct"
        self.n_images_ball = images_ball.shape[0]
        self.n_images_no_ball = images_no_ball.shape[0]
        self.images: Tensor = torch.concat((images_ball, images_no_ball))
        self.labels = torch.Tensor([1, 0]) # we only have ball and background
    
    def __getitem__(self, idx): 
        image = self.images[idx]
        if idx < self.n_images_ball:
            label = self.labels[0] # ball has label "1"
        else:
            label = self.labels[1]

        #  All Pytorch models need their input to be "channel first"
        # i.e. TensorSize(C, H, W)
        image = image.permute(2, 0, 1)
        if self.transforms:
            # Transform image
            self.transforms(image)

        return image, label

    def __len__(self):
        # return the size of the first index in the stack of image tensors, i.e. the number of images tensors
        # that is, the number of images in the dataset
        return self.images.size(0)

