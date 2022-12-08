from typing import Dict, List, Tuple, Union
from torch import Tensor
from torchvision import transforms
import torch
import cv2
import os
import json
import numpy as np

import random
def remove_score_card(image):
    score_card_coordinates = (50, 80, 364, 187) #(x0, y0, x1, y1)
    image = cv2.rectangle(image,
                        pt1=(score_card_coordinates[0], score_card_coordinates[1]),
                        pt2=(score_card_coordinates[2], score_card_coordinates[3]),
                        color=(0, 0, 0), #TODO is there any way to use enum for selecting the color instead of a raw BGR tuple?
                        thickness=-1 # filled rectangle

                        )
    return image

def split_image_into_chunks(image):
    # Ball radius should be between 5 and 10 pixels, so I would start by making the chunks 60x60 pixels
    # Split the image into 20x20 pixel subimages
    width, height = image.shape[:2]
    split_into = 60
    subimages = [image[i:i+split_into, j:j+split_into] for i in range(0, width, split_into) for j in range(0, height, split_into)]
    return subimages
    
def draw_random_circles(images):
   chunks_with_ball = []
   for img in images: # get the size of the image
        height, width, _ = img.shape

        # generate random coordinates for the center of the circle
        x = random.randint(0, width)
        y = random.randint(0, height)

        # generate a random radius for the circle
        radius = random.randint(5, 10)

        # set the color of the circle to a color similar to a tennis ball
        color = (0, 255, 200)

        # draw the circle on the image
        cv2.circle(img, (x, y), radius, color, thickness=-1)
        chunks_with_ball.append(img)

   return chunks_with_ball 

def import_data(images_path='./predictions/image0.jpg'):
    image = cv2.imread(images_path)
    image = remove_score_card(image)
    image_chunks = split_image_into_chunks(image)
    chunks_without_ball = image_chunks.copy()
    chunks_with_ball = draw_random_circles(image_chunks)
    return torch.Tensor(np.array(chunks_with_ball)), torch.Tensor(np.array(chunks_without_ball))