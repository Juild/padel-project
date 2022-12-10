from typing import Dict, List, Tuple, Union
from torch import Tensor
from torchvision import transforms
import torch
import cv2
import numpy as np
import copy
import random
import os

def extract_frames(video_path: str):

    # Open the video file
    video = cv2.VideoCapture(video)

    # Get the total number of frames in the video
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_frames)
    # Iterate over the frames and save each one to a separate image file
    for frame_num in range(int(total_frames)):
        # Read the current frame
        _, frame = video.read()
        
        # Save the current frame as an image file
        cv2.imwrite("frame_{}.jpg".format(frame_num), frame)

    # Close the video file
    video.release()


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

def blur_images(chunks1, chunks2):
    kernel = np.ones((5, 5), dtype=np.float32)/25
    print(type(chunks2[0]))
    chunks1_filtered = []
    chunks2_filtered = []
    for block in chunks1:
        for image in block:
            image = cv2.filter2D(image, -1, kernel)
            chunks1_filtered.append(image)

    for block in chunks2:
        for image in block:
            image = cv2.filter2D(image, -1, kernel)
            chunks2_filtered.append(image)
    return chunks1_filtered, chunks2_filtered
    

def import_data(images_path: str):
    all_chunks_ball = []
    all_chunks_no_ball = []
    for image_file_name in os.listdir(images_path):
        print(image_file_name)
        image = cv2.imread(images_path + image_file_name)
        image = remove_score_card(image)
        image_chunks = split_image_into_chunks(image)
        # We do a deep copy, otherwise it just gets the same reference of the objects
        # which means that both lists (with and without circles) will share the
        # reference to the same object so both lists will be the same.
        chunks_without_ball = copy.deepcopy(image_chunks) 
        chunks_with_ball = draw_random_circles(image_chunks)
        all_chunks_ball.append(chunks_with_ball)
        all_chunks_no_ball.append(chunks_without_ball)

    # all_chunks_no_ball, all_chunks_ball = blur_images(all_chunks_no_ball, all_chunks_ball)
    all_chunks_ball = torch.Tensor(np.array(all_chunks_ball))
    all_chunks_no_ball = torch.Tensor(np.array(all_chunks_no_ball))
    print(all_chunks_no_ball.shape)

    all_chunks_ball = all_chunks_ball.reshape(
        all_chunks_ball.shape[0] * all_chunks_ball.shape[1],
        all_chunks_ball.shape[2],
        all_chunks_ball.shape[3],
        all_chunks_ball.shape[4],
        
    )
    all_chunks_no_ball = all_chunks_no_ball.reshape(
        all_chunks_no_ball.shape[0] * all_chunks_no_ball.shape[1],
        all_chunks_no_ball.shape[2],
        all_chunks_no_ball.shape[3],
        all_chunks_no_ball.shape[4],
    )
    return all_chunks_ball, all_chunks_no_ball


def import_test_image(image_path: str):
    image = cv2.imread(image_path)
    image = remove_score_card(image)
    image_chunks = split_image_into_chunks(image)
    return torch.Tensor( np.array(image_chunks) )