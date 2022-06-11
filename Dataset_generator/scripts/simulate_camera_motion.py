import argparse
from tqdm import tqdm
import os
import cv2
import numpy as np
import random
from PIL import Image

def motion_blur(img):
    size_list = [9, 11, 13, 15, 17, 19, 21, 23]

    size = random.choice(size_list)

    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    output = cv2.filter2D(np.array(img).astype(np.uint8), -1, kernel_motion_blur)

    return Image.fromarray(output)

def generate_motion(image_dir, output_dir, frac):
    image_list = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"): 
            image_list.append(filename)

    for i in range(int(len(image_list)*frac)):
        random_item_from_list = random.choice(image_list)
        image_list.remove(random_item_from_list)

    for image in tqdm(image_list, desc='generation motion files'):
        img = Image.open(os.path.join(image_dir, image))
        new_img = motion_blur(img)
        new_img.save(os.path.join(output_dir, image))

if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description=
       'Reads the images in a given folder and generate new images applying a camera motion effect'
   )

   parser.add_argument('image_dir',
                       metavar='image_dir',
                       type=str,
                       help='Path to the images')

   parser.add_argument(
       'output_dir',
       metavar='output_dir',
       type=str,
       help='Directory where the new images will be saved'
   )
   parser.add_argument(
       'frac',
       metavar='frac',
       type=float,
       default=1.0,
       help='fraction of the images in directory that will be used to generate new "motion" images'
   )

   args = parser.parse_args()

   generate_motion(args.image_dir, args.output_dir, args.frac)
