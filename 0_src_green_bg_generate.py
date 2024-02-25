import cv2
from glob import glob
import os
import numpy as np
DEFAULT_IMAGE_WIDTH = 1920
DEFAULT_IMAGE_HEIGHT = 1080

# https://www.rapidtables.com/web/color/green-color.html
round_1_green_colors = [(8, 143, 143), (175, 225, 175), (170, 255, 0), (80, 200, 120), (34, 139, 34), (124, 252, 0), (0, 128, 0), (53, 94, 59), (0, 163, 108),
                (42, 170, 138), (76, 187, 23), (144, 238, 144), (50, 205, 50), (11, 218, 81), (152, 251, 152), (15, 255, 80), (147, 197, 114),
                (46, 139, 87), (0, 158, 96), (0, 255, 127), (0, 255, 0)]


def create_green_bg_image(data_save_dir):
    width = DEFAULT_IMAGE_WIDTH
    height = DEFAULT_IMAGE_HEIGHT

    for index in range(len(round_1_green_colors)):

        green_color = round_1_green_colors[index]  # BGR tuple for green color
        green_image = np.full((height, width, 3), green_color, dtype=np.uint8)

        # Save the image
        data_save_path = os.path.join(data_save_dir, str(index) + '.png')
        cv2.imwrite(data_save_path, green_image)


if __name__ == '__main__':

    green_img_save_path = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_default_bg_imgs"
    # green_img_save_path = os.path.join(green_img_save_path, 'bg.png')
    os.makedirs(green_img_save_path, exist_ok=True)
    create_green_bg_image(green_img_save_path)