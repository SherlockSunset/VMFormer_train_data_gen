import cv2
import numpy as np
import os
from glob import glob

from image_2_video_augment_functions import image_2_video_decrease_light_intensity, image_2_video_increase_light_intensity, image_2_video_remain_light_intensity


def main_entrance(src_bg_image_paths, data_save_dir_input, duration_sec = 4, frame_rate = 25):
        
    for src_bg_image_path in src_bg_image_paths:
        # Load the single image
        image = cv2.imread(src_bg_image_path)

        current_video_frame_save_dir = os.path.join(data_save_dir_input, os.path.basename(src_bg_image_path).split('.')[0] + '_light_incre')
        os.makedirs(current_video_frame_save_dir, exist_ok=True)

        # image, output_video_path, frame_rate, duration_sec
        image_2_video_increase_light_intensity(image, current_video_frame_save_dir, frame_rate, duration_sec)

        current_video_frame_save_dir = os.path.join(data_save_dir_input, os.path.basename(src_bg_image_path).split('.')[0] + '_light_decre')
        os.makedirs(current_video_frame_save_dir, exist_ok=True)

        # image, output_video_path, frame_rate, duration_sec
        image_2_video_decrease_light_intensity(image, current_video_frame_save_dir, frame_rate, duration_sec)

        current_video_frame_save_dir = os.path.join(data_save_dir_input, os.path.basename(src_bg_image_path).split('.')[0] + '_light_remain')
        os.makedirs(current_video_frame_save_dir, exist_ok=True)

        # image, output_video_path, frame_rate, duration_sec
        image_2_video_remain_light_intensity(image, current_video_frame_save_dir, frame_rate, duration_sec)


if __name__ == '__main__':
    # # Directory containing input images
    # input_dir = r'F:\4_image_matting_dataset\1_default_bg_imgs\augmented_bg\global.png'

    # # Output video file name
    # output_video = r'F:\4_image_matting_dataset\1_default_bg_imgs\augmented_bg\output.mp4'

    # # import cv2
    # # import numpy as np

    src_aug_bg_image_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_bg_dataset/augmented_bg_noise"
    default_bg_image_paths_input = glob(os.path.join(src_aug_bg_image_dir, '*.png'))

    data_save_dir = os.path.join(os.path.dirname(src_aug_bg_image_dir), 'augmented_bg_noise_video')
    os.makedirs(data_save_dir, exist_ok = True)

    main_entrance(default_bg_image_paths_input, data_save_dir)


    
