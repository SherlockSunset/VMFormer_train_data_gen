import cv2
import os
import random
import numpy as np
from glob import glob
import shutil

green_bg_lower_ratio = 0.1
green_bg_upper_ratio = 0.3

def add_green_light(image_path, pha_image_path, green_img):
    # Load the image
    img_raw = cv2.imread(image_path)

    height, width, _ = img_raw.shape

    # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    mask_raw = cv2.imread(pha_image_path)
    # mask_raw[mask_raw < 255] = 0
    mask = mask_raw / 255

    random_float = random.uniform(green_bg_lower_ratio, green_bg_upper_ratio)

    print('Random num: ', random_float)

    green_img = cv2.resize(green_img, (width, height))

    green_img = green_img*mask
    green_light_layer = green_img.astype(np.uint8)

    # Blend the original image with the green light layer
    blended_image = cv2.addWeighted(img_raw, 1 - random_float, green_light_layer, random_float, 0)

    return blended_image, mask_raw

if __name__ == '__main__':
    current_data_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/VideoMatte240K_JPEG_SD/train"
    # current_data_dir = os.path.join(current_data_dir, 'fgr')
    # os.rename(current_data_dir, current_data_dir + '_raw')
    dst_save_dir = current_data_dir + '_green'
    os.makedirs(dst_save_dir, exist_ok = True)

    img_format = '.png'

    current_fgr_data_dir = os.path.join(current_data_dir, 'fgr')
    all_images = glob(os.path.join(current_fgr_data_dir, '*.png'))

    if len(all_images) == 0:
        img_format = '.jpg'
        all_images = glob(os.path.join(current_fgr_data_dir, '*.jpg'))
    all_images = sorted(all_images)
    
    current_pha_data_dir = os.path.join(current_data_dir, 'pha')
    all_pha_images = glob(os.path.join(current_pha_data_dir, '*' + img_format))
    all_pha_images = sorted(all_pha_images)

    all_green_bg_imgs = []
    green_bg_image_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_bg_dataset/augmented_bg_noise"
    all_green_bg_img_paths = glob(os.path.join(green_bg_image_dir, '*.png'))

    for green_bg_img_path in all_green_bg_img_paths:
        # green_image_path = r"F:\4_image_matting_dataset\test\green_bg.jpg"
        green_img = cv2.imread(green_bg_img_path)
        all_green_bg_imgs.append(green_img)

    
    
    
    dst_fgr_save_dir = os.path.join(dst_save_dir, 'fgr')
    os.makedirs(dst_fgr_save_dir, exist_ok = True)

    for image_path, pha_image_path in zip(all_images, all_pha_images):
        # Read the image
        # image = cv2.imread(image_path)
        random_index = random.randint(0, len(all_green_bg_imgs))

        green_img = all_green_bg_imgs[random_index]

        # Add the green light effect
        result, mask = add_green_light(image_path, pha_image_path, green_img)

        image_savename = os.path.join(dst_fgr_save_dir, os.path.basename(image_path))

        print('Current image save name: ', image_savename)
        cv2.imwrite(image_savename, result)

        # image_savename = os.path.join(dst_save_dir, os.path.basename(image_path).split('.')[0] + '_mask.png')
        # cv2.imwrite(image_savename, mask)

    

    dst_pha_save_dir = os.path.join(dst_save_dir, 'pha')
    os.makedirs(dst_pha_save_dir, exist_ok = True)

    for pha_image in all_pha_images:
        dst_pha_path = os.path.join(dst_pha_save_dir, os.path.basename(pha_image))
        shutil.copyfile(pha_image, dst_pha_path)
