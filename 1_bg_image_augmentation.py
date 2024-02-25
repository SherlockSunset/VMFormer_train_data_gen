import cv2
from glob import glob
import os
import numpy as np

from single_image_augment_functions import adjust_global_intensity, vary_intensity_from_left_2_right, vary_intensity_random_direction, add_gaussian_noise, add_salt_and_pepper_noise, add_uniform_noise

def intensity_augment(default_bg_image_paths, data_save_dir):
    for image_path in default_bg_image_paths:
        image = cv2.imread(image_path)

        current_imagename = os.path.basename(image_path).split('.')[0]

        # Global intensity adjustment
        global_adjusted_image = adjust_global_intensity(image, alpha=0.9, beta=10)

        height, width, _ = image.shape
        gammas = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                # Example: Adjust intensity based on pixel position
                gammas[i, j] = (i + j) / (height + width)

        # Specify parameters
        color = (0, 0, 0)
        blur_kernel = 35
        color_patch_size = 50
        intensity_variation_factor = 0.5  # Example: 0.5 means half intensity variation

        
        intensity_varied_image_1 = vary_intensity_from_left_2_right(image, intensity_variation_factor, blur_kernel)

        # Vary intensity of the pure color image locally
        intensity_variation_factor = 0.2  # Example: 0.5 means half intensity variation
        intensity_varied_image_2 = vary_intensity_random_direction(image, color_patch_size, color, intensity_variation_factor, blur_kernel)


        global_ligh_adjusted_save_path = os.path.join(data_save_dir, current_imagename + '_global.png')
        cv2.imwrite(global_ligh_adjusted_save_path, global_adjusted_image)

        local_ligh_adjusted_save_path = os.path.join(data_save_dir, current_imagename + '_local_1.png')
        cv2.imwrite(local_ligh_adjusted_save_path, intensity_varied_image_1)

        local_ligh_adjusted_save_path = os.path.join(data_save_dir, current_imagename + '_local_2.png')
        cv2.imwrite(local_ligh_adjusted_save_path, intensity_varied_image_2)


def pixel_noise_augment(default_bg_image_paths, data_save_dir):
    for image_path in default_bg_image_paths:
        image = cv2.imread(image_path)

        current_imagename = os.path.basename(image_path).split('.')[0]

        # # Add Gaussian noise
        # noisy_image_gaussian = add_gaussian_noise(image, mean=0, std=50)

        # Add salt and pepper noise
        for noise_level in range(1, 10, 2):
            salt_pepper_amount_level = noise_level*0.001
            noisy_image_salt_pepper = add_salt_and_pepper_noise(image, salt_pepper_amount=salt_pepper_amount_level)

            noisy_image_salt_save_path = os.path.join(data_save_dir, current_imagename + f'_salt_{round(salt_pepper_amount_level, 4)}.png')
            cv2.imwrite(noisy_image_salt_save_path, noisy_image_salt_pepper)

        raw_image_save_path = os.path.join(data_save_dir, os.path.basename(image_path))
        cv2.imwrite(raw_image_save_path, image)



if __name__ == '__main__':

    # green_img_save_path = r"F:\4_image_matting_dataset\1_default_bg_imgs"
    # green_img_save_path = os.path.join(green_img_save_path, 'bg.png')

    # create_green_bg_image(green_img_save_path)

    bg_dataset_save_name = 'round_2' ## with shadow effect + overall combination effect

    default_bg_image_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_default_bg_imgs"
    default_bg_image_paths_input = glob(os.path.join(default_bg_image_dir, '*.png'))

    bg_dataset_default_save_dir_root = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_bg_dataset"
    bg_dataset_default_save_dir = os.path.join(bg_dataset_default_save_dir_root, bg_dataset_save_name)
    data_save_dir_input = os.path.join(bg_dataset_default_save_dir, 'augmented_bg_inten')
    os.makedirs(data_save_dir_input, exist_ok = True)

    intensity_augment(default_bg_image_paths_input, data_save_dir_input)


    intensity_augmented_image_paths = glob(os.path.join(data_save_dir_input, '*.png'))

    data_save_dir_input = os.path.join(bg_dataset_default_save_dir, 'augmented_bg_noise')
    os.makedirs(data_save_dir_input, exist_ok = True)

    pixel_noise_augment(intensity_augmented_image_paths, data_save_dir_input)


    


    