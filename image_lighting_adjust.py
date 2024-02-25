import cv2
from glob import glob
import os
import numpy as np


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


import cv2
import numpy as np

def adjust_global_intensity(image, alpha, beta):
    """
    Adjusts the global intensity of an image using linear transformation:
    output = alpha * input + beta
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def vary_intensity_from_left_2_right(image, intensity_variation_factor, blur_kernel):
    """
    Varies the intensity of a pure color image locally patch by patch.

    Parameters:
        image: Input pure color image.
        intensity_variation_factor: Factor to vary the intensity.

    Returns:
        Image with varied intensity locally.
    """
    height, width, _ = image.shape

    # Create a grayscale image with varying intensity
    gray_intensity = np.linspace(0, 255, width).astype(np.uint8)
    gray_image = np.tile(gray_intensity, (height, 1))

    # Iterate over image patches
    for y in range(0, height, gray_image.shape[0]):
        for x in range(0, width, gray_image.shape[1]):
            # Get the current patch
            patch = image[y:y+gray_image.shape[0], x:x+gray_image.shape[1]]

            # Convert grayscale image to 3 channels
            gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Blend the grayscale image with the current patch
            blended_patch = cv2.addWeighted(patch, 1.0 - intensity_variation_factor, gray_image_3ch, intensity_variation_factor, 0)

            # Replace the current patch with the blended patch
            image[y:y+gray_image.shape[0], x:x+gray_image.shape[1]] = blended_patch

    blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    return blurred_image


def vary_intensity_random_direction(image, color_patch_size, color, intensity_variation_factor, blur_kernel):
    """
    Varies the intensity of a pure color image locally patch by patch, with color variation from an arbitrary direction.

    Parameters:
        image: Input pure color image.
        color_patch_size: Size of the color patch.
        color: RGB tuple for the pure color.
        intensity_variation_factor: Factor to vary the intensity.

    Returns:
        Image with varied intensity locally.
    """
    height, width, _ = image.shape

    # Create a pure color patch
    color_patch = np.full((color_patch_size, color_patch_size, 3), color, dtype=np.uint8)

    # Compute the direction of color variation
    direction = np.random.randn(2)

    # Iterate over image patches
    for y in range(0, height, color_patch_size):
        for x in range(0, width, color_patch_size):
            patch = image[y:y+color_patch_size, x:x+color_patch_size]
            if patch.shape[0] == color_patch_size and patch.shape[1] == color_patch_size:
                # Get the current patch
            
                # color_patch = color_patch[0:patch.shape[0], 0:patch.shape[1], :]

                print('SHape of patch: ', patch.shape)
                print('SHape of color_patch: ', color_patch.shape)

                # Compute intensity variation based on patch position and direction
                intensity_variation = np.dot(direction, np.array([y, x])) / (height + width)

                # Blend the color patch with the current patch
                blended_patch = cv2.addWeighted(patch, 1.0 - intensity_variation_factor * intensity_variation,
                                                color_patch, intensity_variation_factor * intensity_variation, 0)

                # Replace the current patch with the blended patch
                image[y:y+color_patch_size, x:x+color_patch_size] = blended_patch

    blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    return blurred_image


def create_green_bg_image(data_save_path):

    # image_size = 

    width = 1820
    height = 920
    green_color = (0, 255, 0)  # BGR tuple for green color
    green_image = np.full((height, width, 3), green_color, dtype=np.uint8)

    # Save the image
    cv2.imwrite(data_save_path, green_image)





if __name__ == '__main__':

    # green_img_save_path = r"F:\4_image_matting_dataset\1_default_bg_imgs"
    # green_img_save_path = os.path.join(green_img_save_path, 'bg.png')

    # create_green_bg_image(green_img_save_path)

    default_bg_image_dir = r"F:\4_image_matting_dataset\1_default_bg_imgs\raw"
    default_bg_image_paths = glob(os.path.join(default_bg_image_dir, '*.png'))

    data_save_dir = os.path.join(os.path.dirname(default_bg_image_dir), 'augmented_bg')
    os.makedirs(data_save_dir, exist_ok = True)



    for image_path in default_bg_image_paths:
        image = cv2.imread(image_path)

        # Global intensity adjustment
        global_adjusted_image = adjust_global_intensity(image, alpha=1.5, beta=10)

        height, width, _ = image.shape
        gammas = np.zeros((height, width), dtype=np.float32)
        for i in range(height):
            for j in range(width):
                # Example: Adjust intensity based on pixel position
                gammas[i, j] = (i + j) / (height + width)
        # Specify parameters
        color = (0, 0, 0)
        blur_kernel = 25
        color_patch_size = 50
        intensity_variation_factor = 0.1  # Example: 0.5 means half intensity variation

        # Vary intensity of the pure color image locally
        # intensity_varied_image = vary_intensity(image, color_patch_size, color, intensity_variation_factor, blur_kernel)

        intensity_varied_image = vary_intensity_from_left_2_right(image, intensity_variation_factor, blur_kernel)


        global_ligh_adjusted_save_path = os.path.join(data_save_dir, 'global.png')
        cv2.imwrite(global_ligh_adjusted_save_path, global_adjusted_image)

        local_ligh_adjusted_save_path = os.path.join(data_save_dir, 'local.png')
        cv2.imwrite(local_ligh_adjusted_save_path, intensity_varied_image)