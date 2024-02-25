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


def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to the image.
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_pepper_amount=0.01):
    """
    Adds salt and pepper noise to the image.
    """
    salt_pepper = np.random.rand(*image.shape)
    noisy_image = np.copy(image)
    noisy_image[salt_pepper < salt_pepper_amount] = 0
    noisy_image[salt_pepper > 1 - salt_pepper_amount] = 255
    return noisy_image


def add_uniform_noise(image, low=-25, high=25):
    """
    Adds uniform noise to the image.
    """
    noise = np.random.uniform(low, high, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image