import cv2
import numpy as np
import os

# Function to generate frames with gradually changing intensity
def generate_frames(image, num_frames, intensity_inteval = 800): ## smaller intensity_inteval, more obivious the intensity change
    for i in range(num_frames):
        # Interpolate intensity from 0 to 1
        alpha = (i + intensity_inteval) / ((num_frames - 1) + intensity_inteval)

        print('Current alpha: ', alpha)
        
        # Create a blended image with varying intensity
        blended_image = cv2.addWeighted(image, alpha, np.zeros_like(image), 1 - alpha, 0)

        yield blended_image

def image_2_video_increase_light_intensity(image, output_video_frame_dir, frame_rate, duration_sec, increase_inten_flag = True):
    frame_height, frame_width, _ = image.shape
    num_frames = duration_sec * frame_rate

    all_frames = []
    for frame in generate_frames(image, num_frames):
        all_frames.append(frame)

    for single_frame_index in range(len(all_frames)):
        current_image_savename = os.path.join(output_video_frame_dir, '{0:03}'.format(single_frame_index) + '.png')
        current_frame = all_frames[single_frame_index]
        cv2.imwrite(current_image_savename, current_frame)


def image_2_video_decrease_light_intensity(image, output_video_frame_dir, frame_rate, duration_sec, increase_inten_flag = False):
    frame_height, frame_width, _ = image.shape
    num_frames = duration_sec * frame_rate

    all_frames = []
    for frame in generate_frames(image, num_frames):
        all_frames.append(frame)

    all_frames = all_frames[::-1]
    for single_frame_index in range(len(all_frames)):
        current_image_savename = os.path.join(output_video_frame_dir, '{0:03}'.format(single_frame_index) + '.png')
        current_frame = all_frames[single_frame_index]
        cv2.imwrite(current_image_savename, current_frame)


def image_2_video_remain_light_intensity(image, output_video_frame_dir, frame_rate, duration_sec, increase_inten_flag = False):
    frame_height, frame_width, _ = image.shape
    num_frames = duration_sec * frame_rate

    all_frames = []
    for _ in range(num_frames):
        all_frames.append(image)

    # all_frames = all_frames[::-1]
    for single_frame_index in range(len(all_frames)):
        current_image_savename = os.path.join(output_video_frame_dir, '{0:03}'.format(single_frame_index) + '.png')
        current_frame = all_frames[single_frame_index]
        cv2.imwrite(current_image_savename, current_frame)