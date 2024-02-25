import Augmentor
import os
import shutil

results_save_dir = r"F:\IMAGE_MATTING\Image_matting_dataset\0_using_separate_resized\aug_raw"
if os.path.exists(results_save_dir):
    shutil.rmtree(results_save_dir)

os.makedirs(results_save_dir, exist_ok=True)

raw_img_dir = r"F:\IMAGE_MATTING\Image_matting_dataset\0_using_separate_resized\0_raw_data\fg_images"
p = Augmentor.Pipeline(raw_img_dir, results_save_dir)
p = Augmentor.Pipeline(raw_img_dir)

p.rotate(probability=0.7, max_left_rotation=30, max_right_rotation=30)
p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=0.5, percentage_area=0.8)
# p.resize(probability=1.0, width=120, height=120)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.4)

p.random_distortion(probability=0.2, grid_width=20, grid_height=20, magnitude=4)

p.sample(10)

p.process()

# p.ground_truth("/path/to/ground_truth_images")