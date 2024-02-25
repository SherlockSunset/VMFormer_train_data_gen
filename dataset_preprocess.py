import os
import cv2
import numpy as np
import shutil


standard_size_w = 1080

def create_alpha_dataset(src_data_dir, dst_data_dir):
    os.makedirs(dst_data_dir, exist_ok=True)

    dst_image_dir = os.path.join(dst_data_dir, 'alpha')
    os.makedirs(dst_image_dir, exist_ok=True)

    all_image_files_ori = []
    for root, dirs, files in os.walk(src_data_dir):

        for file_ in files:
            if file_.endswith('.png') or file_.endswith('.jpg'):
                current_file_path = os.path.join(root, file_)

                all_image_files_ori.append(current_file_path)

    all_image_files = sorted(all_image_files_ori)

    print('Image number: ', len(all_image_files))

    for image_file_index in range(len(all_image_files)):
        # if image_file_index <= 1000:
            # if image_file_index > 1000 and image_file_index <= 1200:
            image_file = all_image_files[image_file_index]
            dst_image_path = os.path.join(dst_image_dir, os.path.basename(image_file))

            print('Current image file: ', image_file)
            print('Dst image file: ', dst_image_path)

            in_image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            alpha = in_image[:, :, 3]

            cv2.imwrite(dst_image_path, alpha)


def obtain_fg_human_imgs(input_src_data_dir, dst_data_dir):
    os.makedirs(dst_data_dir, exist_ok=True)

    dst_image_dir = os.path.join(dst_data_dir, 'fg_images')
    os.makedirs(dst_image_dir, exist_ok=True)

    src_data_raw_imgs_dir = os.path.join(input_src_data_dir, 'image')
    assert os.path.exists(src_data_raw_imgs_dir)

    src_data_raw_alpha_dir = os.path.join(input_src_data_dir, 'alpha')
    assert os.path.exists(src_data_raw_alpha_dir)

    all_raw_image_files_ori = []
    for root, dirs, files in os.walk(src_data_raw_imgs_dir):

        for file_ in files:
            if file_.endswith('.png') or file_.endswith('.jpg'):
                current_file_path = os.path.join(root, file_)

                all_raw_image_files_ori.append(current_file_path)
    all_image_files = sorted(all_raw_image_files_ori)

    all_alpha_files_ori = []
    for root, dirs, files in os.walk(src_data_raw_alpha_dir):

        for file_ in files:
            if file_.endswith('.png') or file_.endswith('.jpg'):
                current_file_path = os.path.join(root, file_)

                all_alpha_files_ori.append(current_file_path)

    all_alpha_files = sorted(all_alpha_files_ori)
    assert len(all_image_files) == len(all_alpha_files)

    print('Image number: ', len(all_image_files))


    for image_file, alpha_file in zip(all_image_files, all_alpha_files):
        print('Current image file: ', os.path.basename(image_file))
        print('Current alpha file: ', os.path.basename(alpha_file))
        image = cv2.imread(image_file)
        alpha = cv2.imread(alpha_file, 0)

        alpha_rgb = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)

        print('Image shape: ', image.shape)
        print('Alpha shape: ', alpha_rgb.shape)
        # Convert uint8 to float
        image = image.astype(float)
        # background = background.astype(float)

        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha_rgb = alpha_rgb.astype(float) / 255

        foreground = cv2.multiply(alpha_rgb, image)

        current_fg_image_save_path = os.path.join(dst_image_dir, os.path.basename(image_file).split('.')[0] + '.png')
        print('Current save image: ', current_fg_image_save_path)
        cv2.imwrite(current_fg_image_save_path, foreground)


## combine all fg and alpha images from several different sub-folders
def cluster_all_fg_images_and_alpha(overall_data_dir, dst_data_dir):
    all_raw_image_files_ori = []
    for root, dirs, files in os.walk(overall_data_dir):
        if 'fg_images' in root:
            for file_ in files:
                if file_.endswith('.png') or file_.endswith('.jpg'):
                    current_file_path = os.path.join(root, file_)

                    all_raw_image_files_ori.append(current_file_path)
    all_image_files = sorted(all_raw_image_files_ori)

    print('All image files: ', len(all_image_files))

    dst_data_fg_image_folder = os.path.join(dst_data_dir, 'fg_images')
    os.makedirs(dst_data_fg_image_folder, exist_ok=True)

    dst_data_alpha_image_folder = os.path.join(dst_data_dir, 'alpha')
    os.makedirs(dst_data_alpha_image_folder, exist_ok=True)

    for image_file_index in range(len(all_image_files)):

        current_fg_image_file = all_image_files[image_file_index]
        print('Current image name: ', current_fg_image_file)
        current_alpha_image_files = current_fg_image_file.replace('fg_images', 'alpha')

        current_fg_image = cv2.imread(current_fg_image_file)

        if not os.path.exists(current_alpha_image_files):
            current_alpha_image_files = current_alpha_image_files.split('.')[0] + '.jpg'
        current_alpha_image = cv2.imread(current_alpha_image_files)

        image_name = '{0:06}.png'.format(image_file_index)
        dst_fg_image_file = os.path.join(dst_data_fg_image_folder, image_name)
        # shutil.copyfile(current_fg_image_file, dst_fg_image_file)
        cv2.imwrite(dst_fg_image_file, current_fg_image)

        dst_alpha_image_file = os.path.join(dst_data_alpha_image_folder, image_name)
        # shutil.copyfile(current_alpha_image_files, dst_alpha_image_file)
        cv2.imwrite(dst_alpha_image_file, current_alpha_image)



def load_img_and_resize(overall_data_dir, dst_data_dir):
    all_raw_image_files_ori = []
    for root, dirs, files in os.walk(overall_data_dir):
        if 'fg_images' in root:
            for file_ in files:
                if file_.endswith('.png') or file_.endswith('.jpg'):
                    current_file_path = os.path.join(root, file_)

                    all_raw_image_files_ori.append(current_file_path)
    all_image_files = sorted(all_raw_image_files_ori)

    print('All image files: ', len(all_image_files))

    dst_data_fg_image_folder = os.path.join(dst_data_dir, 'fg_images')
    os.makedirs(dst_data_fg_image_folder, exist_ok=True)

    dst_data_alpha_image_folder = os.path.join(dst_data_dir, 'alpha')
    os.makedirs(dst_data_alpha_image_folder, exist_ok=True)

    for image_file_index in range(len(all_image_files)):

        current_fg_image_file = all_image_files[image_file_index]
        print('Current image name: ', current_fg_image_file)
        current_alpha_image_files = current_fg_image_file.replace('fg_images', 'alpha')

        current_fg_image = cv2.imread(current_fg_image_file)

        current_fg_image_h, current_fg_image_w, _ = current_fg_image.shape

        # w_ratio = current_fg_image_w / standard_size_w
        # h_ratio = current_fg_image_h / standard_size_h

        new_height = int(current_fg_image_h * (standard_size_w / current_fg_image_w))

        # if w_ratio >= h_ratio:
        current_fg_image = cv2.resize(current_fg_image, (standard_size_w, new_height))
        # w_ratio = h_ratio

        if not os.path.exists(current_alpha_image_files):
            current_alpha_image_files = current_alpha_image_files.split('.')[0] + '.jpg'
        current_alpha_image = cv2.imread(current_alpha_image_files)

        current_alpha_image = cv2.resize(current_alpha_image, (standard_size_w, new_height))

        image_name = '{0:06}.png'.format(image_file_index)
        dst_fg_image_file = os.path.join(dst_data_fg_image_folder, image_name)
        # shutil.copyfile(current_fg_image_file, dst_fg_image_file)
        cv2.imwrite(dst_fg_image_file, current_fg_image)

        dst_alpha_image_file = os.path.join(dst_data_alpha_image_folder, image_name)
        # shutil.copyfile(current_alpha_image_files, dst_alpha_image_file)
        cv2.imwrite(dst_alpha_image_file, current_alpha_image)


if __name__ == '__main__':
    # # option = 1
    # input_src_data_dir = r"F:\IMAGE_MATTING\Image_matting_dataset\RealWorldPortrait-636"
    # # dst_data_dir =r""
    # obtain_fg_human_imgs(input_src_data_dir, input_src_data_dir)

    ### create alpha dataset
    src_data_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/PhotoMatte85_with_alpha/using_data/train/fgr"
    dst_data_dir = os.path.dirname(src_data_dir)
    create_alpha_dataset(src_data_dir, dst_data_dir)

    # overall_data_dir = r"F:\IMAGE_MATTING\Image_matting_dataset\0_using"
    # dst_data_dir = overall_data_dir + '_merged'
    # os.makedirs(dst_data_dir, exist_ok=True)
    # cluster_all_fg_images_and_alpha(overall_data_dir, dst_data_dir)

    # overall_data_dir = r"F:\IMAGE_MATTING\Image_matting_dataset\0_using_separate"
    # dst_data_dir = overall_data_dir + '_resized'
    # os.makedirs(dst_data_dir, exist_ok=True)

    # load_img_and_resize(overall_data_dir, dst_data_dir)



