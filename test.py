import os
from glob import glob
import random
import shutil
import cv2

def portrait_photo_split():

    src_data_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/PhotoMatte85_with_alpha/src"
    src_image_dir = os.path.join(src_data_dir, 'image')
    src_alpha_dir = os.path.join(src_data_dir, 'alpha')


    dst_data_save_dir = os.path.join(os.path.dirname(src_data_dir), 'using_data')
    os.makedirs(dst_data_save_dir, exist_ok = True)

    dst_data_save_dir_train = os.path.join(dst_data_save_dir, 'train')
    dst_data_save_dir_val = os.path.join(dst_data_save_dir, 'val')

    dst_image_data_save_dir_train = os.path.join(dst_data_save_dir_train, 'fgr')
    dst_alpha_data_save_dir_train = os.path.join(dst_data_save_dir_train, 'pha')
    os.makedirs(dst_image_data_save_dir_train, exist_ok = True)
    os.makedirs(dst_alpha_data_save_dir_train, exist_ok = True)


    dst_image_data_save_dir_val = os.path.join(dst_data_save_dir_val, 'fgr')
    dst_alpha_data_save_dir_val = os.path.join(dst_data_save_dir_val, 'pha')
    os.makedirs(dst_image_data_save_dir_val, exist_ok = True)
    os.makedirs(dst_alpha_data_save_dir_val, exist_ok = True)


    
    all_src_image_paths = glob(os.path.join(src_image_dir, '*.png'))
    all_src_alpha_paths = glob(os.path.join(src_alpha_dir, '*.png'))

    all_src_image_paths = sorted(all_src_image_paths)
    all_src_alpha_paths = sorted(all_src_alpha_paths)

    random.seed(66)
    random.shuffle(all_src_image_paths)
    # random.shuffle(all_src_alpha_paths)

    total_image_num = len(all_src_image_paths)

    train_src_image_paths = all_src_image_paths[:int(total_image_num*0.9)]
    val_src_image_paths = all_src_image_paths[int(total_image_num*0.9):]

    # train_src_alpha_paths = all_src_alpha_paths[:int(total_image_num*0.9)]
    # val_src_alpha_paths = all_src_alpha_paths[int(total_image_num*0.9):]


    ## train data
    for image_path in train_src_image_paths:
        image_save_path = os.path.join(dst_image_data_save_dir_train, os.path.basename(image_path).split('.')[0] + '.jpg')
        # shutil.copyfile(image_path, image_save_path)
        src_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(image_save_path, src_image)

        alpha = src_image[:, :, 3]
        alpha_save_path = os.path.join(dst_alpha_data_save_dir_train, os.path.basename(image_path).split('.')[0] + '.jpg')
        cv2.imwrite(alpha_save_path, alpha)
    
    # for image_path in train_src_alpha_paths:
    #     image_save_path = os.path.join(dst_alpha_data_save_dir_train, os.path.basename(image_path).split('.')[0] + '.jpg')
    #     # shutil.copyfile(image_path, image_save_path)
    #     src_image = cv2.imread(image_path)
    #     cv2.imwrite(image_save_path, src_image)

    ## alpha data
    for image_path in val_src_image_paths:
        image_save_path = os.path.join(dst_image_data_save_dir_val, os.path.basename(image_path).split('.')[0] + '.jpg')
        # shutil.copyfile(image_path, image_save_path)
        src_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(image_save_path, src_image)

        alpha = src_image[:, :, 3]
        alpha_save_path = os.path.join(dst_alpha_data_save_dir_val, os.path.basename(image_path).split('.')[0] + '.jpg')
        cv2.imwrite(alpha_save_path, alpha)
    
    # for image_path in val_src_alpha_paths:
    #     image_save_path = os.path.join(dst_alpha_data_save_dir_val, os.path.basename(image_path).split('.')[0] + '.jpg')
    #     # shutil.copyfile(image_path, image_save_path)
    #     src_image = cv2.imread(image_path)
    #     cv2.imwrite(image_save_path, src_image)


def bg_image_random_split():
    src_data_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_bg_dataset/augmented_bg_noise"

    dst_save_dir = os.path.join(os.path.dirname(src_data_dir), '0_use_dataset', 'bg_images')
    os.makedirs(dst_save_dir, exist_ok=True)

    train_data_save_dir = os.path.join(dst_save_dir, 'train')
    os.makedirs(train_data_save_dir, exist_ok=True)

    valid_data_save_dir = os.path.join(dst_save_dir, 'valid')
    os.makedirs(valid_data_save_dir, exist_ok=True)


    all_src_image_paths = glob(os.path.join(src_data_dir, '*.png'))

    random.shuffle(all_src_image_paths)

    total_image_num = len(all_src_image_paths)

    train_src_image_paths = all_src_image_paths[:int(total_image_num*0.9)]
    val_src_image_paths = all_src_image_paths[int(total_image_num*0.9):]

    for image_path in train_src_image_paths:
        image_save_path = os.path.join(train_data_save_dir, os.path.basename(image_path).split('.')[0] + '.jpg')
        # shutil.copyfile(image_path, image_save_path)
        src_image = cv2.imread(image_path)
        cv2.imwrite(image_save_path, src_image)

    for image_path in val_src_image_paths:
        image_save_path = os.path.join(valid_data_save_dir, os.path.basename(image_path).split('.')[0] + '.jpg')
        # shutil.copyfile(image_path, image_save_path)
        src_image = cv2.imread(image_path)
        cv2.imwrite(image_save_path, src_image)


def video_data_train_val_split(all_src_video_paths, dst_save_dir):
    train_data_save_dir = os.path.join(dst_save_dir, 'train')
    os.makedirs(train_data_save_dir, exist_ok=True)

    valid_data_save_dir = os.path.join(dst_save_dir, 'valid')
    os.makedirs(valid_data_save_dir, exist_ok=True)


    # all_src_video_paths = glob(os.path.join(src_data_dir, '*.png'))

    random.shuffle(all_src_video_paths)

    total_image_num = len(all_src_video_paths)

    train_src_image_paths = all_src_video_paths[:int(total_image_num*0.9)]
    val_src_image_paths = all_src_video_paths[int(total_image_num*0.9):]

    for video_path in train_src_image_paths:
        # image_save_path = os.path.join(train_data_save_dir, os.path.basename(image_path))
        # shutil.copyfile(image_path, image_save_path)

        dst_video_frame_save_folder = os.path.join(train_data_save_dir, os.path.basename(video_path))
        os.makedirs(dst_video_frame_save_folder, exist_ok=True)

        all_src_video_frames = glob(os.path.join(video_path, '*.png'))

        for video_frame_path in all_src_video_frames:
            current_frame = cv2.imread(video_frame_path)
            current_savename = os.path.join(dst_video_frame_save_folder, os.path.basename(video_frame_path).replace('.png', '.jpg'))
            cv2.imwrite(current_savename, current_frame)
        

    for video_path in val_src_image_paths:
        # image_save_path = os.path.join(valid_data_save_dir, os.path.basename(image_path))
        # shutil.copyfile(image_path, image_save_path)

        dst_video_frame_save_folder = os.path.join(valid_data_save_dir, os.path.basename(video_path))
        os.makedirs(dst_video_frame_save_folder, exist_ok=True)

        all_src_video_frames = glob(os.path.join(video_path, '*.png'))

        for video_frame_path in all_src_video_frames:
            current_frame = cv2.imread(video_frame_path)
            current_savename = os.path.join(dst_video_frame_save_folder, os.path.basename(video_frame_path).replace('.png', '.jpg'))
            cv2.imwrite(current_savename, current_frame)


if __name__ == '__main__':
    
    portrait_photo_split()


    # image_random_split()

    # src_data_dir = r"/projects/0_Xiaohua_codes_data/projects/1_Matting_solutions/0_Matting_dataset/0_bg_dataset/augmented_bg_noise_video"
    # all_src_video_folders = glob(os.path.join(src_data_dir, '*'))
    # dst_save_dir = os.path.join(os.path.dirname(src_data_dir), '0_use_dataset', 'bg_videos')
    # os.makedirs(dst_save_dir, exist_ok=True)
    # video_data_train_val_split(all_src_video_folders, dst_save_dir)
        

