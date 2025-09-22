import numpy as np
from glob import glob


# Load the file names for train, test and validation data into a list
def extract_filenm(path_list,save_filenm_path):
    filenm_coll = []
    for path in path_list:
        filenm_coll.append(f"{path.split('/')[-1].split('.')[0]} \n")

    with open(f'{save_filenm_path}', 'w') as f:
        f.writelines(filenm_coll)

    print(f'Annotations files are saved at {save_filenm_path}')




fold_names = ['fold1','fold2','fold3','fold4','final_config_data']


for j in range(len(fold_names)):
# Load the paths to train, test and validation data
    cropped_img_path_train = sorted(glob(f'../mmdetection/data/ewis/{fold_names[j]}/train_images/*.png', recursive=True)) 
    cropped_img_path_test = sorted(glob(f'../mmdetection/data/ewis/{fold_names[j]}/test_images/*.png', recursive=True))
    cropped_img_path_val = sorted(glob(f'../mmdetection/data/ewis/{fold_names[j]}/val_images/*.png', recursive=True))

    annotation_file_train = f'../mmdetection/data/ewis/{fold_names[j]}/train.txt'
    annotation_file_test = f'../mmdetection/data/ewis/{fold_names[j]}/test.txt'
    annotation_file_val = f'../mmdetection/data/ewis/{fold_names[j]}/val.txt'

    extract_filenm(cropped_img_path_train,annotation_file_train)
    extract_filenm(cropped_img_path_test,annotation_file_test)
    extract_filenm(cropped_img_path_val,annotation_file_val)