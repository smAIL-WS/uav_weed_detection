import os
import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
from skimage.io import imsave
import matplotlib.pyplot as plt
from glob import glob

def img_to_list(img_path):
    img = cv2.imread(img_path)
    return img

def parse_xml_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    for obj in root.findall('object'):
        category = obj.find('name').text
        category_id = 1 if category == "crop" else 2  # Modify as per category mapping
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        
        annotations.append({
            "category_id": category_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min)
        })
    
    return annotations

def plot_sample_crops(json_file, image_folder, num_samples=5):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    images = coco_data["images"][:num_samples]
    annotations = coco_data["annotations"]
    categories = {category["id"]: category["name"] for category in coco_data["categories"]}
    
    for img_data in images:
        img_path = os.path.join(image_folder, img_data["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, figsize=(6, 6))
        ax.imshow(img)
        
        img_id = img_data["id"]
        for ann in annotations:
            if ann["image_id"] == img_id:
                x, y, w, h = ann["bbox"]
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                # Get the class name from category_id
                class_name = categories.get(ann["category_id"], "Unknown")
                
                # Print the class name next to the bounding box
                ax.text(x, y - 5, class_name, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
        
        plt.title(img_data["file_name"])
        plt.show()


def crop_img(img, dst_folder, filename, xml_path, crop_size, coco_data, annotation_id, image_id):
    h, w, d = img.shape
    fnm_split = filename.split('/')
    orig_name = fnm_split[-1].split('.png')[0]
    
    orig_annotations = parse_xml_annotations(xml_path)
    
    # We loop using the range, but we will handle the "over-hang" inside
    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):
            # 1. Extract the actual available pixels
            # This might be smaller than crop_size at the right/bottom edges
            actual_crop = img[i:min(i + crop_size, h), j:min(j + crop_size, w), :]
            curr_h, curr_w, _ = actual_crop.shape
            
            # 2. Initialize a black canvas of the full crop_size
            # This handles the zero-padding automatically
            img_padded = np.zeros((crop_size, crop_size, d), dtype=img.dtype)
            
            # 3. Paste the actual image onto the top-left of the canvas
            img_padded[0:curr_h, 0:curr_w, :] = actual_crop
            
            # --- Processing & Saving ---
            savename = orig_name + f'_patch_{i//crop_size}_{j//crop_size}.png'
            savepath = os.path.join(dst_folder, savename)
            
            # Normalize and save
            # Note: Using a robust min/max check to avoid division by zero in pure black patches
            denom = (np.max(img_padded) - np.min(img_padded))
            if denom > 0:
                img_final = (255 * (img_padded - np.min(img_padded)) / denom).astype(np.uint8)
            else:
                img_final = img_padded.astype(np.uint8)
                
            cv2.imwrite(savepath, img_final)
            
            # Add image info (Width/Height are now guaranteed to be crop_size)
            coco_data["images"].append({
                "id": image_id,
                "file_name": savename,
                "width": crop_size,
                "height": crop_size
            })
            
            # --- Annotation Handling ---
            for ann in orig_annotations:
                x_min, y_min, box_w, box_h = ann["bbox"]
                x_max, y_max = x_min + box_w, y_min + box_h
                
                # Check intersection using original global coordinates
                if (x_min < (j + crop_size) and x_max > j and
                    y_min < (i + crop_size) and y_max > i):
                    
                    # Shift coordinates relative to the crop start (i, j)
                    new_x_min = max(0, x_min - j)
                    new_y_min = max(0, y_min - i)
                    new_x_max = min(crop_size, x_max - j)
                    new_y_max = min(crop_size, y_max - i)

                    # CLIP coordinates to the actual image pixels available 
                    # If a box is in the padded (black) area, this makes it valid
                    new_x_max = min(new_x_max, curr_w)
                    new_y_max = min(new_y_max, curr_h)

                    new_width = new_x_max - new_x_min
                    new_height = new_y_max - new_y_min

                    # Only add if the resulting box still has area 
                    # (i.e., it wasn't entirely in the padded region)
                    if new_width > 0 and new_height > 0:
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": ann["category_id"],
                            "bbox": [new_x_min, new_y_min, new_width, new_height],
                            "area": float(new_width * new_height),
                            "iscrowd": 0
                        })
                        annotation_id += 1
            
            image_id += 1
    
    return annotation_id, image_id


def process_images(image_paths, xml_paths, dst_folder, crop_size, output_json):
    coco_data = {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "crop"},
        {"id": 2, "name": "weed"}
    ]}
    
    annotation_id = 1
    image_id = 1
    
    for img_path, xml_path in zip(image_paths, xml_paths):
        img = img_to_list(img_path)
        annotation_id, image_id = crop_img(img, dst_folder, img_path, xml_path, crop_size, coco_data, annotation_id, image_id)
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=4)




def train_val_test_paths(img_paths, xml_paths, fold_name):
    train_img_paths = []
    train_xml_paths = []
    val_img_paths = []
    val_xml_paths = []
    test_img_paths = []
    test_xml_paths = []
    for i in range(len(img_paths)):
        if fold_name == 'fold1':
            if img_paths[i].split('/')[-1].split('_')[2][0] == 'A' or img_paths[i].split('/')[-1].split('_')[2][0] == 'B' or img_paths[i].split('/')[-1].split('_')[2][0] == 'C':
                train_img_paths.append(img_paths[i])
                train_xml_paths.append(xml_paths[i])
            else:
                val_img_paths.append(img_paths[i])
                val_xml_paths.append(xml_paths[i])
                test_img_paths.append(img_paths[i])
                test_xml_paths.append(xml_paths[i])
        elif fold_name == 'fold2':
            if img_paths[i].split('/')[-1].split('_')[2][0] == 'A' or img_paths[i].split('/')[-1].split('_')[2][0] == 'B' or img_paths[i].split('/')[-1].split('_')[2][0] == 'D':
                train_img_paths.append(img_paths[i])
                train_xml_paths.append(xml_paths[i])
            else:
                val_img_paths.append(img_paths[i])
                val_xml_paths.append(xml_paths[i])
                test_img_paths.append(img_paths[i])
                test_xml_paths.append(xml_paths[i])
        elif fold_name == 'fold3':
            if img_paths[i].split('/')[-1].split('_')[2][0] == 'A' or img_paths[i].split('/')[-1].split('_')[2][0] == 'C' or img_paths[i].split('/')[-1].split('_')[2][0] == 'D':
                train_img_paths.append(img_paths[i])
                train_xml_paths.append(xml_paths[i])
            else:
                val_img_paths.append(img_paths[i])
                val_xml_paths.append(xml_paths[i])
                test_img_paths.append(img_paths[i])
                test_xml_paths.append(xml_paths[i])
        elif fold_name == 'fold4':
            if img_paths[i].split('/')[-1].split('_')[2][0] == 'B' or img_paths[i].split('/')[-1].split('_')[2][0] == 'C' or img_paths[i].split('/')[-1].split('_')[2][0] == 'D':
                train_img_paths.append(img_paths[i])
                train_xml_paths.append(xml_paths[i])
            else:
                val_img_paths.append(img_paths[i])
                val_xml_paths.append(xml_paths[i])
                test_img_paths.append(img_paths[i])
                test_xml_paths.append(xml_paths[i])

        elif fold_name == 'final_config_data':
            val_img_paths.append(img_paths[i])
            val_xml_paths.append(xml_paths[i])
            test_img_paths.append(img_paths[i])
            test_xml_paths.append(xml_paths[i])

    
    return train_img_paths, train_xml_paths, val_img_paths, val_xml_paths, test_img_paths, test_xml_paths

        
fold_names = ["fold1","fold2","fold3","fold4","final_config_data"]
split_names = ["train","val","test"]
crop_size = 512

for i in range(len(fold_names)):

    if fold_names[i] != 'final_config_data':
        all_img_path = sorted(glob("<specify_paths_to_images>/images/train/**/*.png",recursive=True))
        all_xml_path = sorted(glob("<specify_paths_to_annotations>/annotations/train/**/*.xml",recursive=True))
        train_img_paths, train_xml_paths, val_img_paths, val_xml_paths, test_img_paths, test_xml_paths = train_val_test_paths(all_img_path, all_xml_path, fold_names[i])

        #Loading train_images and annotations
        train_output_json = f"../mmdetection/data/ewis/{fold_names[i]}/train.json"
        train_img_dst_folder = f"../mmdetection/data/ewis/{fold_names[i]}/train_images/"
        if not os.path.exists(train_img_dst_folder):
            os.makedirs(train_img_dst_folder) 

        process_images(train_img_paths, train_xml_paths, train_img_dst_folder, crop_size, train_output_json)

        #Loading val_images and annotations
        val_output_json = f"../mmdetection/data/ewis/{fold_names[i]}/val.json"
        val_img_dst_folder = f"../mmdetection/data/ewis/{fold_names[i]}/val_images/"
        if not os.path.exists(val_img_dst_folder):
            os.makedirs(val_img_dst_folder) 

        process_images(val_img_paths, val_xml_paths, val_img_dst_folder, crop_size, val_output_json)

        #Loading test_images and annotations
        test_output_json = f"../mmdetection/data/ewis/{fold_names[i]}/test.json"
        test_img_dst_folder = f"../mmdetection/data/ewis/{fold_names[i]}/test_images/"
        if not os.path.exists(test_img_dst_folder):
            os.makedirs(test_img_dst_folder) 

        process_images(test_img_paths, test_xml_paths, test_img_dst_folder, crop_size, test_output_json)
    
    else:
        all_img_path = sorted(glob("<specify_paths_to_images>/images/train/**/*.png",recursive=True))
        all_xml_path = sorted(glob("<specify_paths_to_annotations>/annotations/train/**/*.xml",recursive=True))
        

        #Loading train_images and annotations
        train_output_json = f"../mmdetection/data/ewis/{fold_names[i]}/train.json"
        train_img_dst_folder = f"../mmdetection/data/ewis/{fold_names[i]}/train_images/"
        if not os.path.exists(train_img_dst_folder):
            os.makedirs(train_img_dst_folder) 

        process_images(all_img_path, all_xml_path, train_img_dst_folder, crop_size, train_output_json)


        all_img_path = sorted(glob("<specify_paths_to_images>/images/test/**/*.png",recursive=True))
        all_xml_path = sorted(glob("<specify_paths_to_annotations>/annotations/test/**/*.xml",recursive=True))
        _, _, val_img_paths, val_xml_paths, test_img_paths, test_xml_paths = train_val_test_paths(all_img_path, all_xml_path, fold_names[i])

        #Loading val_images and annotations
        val_output_json = f"../mmdetection/data/ewis/{fold_names[i]}/val.json"
        val_img_dst_folder = f"../mmdetection/data/ewis/{fold_names[i]}/val_images/"
        if not os.path.exists(val_img_dst_folder):
            os.makedirs(val_img_dst_folder) 

        process_images(val_img_paths, val_xml_paths, val_img_dst_folder, crop_size, val_output_json)

        #Loading test_images and annotations
        test_output_json = f"../mmdetection/data/ewis/{fold_names[i]}/test.json"
        test_img_dst_folder = f"../mmdetection/data/ewis/{fold_names[i]}/test_images/"
        if not os.path.exists(test_img_dst_folder):
            os.makedirs(test_img_dst_folder) 

        process_images(test_img_paths, test_xml_paths, test_img_dst_folder, crop_size, test_output_json)