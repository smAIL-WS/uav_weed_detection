import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


def sample_half_dataset_by_subfolder(image_root, anno_root, output_root, seed=50):
    random.seed(seed)
    
    # Get all subfolders inside the train folders
    image_subfolders = sorted([f for f in Path(image_root).iterdir() if f.is_dir()])
    anno_subfolders = sorted([f for f in Path(anno_root).iterdir() if f.is_dir()])

    # Sanity check
    assert len(image_subfolders) == len(anno_subfolders), "Mismatch in subfolder counts"

    all_image_paths = []
    folder_to_images = {}

    # Collect all image paths per subfolder
    for subfolder in image_subfolders:
        images = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpeg"))
        folder_to_images[subfolder.name] = images
        all_image_paths.extend(images)

    total_images = len(all_image_paths)
    target_sample_size = max(len(image_subfolders), total_images // 2)

    # Step 1: Ensure at least one image per folder
    selected_images = []
    selected_image_names = set()
    for folder, images in folder_to_images.items():
        if images:
            img = random.choice(images)
            selected_images.append(img)
            selected_image_names.add(img.name)

    # Step 2: Sample remaining needed images randomly from the rest
    remaining = target_sample_size - len(selected_images)
    remaining_pool = [img for img in all_image_paths if img.name not in selected_image_names]

    if remaining > 0:
        additional_samples = random.sample(remaining_pool, min(remaining, len(remaining_pool)))
        selected_images.extend(additional_samples)

    # Copy images and annotations to respective subfolders
    for img_path in selected_images:
        subfolder = img_path.parent.name
        anno_filename = img_path.with_suffix(".xml").name
        anno_path = Path(anno_root) / subfolder / anno_filename

        dest_img_folder = Path(output_root) / "images" / "train" / subfolder
        dest_anno_folder = Path(output_root) / "annotations" / "train" / subfolder
        dest_img_folder.mkdir(parents=True, exist_ok=True)
        dest_anno_folder.mkdir(parents=True, exist_ok=True)

        dest_img = dest_img_folder / img_path.name
        dest_anno = dest_anno_folder / anno_filename

        shutil.copy(img_path, dest_img)
        if anno_path.exists():
            shutil.copy(anno_path, dest_anno)
        else:
            print(f"[Warning] Annotation file not found for: {img_path.name}")

    print(f"Sampled {len(selected_images)} images across folders into: {output_root}")


sample_half_dataset_by_subfolder(
    image_root="EWIS_Dataset/new_full/images/train/",
    anno_root="EWIS_Dataset/new_full/annotations/train/",
    output_root="EWIS_Dataset/new_half_v1/"
)


def sample_quarter_dataset_by_subfolder(image_root, anno_root, output_root, seed=50):
    random.seed(seed)
    
    # Get all subfolders inside the train folders
    image_subfolders = sorted([f for f in Path(image_root).iterdir() if f.is_dir()])
    anno_subfolders = sorted([f for f in Path(anno_root).iterdir() if f.is_dir()])

    # Sanity check
    assert len(image_subfolders) == len(anno_subfolders), "Mismatch in subfolder counts"

    all_image_paths = []
    folder_to_images = {}

    # Collect all image paths per subfolder
    for subfolder in image_subfolders:
        images = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpeg"))
        folder_to_images[subfolder.name] = images
        all_image_paths.extend(images)

    total_images = len(all_image_paths)
    target_sample_size = max(len(image_subfolders), total_images // 4)

    # Step 1: Ensure at least one image per folder
    selected_images = []
    selected_image_names = set()
    for folder, images in folder_to_images.items():
        if images:
            img = random.choice(images)
            selected_images.append(img)
            selected_image_names.add(img.name)

    # Step 2: Sample remaining needed images randomly from the rest
    remaining = target_sample_size - len(selected_images)
    remaining_pool = [img for img in all_image_paths if img.name not in selected_image_names]

    if remaining > 0:
        additional_samples = random.sample(remaining_pool, min(remaining, len(remaining_pool)))
        selected_images.extend(additional_samples)

    # Copy images and annotations to respective subfolders
    for img_path in selected_images:
        subfolder = img_path.parent.name
        anno_filename = img_path.with_suffix(".xml").name
        anno_path = Path(anno_root) / subfolder / anno_filename

        dest_img_folder = Path(output_root) / "images" / "train" / subfolder
        dest_anno_folder = Path(output_root) / "annotations" / "train" / subfolder
        dest_img_folder.mkdir(parents=True, exist_ok=True)
        dest_anno_folder.mkdir(parents=True, exist_ok=True)

        dest_img = dest_img_folder / img_path.name
        dest_anno = dest_anno_folder / anno_filename

        shutil.copy(img_path, dest_img)
        if anno_path.exists():
            shutil.copy(anno_path, dest_anno)
        else:
            print(f"[Warning] Annotation file not found for: {img_path.name}")

    print(f"Sampled {len(selected_images)} images across folders into: {output_root}")


sample_quarter_dataset_by_subfolder(
    image_root="EWIS_Dataset/new_full/images/train/",
    anno_root="EWIS_Dataset/new_full/annotations/train/",
    output_root="EWIS_Dataset/new_quarter_v1/"
)




def extract_stripe_from_filename(filename):
    """Extract stripe label (A, B, C, D) from filename."""
    for stripe in ['A', 'B', 'C', 'D']:
        if f"_{stripe}" in filename or f"_{stripe.lower()}" in filename:
            return stripe
    return None  # No stripe info found

def sample_one_image_per_growth_stage(image_root, anno_root, output_root, seed=0):
    random.seed(seed)
    image_root = Path(image_root)
    anno_root = Path(anno_root)
    output_root = Path(output_root)

    image_subfolders = sorted([f for f in image_root.iterdir() if f.is_dir()])

    selected_images = []
    stripe_counts = defaultdict(int)

    for subfolder in image_subfolders:
        images = list(subfolder.glob("*.jpg")) + list(subfolder.glob("*.png")) + list(subfolder.glob("*.jpeg"))

        # Shuffle to randomize selection
        random.shuffle(images)

        found_valid = False
        for img_path in images:
            stripe = extract_stripe_from_filename(img_path.name)
            if stripe and stripe_counts[stripe] < 2:
                stripe_counts[stripe] += 1
                selected_images.append(img_path)
                found_valid = True
                break  # Only one image per growth stage
        if not found_valid:
            print(f"[Warning] No suitable image found for subfolder: {subfolder.name}")

    # Copy selected images and corresponding annotations
    for img_path in selected_images:
        subfolder = img_path.parent.name
        anno_filename = img_path.with_suffix(".xml").name
        anno_path = anno_root / subfolder / anno_filename

        dest_img_folder = output_root / "images" / "train" / subfolder
        dest_anno_folder = output_root / "annotations" / "train" / subfolder
        dest_img_folder.mkdir(parents=True, exist_ok=True)
        dest_anno_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, dest_img_folder / img_path.name)
        if anno_path.exists():
            shutil.copy(anno_path, dest_anno_folder / anno_filename)
        else:
            print(f"[Warning] Missing annotation for: {img_path.name}")

    print(f"✅ Sampled {len(selected_images)} images. Stripe counts: {dict(stripe_counts)}")

sample_one_image_per_growth_stage(
    image_root="EWIS_Dataset/new_full/images/train/",
    anno_root="EWIS_Dataset/new_full/annotations/train/",
    output_root="EWIS_Dataset/new_single_v1/"
)



