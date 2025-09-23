import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET

def load_groundtruth_from_xml(xml_file, class_map={"crop": 0, "weed": 1}):
    """
    Load ground truth annotations from Pascal VOC-style XML file.
    Returns: gt_bboxes, gt_labels
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes, labels = [], []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in class_map:
            continue  # skip unknown classes

        cls_id = class_map[cls_name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls_id)

    return [bboxes], [labels]  # wrapped in list → per-image format




def plot_predictions(
    image_path,
    pred_bboxes, pred_labels, pred_scores,
    gt_bboxes, gt_labels,
    score_thresh=0.5,
    class_map={0: "crop", 1: "weed"},
    save_dir="./visualizations"
):
    """
    Plot ground truth and predictions side by side with a shared legend.
    Saves the figure in high resolution.
    """

    img = Image.open(image_path)

    # Create two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))  # large figsize for drone images

    # --- Ground Truth Plot ---
    axes[0].imshow(img)
    axes[0].set_title("Ground Truth", fontsize=20)
    for bbox, label in zip(gt_bboxes, gt_labels):
        xmin, ymin, xmax, ymax = bbox
        width, height = xmax - xmin, ymax - ymin
        cls_name = class_map[label]
        color = "blue" if cls_name == "crop" else "red"

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(
            xmin, ymin - 5, f"{cls_name}",
            fontsize=12, color=color, weight="bold"
        )
    axes[0].axis("off")

    # --- Prediction Plot ---
    axes[1].imshow(img)
    axes[1].set_title("Predictions", fontsize=20)
    for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
        if score <= score_thresh:
            continue

        xmin, ymin, xmax, ymax = bbox
        width, height = xmax - xmin, ymax - ymin
        cls_name = class_map[label]
        color = "blue" if cls_name == "crop" else "red"

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(
            xmin, ymin - 5, f"{cls_name} ({score:.2f})",
            fontsize=12, color=color, weight="bold"
        )
    axes[1].axis("off")

    # --- Shared Legend ---
    handles = [
        patches.Patch(color="blue", label="Crop"),
        patches.Patch(color="red", label="Weed"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=16,
        frameon=False
    )

    # --- Save the figure ---
    save_path = f"{save_dir}/gt_pred_{image_path.split('/')[-1]}"
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend at bottom
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, format="png", dpi=600)  # high resolution
    plt.close(fig)


# ==== Example usage ====
pred_bboxes = torch.load('./predictions/full_pred_boxes_gd_best_config.pt')
pred_scores = torch.load('./predictions/full_pred_scores_gd_best_config.pt')
pred_labels = torch.load('./predictions/full_pred_labels_gd_best_config.pt')
gt_file   = "./groundtruths/annotations/test_image.xml"

# Load files
gt_bboxes, gt_labels = load_groundtruth_from_xml(gt_file)

# Example
test_image_path = "./groundtruths/images/test_image.png"

# Plot only first image’s predictions
plot_predictions(
    test_image_path,
    pred_bboxes,  
    pred_labels,  
    pred_scores,  
    gt_bboxes, 
    gt_labels,
    score_thresh=0.5   
)