import cv2
import os
import albumentations as A
import numpy as np

# Define the augmentation pipeline
augmentation = A.Compose(
    [
         # Geometric transformations
        A.HorizontalFlip(p=0.5),  # Flip 50% of the images horizontally
        A.VerticalFlip(p=0.5),  # Flip 50% of the images vertically
        A.RandomRotate90(p=0.5),  # Rotate by 90 degrees randomly
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),  # Random shifting, scaling, rotation
        A.Transpose(p=0.5),  # Random transpose of the image
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),  # Optical distortion
        #A.GridDistortion(p=0.5),  # Apply grid distortion
        A.ElasticTransform(p=0.5),  # Apply elastic deformation

        # Random crops and resizes
        #A.RandomCrop(width=256, height=256, p=0.5),  # Random crop
        #A.CenterCrop(width=256, height=256, p=0.5),  # Center crop
        #A.Resize(height=512, width=512, p=1.0),  # Resize the image to a fixed size
        
        # Brightness, contrast, and color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjust brightness and contrast
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Adjust hue, saturation, and value
        #A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),  # Shift RGB channels
       # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),  # Apply color jitter

        # Blur and noise
        A.MotionBlur(blur_limit=7, p=0.5),  # Apply motion blur
        A.MedianBlur(blur_limit=7, p=0.5),  # Apply median blur
        A.GaussianBlur(blur_limit=7, p=0.5),  # Apply Gaussian blur
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),  # Add Gaussian noise
        #A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),  # Add ISO noise
        
        # Other augmentations
        #A.CLAHE(clip_limit=4.0, p=0.5),  # Contrast Limited Adaptive Histogram Equalization
        #A.ChannelShuffle(p=0.5),  # Randomly shuffle the RGB channels
        #A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),  # Randomly cut out squares
        #A.Perspective(scale=(0.05, 0.1), p=0.5),  # Apply perspective transform
        #A.Solarize(p=0.5),  # Solarize the image (invert pixels above a threshold)
        #A.Equalize(p=0.5),  # Equalize the image histogram
        #A.Posterize(p=0.5),  # Reduce the number of bits for each color channel
        #A.InvertImg(p=0.5),  # Invert the image colors
        #A.ToGray(p=0.5),  # Convert the image to grayscale
        #A.ToSepia(p=0.5),  # Convert the image to sepia tone
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

# Path to images and corresponding .txt files
image_dir = "images/"
label_dir = "labels/"
aug_image_dir = "augmented_images/"
aug_label_dir = "augmented_labels/"
aug_visualization_dir = "augmented_visualization/"

if not os.path.exists(aug_image_dir):
    os.makedirs(aug_image_dir)
if not os.path.exists(aug_label_dir):
    os.makedirs(aug_label_dir)
if not os.path.exists(aug_visualization_dir):
    os.makedirs(aug_visualization_dir)


# Function to load YOLO labels from .txt file
def load_yolo_labels(txt_path):
    boxes = []
    class_labels = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            values = line.strip().split()
            if len(values) == 5:  # Make sure there are 5 values in the line
                class_label, x_center, y_center, width, height = map(float, values)
                boxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_label))  # Collect class labels
            else:
                print(f"Warning: Skipping invalid label format in {txt_path}: {values}")
    return boxes, class_labels


# Function to save augmented YOLO labels to .txt file
def save_yolo_labels(txt_path, boxes, class_labels):
    with open(txt_path, "w") as f:
        for box, class_label in zip(boxes, class_labels):
            x_center, y_center, width, height = box
            f.write(f"{class_label} {x_center} {y_center} {width} {height}\n")


# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, boxes, class_labels):
    h, w, _ = image.shape
    for box, class_label in zip(boxes, class_labels):
        x_center, y_center, width, height = box
        # Convert YOLO format (normalized) to pixel values
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)
        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Optionally, put the class label as text on the bounding box
        cv2.putText(
            image,
            str(class_label),
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )


# Loop through all images and augment them
for image_file in os.listdir(image_dir):
    if image_file.endswith(".jpg"):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))

        # Load image and corresponding YOLO labels
        image = cv2.imread(image_path)
        bboxes, class_labels = load_yolo_labels(label_path)

        # Print loaded bounding boxes for debugging
        print(f"Loaded boxes for {image_file}: {bboxes}")
        for bbox in bboxes:
            assert len(bbox) == 4, f"Error: Bounding box {bbox} does not have 4 values"

        # Generate exactly 5 augmented images
        for i in range(5):  # Set to generate 5 augmented copies per image
            augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_class_labels = augmented["class_labels"]  # Extract augmented class labels

            # Ensure augmented bounding boxes have the correct format
            for aug_bbox in aug_bboxes:
                assert len(aug_bbox) == 4, f"Error: Augmented box {aug_bbox} does not have 4 values"

            # Save the augmented image
            aug_image_name = f"aug_{i}_{image_file}"
            aug_image_path = os.path.join(aug_image_dir, aug_image_name)
            cv2.imwrite(aug_image_path, aug_image)

            # Save the augmented labels
            aug_label_path = os.path.join(
                aug_label_dir, aug_image_name.replace(".jpg", ".txt")
            )
            save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)

            # Draw bounding boxes on the augmented image
            aug_image_with_boxes = aug_image.copy()
            draw_bounding_boxes(aug_image_with_boxes, aug_bboxes, aug_class_labels)

            # Save the visualization image with bounding boxes
            aug_vis_path = os.path.join(aug_visualization_dir, aug_image_name)
            cv2.imwrite(aug_vis_path, aug_image_with_boxes)
