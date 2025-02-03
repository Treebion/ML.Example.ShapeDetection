import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw

# Configurations
IMAGE_SIZE = 256
NUM_TRAIN_IMAGES = 200
NUM_TEST_IMAGES = 20
TRAIN_FOLDER = os.path.abspath("output-1/train_images")  # Training images
TEST_FOLDER = os.path.abspath("output-1/test_images")  # Testing images
COCO_JSON_FILE = os.path.abspath("output-1/coco_annotations.json")  # COCO JSON for training images

SHAPES = ["circle", "square", "star", "pentagon", "hexagon"]
SHAPE_LABELS = {shape: i + 1 for i, shape in enumerate(SHAPES)}  # Assign numerical IDs

# Ensure output directories exist
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# COCO JSON structure for training data
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": v, "name": k} for k, v in SHAPE_LABELS.items()]
}

annotation_id = 1  # Unique ID for annotations


def get_random_color():
    """Returns a random RGB color that is not too dark."""
    return tuple(random.randint(50, 255) for _ in range(3))


def draw_rotated_shape(draw, x, y, size, sides, color, angle):
    """Draws a rotated polygon (pentagon, hexagon, star, etc.)."""
    points = []
    step = 2 * np.pi / sides  # Angle step for each side

    for i in range(sides):
        points.append((
            x + size / 2 + np.cos(angle + step * i) * size / 2,
            y + size / 2 + np.sin(angle + step * i) * size / 2
        ))

    draw.polygon(points, fill=color)


def draw_star(draw, x, y, size, color, angle):
    """Draws a five-pointed star with rotation."""
    points = []
    step = np.pi / 5  # Angle step for star points

    for i in range(10):
        radius = size / 2 if i % 2 == 0 else size / 4  # Outer/inner radius
        points.append((
            x + size / 2 + np.cos(angle + step * i) * radius,
            y + size / 2 + np.sin(angle + step * i) * radius
        ))

    draw.polygon(points, fill=color)


def generate_image(image_id, file_name, folder, coco_annotations=True, forced_shape=None):
    """Generates an image with a forced shape type to ensure filename matches."""
    global annotation_id

    # Create a random background color
    background_color = get_random_color()
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), background_color)
    draw = ImageDraw.Draw(img)

    # Use the forced shape if provided, otherwise pick randomly
    shape_type = forced_shape if forced_shape else random.choice(SHAPES)
    category_id = SHAPE_LABELS[shape_type]

    # Generate random size and position
    shape_size = random.randint(int(IMAGE_SIZE * 0.3), int(IMAGE_SIZE * 0.5))
    pos_x = random.randint(0, IMAGE_SIZE - shape_size)
    pos_y = random.randint(0, IMAGE_SIZE - shape_size)

    # Ensure shape color contrasts with background
    shape_color = get_random_color()
    while shape_color == background_color:
        shape_color = get_random_color()

    # Generate a random rotation angle
    rotation_angle = random.uniform(0, 2 * np.pi)

    # Draw the correct shape with rotation
    if shape_type == "circle":
        draw.ellipse([pos_x, pos_y, pos_x + shape_size, pos_y + shape_size], fill=shape_color)
    elif shape_type == "square":
        draw_rotated_shape(draw, pos_x, pos_y, shape_size, 4, shape_color, rotation_angle)
    elif shape_type == "star":
        draw_star(draw, pos_x, pos_y, shape_size, shape_color, rotation_angle)
    elif shape_type == "pentagon":
        draw_rotated_shape(draw, pos_x, pos_y, shape_size, 5, shape_color, rotation_angle)
    elif shape_type == "hexagon":
        draw_rotated_shape(draw, pos_x, pos_y, shape_size, 6, shape_color, rotation_angle)

    # Save image with full path
    img_path = os.path.join(folder, file_name)
    img.save(img_path)

    # Add to COCO annotations if required (for training set only)
    if coco_annotations:
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_path,  # Full absolute path
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE
        })

        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [pos_x, pos_y, shape_size, shape_size],  # Correct COCO format
            "area": shape_size * shape_size,
            "iscrowd": 0
        })
        annotation_id += 1


# Generate 2,000 training images
for i in range(NUM_TRAIN_IMAGES):
    file_name = f"img_{i:05d}.png"
    generate_image(i + 1, file_name, TRAIN_FOLDER, coco_annotations=True)
    if i % 1000 == 0:
        print(f"✅ {i} training images created...")

# Generate 200 test images with shape name in filename
for i in range(NUM_TEST_IMAGES):
    shape_type = random.choice(SHAPES)  # Choose shape **once**
    file_name = f"{shape_type}_{i:05d}.png"  # Ensure filename matches chosen shape
    generate_image(i + 1, file_name, TEST_FOLDER, coco_annotations=False, forced_shape=shape_type)  # Pass shape
    if i % 100 == 0:
        print(f"✅ {i} test images created...")

# Save COCO JSON (only for training set)
with open(COCO_JSON_FILE, "w") as json_file:
    json.dump(coco_data, json_file, indent=4)

print(f"✅ COCO JSON saved: {COCO_JSON_FILE}")
print(f"✅ All images generated successfully!")
