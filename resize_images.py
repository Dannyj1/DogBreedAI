import os

import PIL
from PIL import Image

img_path = r"H:\Datasets\DogBreed"
image_files = list()

for root, dirs, files in os.walk(img_path):
    for file in files:
        if file.endswith(".jpg"):
            image_files.append(os.path.join(root, file))

image_count = len(image_files)

print("Found " + str(image_count) + " images.")
print("Resizing images...")
total_height = 0
total_width = 0
scatter_widths = []
scatter_heights = []

for image_file in image_files:
    image = Image.open(image_file)
    width, height = image.size
    replace_path = image_file.replace(".jpg", "-resized.jpg")

    if width > 600 or height > 600:
        print("Resizing " + image_file + " from " + str(image.size) + "...")
        image.thumbnail((600, 600), PIL.Image.BILINEAR)
        image.save(replace_path)
        image.close()
        os.remove(image_file)
        os.replace(src=replace_path, dst=image_file)
