import os

from PIL import Image
from matplotlib import pyplot as plt

img_path = r"H:\Datasets\DogBreed"
image_files = list()

for root, dirs, files in os.walk(img_path):
    for file in files:
        if file.endswith(".jpg"):
            image_files.append(os.path.join(root, file))

image_count = len(image_files)

print("Found " + str(image_count) + " images.")
print("Calculating average...")
total_height = 0
total_width = 0
scatter_widths = []
scatter_heights = []

for image_file in image_files:
    image = Image.open(image_file)
    width, height = image.size

    # Get rid of outliers
    if width >= 1000 or height >= 1000:
        image_count -= 1
        continue

    total_width += width
    total_height += height
    scatter_heights.append(height)
    scatter_widths.append(width)

print("Average width: " + str(total_width // image_count))
print("Average height: " + str(total_height // image_count))
plt.scatter(scatter_widths, scatter_heights)
plt.show()
