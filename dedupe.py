import math
import os

import imagehash
from PIL import Image


breed = "Yorkshire Terrier"


def hash_all_images(image_paths):
    result = dict()

    for path in image_paths:
        result[path] = imagehash.average_hash(Image.open(path))

    return result


train_path = "H:\\Datasets\\DogBreed\\train\\" + breed
# train_path = "H:\\Datasets\\DogBreed"
val_path = "H:\\Datasets\\DogBreed\\val\\" + breed
test_path = "H:\\Datasets\\DogBreed\\test\\" + breed
image_files = list()

for root, dirs, files in os.walk(train_path):
    for file in files:
        image_files.append(os.path.join(root, file))

for root, dirs, files in os.walk(val_path):
    for file in files:
        image_files.append(os.path.join(root, file))

for root, dirs, files in os.walk(test_path):
    for file in files:
        image_files.append(os.path.join(root, file))

print("Found " + str(len(image_files)) + " images.")
print("Creating hashes...")
hashes = hash_all_images(image_files)

print("Finding duplicates...")
file_num = 0
removed_files = 0
sim_total = 0

for item in hashes.items():
    try:
        path1 = item[0]
        hash1 = item[1]
        hits = 0
        file_num += 1
        percentage = math.floor(file_num / len(hashes) * 100)

        print("(" + str(percentage) + "%) Analyzing file " + str(file_num) + " of " + str(len(hashes)) + "...")

        if hash1 is None:
            continue

        total = 0
        for other_item in hashes.items():
            path2 = other_item[0]
            hash2 = other_item[1]

            if path2 == path1:
                continue

            if hash2 is None:
                continue

            if hash2 == hash1 or hash1 - hash2 >= 65:
                print("Found duplicate file: " + path2 + " is simlar to " + path1)
                image1 = Image.open(path1)
                image2 = Image.open(path2)
                width1, height1 = image1.size
                width2, height2 = image2.size

                image1.close()
                image2.close()

                if height1 >= height2:
                    os.remove(path2)
                    hashes[path2] = None
                else:
                    os.remove(path1)
                    hashes[path1] = None

                removed_files += 1
                break
    except Exception as ex:
        print(ex)
        continue

print("Removed " + str(removed_files) + " duplicate files!")
