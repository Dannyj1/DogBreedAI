import os

import imagehash
from PIL import Image

phash_result = dict()
removed_files = 0


def hash_image(path):
    global phash_result
    global removed_files

    image = None
    try:
        # print("Creating a hash of " + str(path) + "...")
        image = Image.open(path)
        phash = imagehash.phash(image)
        image.close()
        image = None

        if str(phash) in phash_result:
            print("(PHash) Found duplicate file: " + path + " is a duplicate image! Deleting...")
            os.remove(path)
            removed_files += 1
            return

        phash_result[str(phash)] = path
    except Exception as ex:
        print(path + " is an invalid image! Deleting... (" + str(ex) + ")")
        if image is not None:
            image.close()

        os.remove(path)


if __name__ == '__main__':
    print("Searching for files...")
    train_path = r"H:\Datasets\DogBreed\train"
    val_path = r"H:\Datasets\DogBreed\val"
    test_path = r"H:\Datasets\DogBreed\test"
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
    for path in image_files:
        hash_image(path)

    print("Removed " + str(removed_files) + " duplicate files!")
