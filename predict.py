import os

import numpy as np
from tensorflow.keras import Model, models
from tensorflow.keras.utils import image_dataset_from_directory

from layers import RandomBrightness

class_dirs = [x[0] for x in os.walk(r"H:\Datasets\DogBreed\train")]
class_dirs.remove("H:\\Datasets\\DogBreed\\train")
classes = [x.replace("H:\\Datasets\\DogBreed\\train\\", "") for x in class_dirs]
test_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\test", shuffle=False, batch_size=1)
file_paths = test_data.file_paths
model: Model = models.load_model("model.h5", custom_objects={"RandomBrightness": RandomBrightness})
index = 0
total_img = len(test_data)
wrong_predictions = 0
wrong_preds_per_class = dict()

for x, y in test_data:
    label = y[0]
    prediction = model.predict(x)
    prediction = np.argmax(prediction, axis=1)[0]

    if prediction != label:
        print("Wrong prediction for image " + file_paths[index] + ": Predicted '" + classes[prediction]
              + "' but should be '" + classes[label] + "'")
        wrong_predictions += 1

        if wrong_preds_per_class.get(classes[label]) is None:
            wrong_preds_per_class[classes[label]] = 0

        wrong_preds_per_class[classes[label]] = wrong_preds_per_class[classes[label]] + 1

    index += 1

print("Accuracy: " + str(100 - (wrong_predictions / total_img * 100)))
print("Wrong predictions per class:")
print(wrong_preds_per_class)
