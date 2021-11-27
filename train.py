import os

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, SpatialDropout2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom, Resizing, \
    RandomTranslation, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow.keras.regularizers import l2
from wandb.integration.keras import WandbCallback

import wandb
from layers import RandomBrightness

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

class_dirs = [x[0] for x in os.walk(r"H:\Datasets\DogBreed\train")]
class_dirs.remove("H:\\Datasets\\DogBreed\\train")
#classes = [x.replace("H:\\Datasets\\DogBreed\\train\\", "") for x in class_dirs]

total = 0
class_num = 0
class_sizes = dict()
class_weights = dict()

for class_dir in class_dirs:
    amount = len(os.listdir(class_dir))
    total += amount
    class_sizes[class_num] = amount
    class_num += 1

class_num = 0
for class_dir in class_dirs:
    class_weights[class_num] = total / class_sizes[class_num] / len(class_dirs)
    class_num += 1


print("Loading data...")
train_batch_size = 128
val_batch_size = 128
seed = None
image_size = 180

train_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\train", batch_size=train_batch_size, seed=seed).prefetch(AUTOTUNE)
val_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\val", batch_size=val_batch_size, seed=seed).prefetch(AUTOTUNE)
test_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\test", batch_size=val_batch_size, seed=seed).prefetch(AUTOTUNE)

print("Building network...")

model = Sequential([
    Resizing(image_size, image_size),
    Rescaling(scale=1. / 255),
    RandomTranslation(width_factor=0.1, height_factor=0.1, seed=seed),
    RandomFlip(mode="horizontal", seed=seed),
    RandomBrightness(factor=0.2, seed=seed),
    RandomZoom(height_factor=0.1, width_factor=0.1, seed=seed),
    RandomRotation(factor=0.1, seed=seed),
    #RandomGrayscale(factor=0.25, seed=seed),

    Conv2D(32, 9, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    Conv2D(32, 9, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    MaxPooling2D(),

    Conv2D(64, 3, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    Conv2D(64, 3, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    MaxPooling2D(),

    Conv2D(128, 3, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    Conv2D(128, 3, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    MaxPooling2D(),

    Conv2D(256, 3, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    Conv2D(256, 3, activation="relu"),
    #SpatialDropout2D(0.4, seed=seed),
    MaxPooling2D(),

    Conv2D(512, 3, activation="relu"),
    # SpatialDropout2D(0.4, seed=seed),
    Conv2D(512, 3, activation="relu"),
    # SpatialDropout2D(0.4, seed=seed),
    MaxPooling2D(),

    Flatten(),
    Dropout(0.2, seed=seed),
    Dense(len(class_dirs), activation="softmax", dtype="float32")
])

opt = Adam(learning_rate=0.0001)
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=opt)

print("Training...")
wandb.init(project="dogbreed-ai")
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.05, patience=7, monitor="val_loss")
model.fit(train_data, epochs=2000, validation_data=val_data, callbacks=[WandbCallback(), es, reduce_lr], class_weight=class_weights)

print("Evaluating...")
model.evaluate(test_data)

print("Saving...")
model.save("model.h5", include_optimizer=False)
