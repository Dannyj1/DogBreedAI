import os

import tensorflow as tf
import wandb
from albumentations import Compose, HorizontalFlip, \
    VerticalFlip, RandomBrightnessContrast, HueSaturationValue, \
    GaussNoise, ShiftScaleRotate, Blur, RGBShift, ChannelShuffle, OneOf, MotionBlur, MedianBlur, \
    RandomGamma
from tensorflow.keras import Sequential
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, BatchNormalization, SpatialDropout2D, GlobalMaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory
from wandb.integration.keras import WandbCallback

mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

class_dirs = [x[0] for x in os.walk(r"H:\Datasets\DogBreed\train")]
class_dirs.remove(r"H:\Datasets\DogBreed\train")

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
batch_size = 64
seed = None
image_size = 124

transforms = Compose([
    ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.1, rotate_limit=70),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    OneOf([
        MotionBlur(p=.2),
        MedianBlur(blur_limit=3, p=0.1),
        Blur(blur_limit=3, p=0.1),
    ], p=0.5),
    OneOf([
        RandomBrightnessContrast(p=0.5, brightness_limit=0.6, contrast_limit=0.4),
        RandomGamma(p=0.5),
    ], p=0.5),
    GaussNoise(p=1.0, always_apply=True, var_limit=(10.0, 50.0)),
    OneOf([ChannelShuffle(p=0.5), RGBShift(p=0.5), HueSaturationValue(p=0.5)], p=0.3),
])


def aug_fn(image):
    data = {"image": image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]

    return aug_img


def augment_data(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
    tf.map_fn(lambda elem: tf.numpy_function(func=aug_fn, inp=[elem], Tout=tf.float32), image)

    return preprocess_data(image, label)


def preprocess_data(image, label):
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0

    return image, label


train_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\train", batch_size=batch_size, seed=seed).map(lambda image, label: preprocess_data(image, label), num_parallel_calls=AUTOTUNE).cache()\
    .map(lambda image, label: augment_data(image, label), num_parallel_calls=AUTOTUNE).shuffle(64).prefetch(AUTOTUNE)
val_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\val", batch_size=batch_size, seed=seed).map(lambda image, label: preprocess_data(image, label), num_parallel_calls=AUTOTUNE).cache()
test_data = image_dataset_from_directory(r"H:\Datasets\DogBreed\test", batch_size=batch_size, seed=seed).map(lambda image, label: preprocess_data(image, label), num_parallel_calls=AUTOTUNE)
print("Building network...")

model = Sequential([
    Conv2D(32, 5, activation="PReLU"),
    BatchNormalization(),
    SpatialDropout2D(0.25, seed=seed),
    MaxPooling2D(),

    Conv2D(64, 3, activation="PReLU"),
    BatchNormalization(),
    Conv2D(64, 3, activation="PReLU"),
    BatchNormalization(),
    SpatialDropout2D(0.25, seed=seed),
    MaxPooling2D(),

    Conv2D(128, 3, activation="PReLU"),
    BatchNormalization(),
    Conv2D(128, 3, activation="PReLU"),
    BatchNormalization(),
    SpatialDropout2D(0.25, seed=seed),
    GlobalMaxPooling2D(),

    Dense(256, activation="PReLU"),
    Dropout(0.4),
    Dense(128, activation="PReLU"),
    Dropout(0.4),
    Dense(64, activation="PReLU"),
    Dropout(0.2),
    Dense(len(class_dirs), activation="softmax", dtype="float32")
])

opt = Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=opt)

print("Training...")
wandb.init(project="dogbreed-ai")
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
model.fit(train_data, epochs=2000, validation_data=val_data, callbacks=[WandbCallback(), es], class_weight=class_weights)

print("Saving...")
model.save("model.h5", include_optimizer=False)

print("Evaluating...")
model.evaluate(test_data)
model.summary()