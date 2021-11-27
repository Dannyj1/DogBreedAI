from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer


def visualize(original, augmented):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)


class RandomBrightness(Layer):
    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
        self.seedarg = seed

        if seed is not None:
            self.seed = (1, seed)
        else:
            self.seed = (1, tf.random.Generator.from_non_deterministic_state().make_seeds()[0][0])

    def call(self, x, training=True, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase.backend.learning_phase()

        if training:
            return tf.image.stateless_random_brightness(x, self.factor, self.seed)
        else:
            return x

    def get_config(self):
        return {
            "seed": self.seedarg,
            "factor": self.factor
        }


class RandomGrayscale(Layer):
    def __init__(self, factor, seed=None, **kwargs):
        super(RandomGrayscale, self).__init__(**kwargs)
        self.factor = factor
        self.seedarg = seed

        if seed is not None:
            self.seed = (1, seed)
        else:
            self.seed = (1, tf.random.Generator.from_non_deterministic_state().make_seeds()[0][0])

        self.rng = tf.random.Generator.from_seed(self.seed)

    def call(self, x, training=True, **kwargs):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            if self.rng.uniform(shape=(1,), minval=0, maxval=1)[0] <= self.factor:
                return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))
            else:
                return x
        else:
            return x

    def get_config(self):
        base_config = super(RandomGrayscale, self).get_config()
        base_config["seed"] = self.seedarg

        return base_config
