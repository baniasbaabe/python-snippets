import tensorflow as tf
import os

def id_to_path(img_id: str, dir: str):
    """Function returns a path to an image file.
    :param img_id: Image Id
    :param dir: Path to the directory with images
    :return: Image file path
    """
    return os.path.join(dir, f'{img_id}.jpg')


def get_image(path: str) -> tf.Tensor:
    """Function loads image from a file and preprocesses it.
    :param path: Path to image file
    :return: Tensor with preprocessed image
    """
    image = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.cast(tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE), dtype=tf.int32)
    return tf.keras.applications.efficientnet.preprocess_input(image)


def process_dataset(path: str, label: int) -> tuple:
    """Function returns preprocessed image and label.
    :param path: Path to image file
    :param label: Class label
    :return: tf.Tensor with preprocessed image, numeric label
    """
    return get_image(path), label

def get_dataset(x, y=None) -> tf.data.Dataset:
    """Function creates batched optimized dataset for the model
    out of an array of file paths and (optionally) class labels.
    :param x: Input data for the model (array of file paths)
    :param y: Target values for the model (array of class indexes)
    :return TensorFlow Dataset object
    """
    if y is not None:
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        return ds.map(process_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(x)
        return ds.map(get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)