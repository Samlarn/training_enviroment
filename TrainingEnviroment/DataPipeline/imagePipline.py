"""
Creates an tf Dataset that map training images to their correspoding labels as tensors.

Notes: 
  - needs to be more dynamic.
  - also add other types of datasets, like: MapDataset and PrefetchDataset
"""
import tensorflow as tf
import pathlib
import random
import os

width = 120
height = 120
channels = 3

def prepare_images(path_to_images, width, height, channels=3):
    width = width
    height = height
    data_root = pathlib.Path(path_to_images)
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    image_count = len(all_image_paths)
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index,name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    return all_image_paths, all_image_labels, image_count


def get_zip_dataset(all_image_paths, all_image_labels):
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    return tf.data.Dataset.zip((image_ds, label_ds))


def get_map_dataset(all_image_paths, all_image_labels):
    ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
    return ds.map(load_and_preprocess_from_path_label)


def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels)
    image = tf.image.resize(image, [width, height])
    image /= 255.0
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)





