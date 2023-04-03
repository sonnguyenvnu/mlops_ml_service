import math
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description='Create TFRecord dataset')
parser.add_argument('--gcs_folder', type=str, default="", metavar='N',
                    help='GCS image folder')
parser.add_argument('--gcs_output', type=str, default="", metavar='N',
                    help='Dataset url')
parser.add_argument('--classes', default=[], nargs='*')
parser.add_argument('--target_size', type=int, default=224, metavar='N',
                    help='Target size')

args = parser.parse_args()
# Pay attention to the classes parameter
print('Args ====> ', args)


def string_to_bytes(string, charset='utf-8'):
    return bytes(string, charset)


def convert_to_byte_classes(str_classes):
    byte_classes = []
    for c in sorted(str_classes):
        byte_classes.append(string_to_bytes(c))
    return byte_classes


AUTOTUNE = tf.data.AUTOTUNE
GCS_FOLDER = args.gcs_folder
GCS_PATTERN_JPG = f"{GCS_FOLDER}/*/*.jpg"
GCS_PATTERN_JPEG = f"{GCS_FOLDER}/*/*.jpeg"
GCS_PATTERN_PNG = f"{GCS_FOLDER}/*/*.png"
GCS_OUTPUT = args.gcs_output
TARGET_SIZE = [args.target_size, args.target_size]
CLASSES = convert_to_byte_classes(args.classes)


def calculate_shards(num_images):
    num_shards = 1
    if num_images > 100 and num_images < 500:
        num_shards = 2
    if num_images > 500 and num_images < 1000:
        num_shards = 4
    if num_images > 1000:
        num_shards = num_images // 250
    shard_size = math.ceil(1.0 * num_images / num_shards)
    print("Total {} images which will be rewritten as {} .tfrec files containing {} images each.".format(
        num_images, num_shards, shard_size))
    return num_shards, shard_size


def decode_jpeg_and_label(filename):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img_raw)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    return img, label


def decode_png_and_label(filename):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_png(img_raw)
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    return img, label


def resize_and_crop_image(image, label):
    # Resize and crop using "fill" algorithm:
    # always make sure the resulting image
    # is cut out from the source image so that
    # it fills the TARGET_SIZE entirely with no
    # black bars and a preserved aspect ratio.
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE[1]
    th = TARGET_SIZE[0]
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(
                        image, [w*tw/w, h*tw/w]),  # if true
                    lambda: tf.image.resize(
                        image, [w*th/h, h*th/h])  # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(
        image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label


def recompress_image(image, label):
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(
        image, optimize_size=True, chroma_downsampling=False)
    return image, label

# Three types of data can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element will be a list of size 1


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints):  # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(tfrec_filewriter, classes, img_bytes, label):
    # 'roses' => 2 (order defined in classes)
    class_num = np.argmax(np.array(classes) == label)
    # [0, 0, 1, 0, 0] for class #2, roses
    one_hot_class = np.eye(len(classes))[class_num]

    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "class": _int_feature([class_num]),        # one class in the list
        # variable length  list of floats, n=len(CLASSES)
        "one_hot_class": _float_feature(one_hot_class.tolist())
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_dataset(dataset, classes, gcs_output):
    print("Writing TFRecords")
    for shard, (image, label) in enumerate(dataset):
        # shard size can be less than pre-calculated shard size in last shard
        shard_size = image.numpy().shape[0]
        # good practice to have the number of records in the filename
        filename = gcs_output + "{:02d}-{}.tfrec".format(shard, shard_size)
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                example = to_tfrecord(out_file, classes,
                                      # re-compressed image: already a byte string
                                      image.numpy()[i],
                                      label.numpy()[i],)
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(
                filename, shard_size))


def write_tfrecord_dataset():
    dataset_jpg, dataset_jpeg, dataset_png = None, None, None
    if len(tf.io.gfile.glob(GCS_PATTERN_JPG)) > 0:
        filenames_jpg = tf.data.Dataset.list_files(GCS_PATTERN_JPG, seed=35155)
        dataset_jpg = filenames_jpg.map(
            decode_jpeg_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    if len(tf.io.gfile.glob(GCS_PATTERN_JPEG)) > 0:
        filenames_jpeg = tf.data.Dataset.list_files(
            GCS_PATTERN_JPEG, seed=35155)
        dataset_jpeg = filenames_jpeg.map(
            decode_jpeg_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    if len(tf.io.gfile.glob(GCS_PATTERN_PNG)) > 0:
        filenames_png = tf.data.Dataset.list_files(GCS_PATTERN_PNG, seed=35155)
        dataset_png = filenames_png.map(
            decode_png_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    datasets = []
    if dataset_jpg is not None:
        datasets.append(dataset_jpg)
    if dataset_jpeg is not None:
        datasets.append(dataset_jpeg)
    if dataset_png is not None:
        datasets.append(dataset_png)

    if len(datasets) > 0:
        dataset = datasets[0]
    for i in range(1, len(datasets)):
        dataset = dataset.concatenate(datasets[i])
    else:
        dataset = None
    num_images = int(tf.data.experimental.cardinality(dataset).numpy())
    print('Total images:', num_images)

    num_shards, shard_size = calculate_shards(num_images)
    dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(recompress_image, num_parallel_calls=AUTOTUNE)
    # sharding: there will be one "batch" of images per file
    dataset = dataset.batch(shard_size)
    write_dataset(dataset, CLASSES, GCS_OUTPUT)


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.MirroredStrategy()

print("Number of accelerators: ", strategy.num_replicas_in_sync)
MIXED_PRECISION = False
if MIXED_PRECISION:
    if tpu:
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    else:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.config.optimizer.set_jit(True)  # XLA compilation
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Mixed precision enabled')
with strategy.scope():
    write_tfrecord_dataset()

# python3 dataset.py --gcs_folder=gs://uet-mlops/images/641ac7fd897b5152aaa371e9 --gcs_output=gs://uet-mlops/flowers/ --target_size=224 --classes ants bees
