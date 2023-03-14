import math
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()
AUTOTUNE = tf.data.AUTOTUNE
TARGET_SIZE = [192, 192]

def string_to_bytes(string, charset='utf-8'):
    return bytes(string, charset)

def convert_to_byte_classes(str_classes):
    byte_classes = []
    for c in sorted(str_classes):
        byte_classes.append(string_to_bytes(c))
    return byte_classes

def calculate_shards(gcs_pattern):
    num_images = len(tf.io.gfile.glob(gcs_pattern))
    num_shards = 1
    if num_images > 100 and num_images < 500:
        num_shards = 2
    if num_images > 500 and num_images < 1000:
        num_shards = 4
    if num_images > 1000:
        num_shards = num_images // 250
    shard_size = math.ceil(1.0 * num_images / num_shards)
    print("Pattern matches {} images which will be rewritten as {} .tfrec files containing {} images each.".format(num_images, num_shards, shard_size))
    return num_images, num_shards, shard_size

def decode_image_and_label(filename):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img_raw)
    # parse flower name from containing directory
    label = tf.strings.split(tf.expand_dims(filename, axis=-1), sep='/')
    label = label.values[-2]
    # label is string, eg: tulips
    # tf.Tensor(b'tulips', shape=(), dtype=string) <class 'tensorflow.python.framework.ops.EagerTensor'>
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
                    lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                    lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label

def recompress_image(image, label):
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label

# Three types of data can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element will be a list of size 1
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))
  
def to_tfrecord(tfrec_filewriter, classes, img_bytes, label):  
    class_num = np.argmax(np.array(classes)==label) # 'roses' => 2 (order defined in classes)
    one_hot_class = np.eye(len(classes))[class_num]     # [0, 0, 1, 0, 0] for class #2, roses

    feature = {
        "image": _bytestring_feature([img_bytes]), # one image in the list
        "class": _int_feature([class_num]),        # one class in the list
        "one_hot_class": _float_feature(one_hot_class.tolist()) # variable length  list of floats, n=len(CLASSES)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))
  
def write_tfrecord_dataset(dataset, classes, gcs_output):
    print("Writing TFRecords")
    for shard, (image, label) in enumerate(dataset):
        # shard size can be less than pre-calculated shard size in last shard
        shard_size = image.numpy().shape[0]
        # good practice to have the number of records in the filename
        filename = gcs_output + "{:02d}-{}.tfrec".format(shard, shard_size)
        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(shard_size):
                example = to_tfrecord(out_file, classes,
                                        image.numpy()[i], # re-compressed image: already a byte string
                                        label.numpy()[i],)
                out_file.write(example.SerializeToString())
            print("Wrote file {} containing {} records".format(filename, shard_size))

def to_tfrecord_dataset(data_info):
    num_images, num_shards, shard_size = calculate_shards(data_info.get('gcs_pattern'))
    byte_classes = convert_to_byte_classes(data_info.get('classes'))

    # This also shuffles the images
    filenames = tf.data.Dataset.list_files(data_info.get('gcs_pattern'), seed=35155) 
    dataset = filenames.map(decode_image_and_label, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(recompress_image, num_parallel_calls=AUTOTUNE)
    # sharding: there will be one "batch" of images per file
    dataset = dataset.batch(shard_size)
    write_tfrecord_dataset(dataset, byte_classes, data_info.get('gcs_output'))
