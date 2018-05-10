import tensorflow as tf


class TfReader(object):
    def __init__(self, data_path, regression=False, size=(512, 512), num_epochs=1):
        self.reader = tf.TFRecordReader()
        self.filename_queue = tf.train.string_input_producer(data_path, num_epochs=num_epochs)
        self.size = size

        self.data_path = data_path
        self.num_epoch = num_epochs

        if regression:
            self.label_type = tf.float32
        else:
            self.label_type = tf.int64

    def read(self, batch_size=50, num_threads=4, capacity=1000, min_after_dequeue=100):
        """
        Read tfrecords file containing image and label.
        :param batch_size: Size of bath.
        :param num_threads: Number of threads used.
        :param capacity: Maximal examples in memory.
        :param min_after_dequeue: Minimal examples for starting training.
        :return: Images and labels array of bath size.
        """

        feature_structure = {'image': tf.FixedLenFeature([], tf.string),
                             'label': tf.FixedLenFeature([], self.label_type)}

        # Read serialized data.
        _, serialized_example = self.reader.read(self.filename_queue)
        features = tf.parse_single_example(serialized=serialized_example, features=feature_structure)

        # Cast to original format.
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [self.size[0], self.size[1], 3])
        label = tf.cast(features['label'], self.label_type)

        # Random bathing.
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                capacity=capacity,
                                                num_threads=num_threads,
                                                min_after_dequeue=min_after_dequeue)
        return images, labels
