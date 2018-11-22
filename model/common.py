import tensorflow as tf


class RegressionBias(object):
    def __init__(self):
        self.bias = {}

    def update(self, predictions, truth):
        try:
            len(predictions)
        except TypeError:
            predictions = [predictions]
            truth = [truth]

        if not (len(predictions) == len(truth)):
            print("==> incompatible length: predictions({0}), truth({1})".format(len(predictions), len(truth)))
            return

        for i in range(len(predictions)):
            bias = predictions[i] - truth[i]
            bias = int(bias + (0.5 if bias > 0 else -0.5))
            if bias not in self.bias:
                self.bias[bias] = 1
            else:
                self.bias[bias] += 1

    def generate_result(self):
        key_list = self.bias.keys()
        key_list.sort()
        result = "bias distribution:\n"
        for key in key_list:
            result += "{0:>6}: {1}\n".format(key, self.bias[key])
        return result


class DummyFile(object):
    def writestr(self, name, content):
        pass
    def close(self):
        pass


class EndSavingHook(tf.train.SessionRunHook):
    def __init__(self, module_list, save_path):
        self.module_list = module_list
        self.save_path = save_path

    def end(self, session):
        for item in self.module_list:
            item.save(sess=session, path=self.save_path)


class LoadInitialValueHook(tf.train.SessionRunHook):
    def __init__(self, module_list, save_path):
        self.module_list = module_list
        self.save_path = save_path

    def after_create_session(self, session, coord):
        for item in self.module_list:
            item.load_once(sess=session, path=self.save_path)


def new_fc_layer(layer_last, num_inputs, num_outputs, use_relu=True):
    """
    Create a new fully connected layer.
    :param layer_last: The previous layer.
    :param num_inputs: Num. inputs from prev. layer.
    :param num_outputs: Num. outputs.
    :param use_relu: Use Rectified Linear Unit (ReLU)?
    :return: 
    """

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(layer_last, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def new_conv_layer(layer_last, num_input_channels, filter_size, num_filters, use_pooling=True):
    """
    Create a new convolution layer.
    :param layer_last: The previous layer.
    :param num_input_channels: Num. channels in prev. layer.
    :param filter_size: Width and height of each filter.
    :param num_filters: Number of filters.
    :param use_pooling: Use 2x2 max-pooling.
    :return: Result layer and layer weights.
    """

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=layer_last,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name='weight')


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]), name='bias')
