from __future__ import print_function
import json
import os
from datetime import datetime

import tensorflow as tf
import numpy as np
from scipy.io import loadmat

from data.common import TfReader
from model.common import new_fc_layer, EndSavingHook, RegressionBias


class Module(object):
    def __init__(self):
        self.variable_scope = ""

    def load(self, sess, path):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope))
        saver.restore(sess=sess, save_path=os.path.join(path, self.variable_scope))

    def save(self, sess, path):
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope))
        saver.save(sess=sess, save_path=os.path.join(path, self.variable_scope))


class SourceVgg(Module):
    def __init__(self, original_model_path, trainable_layers=None, feature_layer="pool5"):
        """
        Code from https://github.com/ZZUTK/Tensorflow-VGG-face, modified by Dick Zhou.
        """

        super(SourceVgg, self).__init__()
        self.variable_scope = "vgg_source"

        data = loadmat(original_model_path)
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="vgg_source_input")

        # read meta info
        meta = data['meta']
        classes = meta['classes']
        class_names = classes[0][0]['description'][0][0]
        normalization = meta['normalization']
        average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
        image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
        input_maps = tf.image.resize_images(input_image, (image_size[0], image_size[1]))

        # read layer info
        layers = data['layers']
        current = input_maps - average_image
        network = {}
        weights = {}
        with tf.variable_scope(self.variable_scope):
            for layer in layers[0]:
                name = layer[0]['name'][0][0]
                layer_type = layer[0]['type'][0][0]
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = layer[0]['stride'][0][0]
                    kernel, bias = layer[0]['weights'][0][0]
                    weight = tf.Variable(kernel, name="weight_" + name)
                    bias = tf.Variable(np.squeeze(bias).reshape(-1), name="bias_" + name)
                    conv = tf.nn.conv2d(current, weight,
                                        strides=(1, stride[0], stride[0], 1), padding=padding)
                    current = tf.nn.bias_add(conv, bias)

                    weights[name] = [weight, bias]
                elif layer_type == 'relu':
                    current = tf.nn.relu(current)

                    weights[name] = []
                elif layer_type == 'pool':
                    stride = layer[0]['stride'][0][0]
                    pool = layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')

                    weights[name] = []
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))

                    weights[name] = []

                network[name] = current

        # config
        self.image_input = input_image
        self.feature = network[feature_layer]
        self.trainable_list = []
        self.weights = weights

        for name in trainable_layers:
            self.trainable_list += weights[name]


class TargetVgg(Module):
    def __init__(self, original_model_path, trainable_layers=None, feature_layer="pool5", source_model=None):
        """
        Code from https://github.com/ZZUTK/Tensorflow-VGG-face, modified by Dick Zhou.
        """

        super(TargetVgg, self).__init__()
        self.variable_scope = "vgg_target"

        data = loadmat(original_model_path)
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="vgg_target_input")

        # read meta info
        meta = data['meta']
        classes = meta['classes']
        class_names = classes[0][0]['description'][0][0]
        normalization = meta['normalization']
        average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
        image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
        input_maps = tf.image.resize_images(input_image, (image_size[0], image_size[1]))

        # read layer info
        layers = data['layers']
        current = input_maps - average_image
        network = {}
        weights = {}
        with tf.variable_scope(self.variable_scope):
            for layer in layers[0]:
                name = layer[0]['name'][0][0]
                layer_type = layer[0]['type'][0][0]
                if layer_type == 'conv':
                    if name[:2] == 'fc':
                        padding = 'VALID'
                    else:
                        padding = 'SAME'
                    stride = layer[0]['stride'][0][0]

                    if source_model:
                        variables = source_model.weights[name]
                    else:
                        kernel, bias = layer[0]['weights'][0][0]
                        variables = [kernel, np.squeeze(bias).reshape(-1)]

                    weight = tf.Variable(variables[0], name="weight_" + name)
                    bias = tf.Variable(variables[1], name="bias_" + name)

                    conv = tf.nn.conv2d(current, weight,
                                        strides=(1, stride[0], stride[0], 1), padding=padding)
                    current = tf.nn.bias_add(conv, bias)

                    weights[name] = [weight, bias]
                elif layer_type == 'relu':
                    current = tf.nn.relu(current)

                    weights[name] = []
                elif layer_type == 'pool':
                    stride = layer[0]['stride'][0][0]
                    pool = layer[0]['pool'][0][0]
                    current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                             strides=(1, stride[0], stride[0], 1), padding='SAME')

                    weights[name] = []
                elif layer_type == 'softmax':
                    current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))

                    weights[name] = []

                network[name] = current

        # config
        self.image_input = input_image
        self.feature = network[feature_layer]
        self.trainable_list = []
        self.weights = weights

        for name in trainable_layers:
            self.trainable_list += weights[name]


class NnRegression(Module):
    def __init__(self, feature, n_hidden=4096):
        super(NnRegression, self).__init__()
        self.variable_scope = "nn_regression"

        with tf.variable_scope(self.variable_scope):
            num_features = feature.get_shape()[1:].num_elements()
            fc_input = tf.reshape(feature, [-1, num_features])

            fc_hidden = new_fc_layer(layer_last=fc_input,
                                     num_inputs=num_features,
                                     num_outputs=n_hidden,
                                     use_relu=True)

            fc_output = new_fc_layer(layer_last=fc_hidden,
                                     num_inputs=n_hidden,
                                     num_outputs=1,
                                     use_relu=False)

            self.label_input = tf.placeholder(dtype=tf.float32)
            self.prediction = fc_output

            self.loss = tf.reduce_sum(tf.pow(tf.transpose(self.prediction) - self.label_input, 2))


class NnClassification(Module):
    def __init__(self, feature, n_classes, n_hidden=4096):
        super(NnClassification, self).__init__()
        self.variable_scope = "nn_regression"

        with tf.variable_scope(self.variable_scope):
            num_features = feature.get_shape()[1:].num_elements()
            fc_input = tf.reshape(feature, [-1, num_features])

            fc_hidden = new_fc_layer(layer_last=fc_input,
                                     num_inputs=num_features,
                                     num_outputs=n_hidden,
                                     use_relu=True)

            fc_output = new_fc_layer(layer_last=fc_hidden,
                                     num_inputs=n_hidden,
                                     num_outputs=n_classes,
                                     use_relu=False)

            self.label_input = tf.placeholder(dtype=tf.float32)
            self.prediction = tf.argmax(tf.nn.softmax(fc_output), dimension=1)

            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_output, labels=self.label_input))


def pre_train(config):
    print("==> pre-train started at {0}.".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("---- CONFIG DUMP ----")
    print(json.dumps(config, indent=1))
    print("---- END ----")
    source_feature_module = SourceVgg(original_model_path=config["vggface_model"],
                                      trainable_layers=config["trainable_layers"],
                                      feature_layer=config["feature_layer"])
    regression_module = NnRegression(feature=source_feature_module.feature)
    image, label = TfReader(data_path=config["source_data"]["path"], regression=True, size=(224, 224),
                            num_epochs=config["source_data"]["epoch"]) \
        .read(batch_size=config["source_data"]["batch_size"])

    global_step_op = tf.Variable(0, trainable=False, name="global_step")
    var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, regression_module.variable_scope) + \
                   source_feature_module.trainable_list
    optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"]).minimize(loss=regression_module.loss,
                                                                                       global_step=global_step_op,
                                                                                       var_list=var_to_train,
                                                                                       colocate_gradients_with_ops=True)
    print("optimizing variables: {0}".format(var_to_train))

    if config["checkpoint"]["enabled"]:
        checkpoint = config["checkpoint"]["path"]
    else:
        checkpoint = None
    hooks = [EndSavingHook(module_list=[source_feature_module, regression_module], save_path=config["save_root"])]
    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=checkpoint) as mon_sess:
        global_step = -1
        current_cost = -1
        try:
            while not mon_sess.should_stop():
                image_batch, label_batch = mon_sess.run([image, label])
                _, global_step, current_cost = mon_sess.run([optimizer, global_step_op, regression_module.loss],
                                                            feed_dict={
                                                                source_feature_module.image_input: image_batch,
                                                                regression_module.label_input: label_batch
                                                            })
                if global_step % config["report_rate"] == 0:
                    print("-- cost at ({0}) : {1}.".format(global_step, current_cost))
        except tf.errors.OutOfRangeError:
            pass

    with open(os.path.join(config["save_root"], "gan_vgg.log"), 'a') as log_file:
        message = "==> pre-train complete at {0} in {1} steps.".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                       global_step)
        log_file.write(message + "\n")
        print(message)

        log_file.write("---- CONFIG DUMP ----\n")
        json.dump(config, log_file, indent=1)
        log_file.write("---- END ----\n")

        message = "cost:  {0:8}".format(current_cost)
        log_file.write(message + "\n")
        print(message)


def adaption(config):
    print("==> adaption started at {0}.".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("---- CONFIG DUMP ----")
    print(json.dumps(config, indent=1))
    print("---- END ----")

    source_feature_module = SourceVgg(original_model_path=config["vggface_model"],
                                      trainable_layers=config["trainable_layers"],
                                      feature_layer=config["feature_layer"])
    with tf.Session() as sess:
        print("loading models...", end='')
        source_feature_module.load(sess=sess, path=config["save_root"])
        print("done.")

    target_feature_module = TargetVgg(original_model_path=config["vggface_model"], source_model=source_feature_module,
                                      trainable_layers=config["trainable_layers"],
                                      feature_layer=config["feature_layer"])
    discriminator_module = NnClassification(feature=target_feature_module.feature, n_classes=2)
    source_image, _ = TfReader(data_path=config["source_data"]["path"], regression=True, size=(224, 224),
                               num_epochs=config["source_data"]["epoch"]) \
        .read(batch_size=config["source_data"]["batch_size"])
    target_image, _ = TfReader(data_path=config["target_data"]["path"], regression=True, size=(224, 224),
                               num_epochs=config["target_data"]["epoch"]) \
        .read(batch_size=config["target_data"]["batch_size"])

    global_step_op = tf.Variable(0, trainable=False, name="global_step")
    optimizer_d = tf.train.AdamOptimizer(learning_rate=config["learning_rate"]) \
        .minimize(loss=discriminator_module.loss,
                  global_step=global_step_op,
                  var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope=discriminator_module.variable_scope),
                  colocate_gradients_with_ops=True)
    optimizer_m = tf.train.AdamOptimizer(learning_rate=config["learning_rate"]) \
        .minimize(loss=discriminator_module.loss,
                  global_step=global_step_op,
                  var_list=target_feature_module.trainable_list,
                  colocate_gradients_with_ops=True)

    if config["checkpoint"]["enabled"]:
        checkpoint = config["checkpoint"]["path"]
    else:
        checkpoint = None
    hooks = [EndSavingHook(module_list=[target_feature_module, discriminator_module], save_path=config["save_root"])]
    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=checkpoint) as mon_sess:
        global_step = -1
        cost_d = -1
        cost_m = -1
        try:
            while not mon_sess.should_stop():
                cost_d = 0
                cost_m = 0

                # read image and compute the feature
                source_image_batch, target_image_batch = mon_sess.run([source_image, target_image])
                source_feature_batch, target_feature_batch = mon_sess.run([source_feature_module.feature, target_feature_module.feature],
                                                                          feed_dict={
                                                                              source_feature_module.image_input: source_image_batch,
                                                                              target_feature_module.image_input: target_image_batch
                                                                          })

                # optimization
                _, global_step, current_cost = mon_sess.run([optimizer_d, global_step_op, discriminator_module.loss],
                                                            feed_dict={
                                                                target_feature_module.feature: source_feature_batch,
                                                                discriminator_module.label_input: [0] * config["target_data"]["batch_size"]
                                                            })
                cost_d += current_cost
                if global_step % config["report_rate"] == 0:
                    print("-- cost of d (source) at ({0}) : {1}.".format(global_step, current_cost))
                _, global_step, current_cost = mon_sess.run([optimizer_d, global_step_op, discriminator_module.loss],
                                                            feed_dict={
                                                                target_feature_module.feature: target_feature_batch,
                                                                discriminator_module.label_input: [1] * config["target_data"]["batch_size"]
                                                            })
                cost_d += current_cost
                if global_step % config["report_rate"] == 0:
                    print("-- cost of d (target) at ({0}) : {1}.".format(global_step, current_cost))
                _, global_step, current_cost = mon_sess.run([optimizer_m, global_step_op, discriminator_module.loss],
                                                            feed_dict={
                                                                target_feature_module.image_input: target_image,
                                                                discriminator_module.label_input: [0] * config["target_data"]["batch_size"]
                                                            })
                cost_m += current_cost
                if global_step % config["report_rate"] == 0:
                    print("-- cost of m at ({0}) : {1}.".format(global_step, current_cost))

        except tf.errors.OutOfRangeError:
            pass

    with open(os.path.join(config["save_root"], "gan_vgg.log"), 'a') as log_file:
        message = "==> adaption complete at {0} in {1} steps.".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                      global_step)
        log_file.write(message + "\n")
        print(message)

        log_file.write("---- CONFIG DUMP ----\n")
        json.dump(config, log_file, indent=1)
        log_file.write("---- END ----\n")

        message = "cost_d:  {0:8}, cost_m:  {0:8}".format(cost_d, cost_m)
        log_file.write(message + "\n")
        print(message)


def test(config, vgg=TargetVgg):
    print("==> test started at {0} for {1}.".format(datetime.now().strftime("%Y-%m-%d %H:%M"), vgg))
    print("---- CONFIG DUMP ----")
    print(json.dumps(config, indent=1))
    print("---- END ----")

    feature_module = vgg(original_model_path=config["vggface_model"],
                         trainable_layers=config["trainable_layers"], feature_layer=config["feature_layer"])
    regression_module = NnRegression(feature=feature_module.feature)
    image, label = TfReader(data_path=config["target_data"]["path"], regression=True, size=(224, 224),
                            num_epochs=config["target_data"]["epoch"]) \
        .read(batch_size=config["target_data"]["batch_size"])

    accuracy = tf.reduce_mean(tf.cast(tf.abs(tf.transpose(regression_module.prediction) - label), tf.float32))
    statistics = RegressionBias()

    # MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession() as mon_sess:
        accumulated_accuracy = 0
        test_step = 0
        print("loading models...", end='')
        feature_module.load(sess=mon_sess, path=config["save_root"])
        regression_module.load(sess=mon_sess, path=config["save_root"])
        print("done.")
        try:
            while not mon_sess.should_stop():
                test_step += 1
                image_batch, label_batch = mon_sess.run([image, label])
                prediction, accuracy = mon_sess.run([regression_module.prediction, accuracy], feed_dict={
                    feature_module.image_input: image_batch
                })
                accumulated_accuracy += accuracy
                statistics.update(predictions=prediction, truth=label_batch)
                if test_step % config["report_rate"] == 0:
                    print("-- accuracy {0:>8}: {1:8}".format(test_step, accumulated_accuracy / test_step))
        except tf.errors.OutOfRangeError:
            pass

    with open(os.path.join(config["save_root"], "gan_vgg.log"), 'a') as log_file:
        message = "==> test for {2} complete at {0} in {1} steps.".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                          test_step, vgg)
        log_file.write(message + "\n")
        print(message)

        log_file.write("---- CONFIG DUMP ----\n")
        json.dump(config, log_file, indent=1)
        log_file.write("---- END ----\n")

        message = "overall result:  {0:8}".format(accumulated_accuracy / test_step)
        log_file.write(message + "\n")
        print(message)

        message = statistics.generate_result()
        log_file.write(message + "\n")
        print(message)


def _main():
    import argparse

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser(description="gan with vgg.")
    parser.add_argument("action", choices=["pretrain", "adaption", "test"],
                        help="action to perform")
    parser.add_argument("-c", "--config", default="gan_vgg.config",
                        help="path to config file")

    parser.add_argument("--test-using-source", action="store_true",
                        help="test source feature performance on target")

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    if args.action == "pretrain":
        pre_train(config)
    elif args.action == "adaption":
        adaption(config)
    elif args.action == "test" and args.test_using_source:
        # noinspection PyTypeChecker
        test(config, vgg=SourceVgg)
    else:
        test(config)


if __name__ == "__main__":
    _main()
