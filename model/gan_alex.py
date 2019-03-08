from __future__ import print_function

import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from data.common import TfReader
from model.common import new_fc_layer, EndSavingHook, LoadInitialValueHook, DummyFile, new_conv_layer, ConfusionMatrix


class Module(object):
    def __init__(self, variable_scope):
        self.variable_scope = variable_scope
        self.saver = None
        self.loaded = False

    def _build_saver(self):
        self.saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.variable_scope))

    def load_once(self, sess, path):
        if not self.loaded:
            print("loading saved {0} model...".format(self.variable_scope))
            self.saver.restore(sess=sess, save_path=os.path.join(path, self.variable_scope))
            self.loaded = True

    def save(self, sess, path):
        self.saver.save(sess=sess, save_path=os.path.join(path, self.variable_scope))


class Source(Module):
    def __init__(self):
        """
        Code from https://github.com/ZZUTK/Tensorflow-VGG-face, modified by Dick Zhou.
        """

        super(Source, self).__init__(variable_scope="source")

        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="source_input")

        # hyper-parameters.
        filter_size1 = 3
        num_filters1 = 32

        filter_size2 = 3
        num_filters2 = 64

        filter_size3 = 3
        num_filters3 = 128

        filter_size4 = 3
        num_filters4 = 48

        with tf.variable_scope(self.variable_scope):
            # building layers.
            layer_conv1, weights_conv1 = new_conv_layer(layer_last=input_image,
                                                        num_input_channels=3,
                                                        filter_size=filter_size1,
                                                        num_filters=num_filters1,
                                                        use_pooling=True)

            layer_conv2, weights_conv2 = new_conv_layer(layer_last=layer_conv1,
                                                        num_input_channels=num_filters1,
                                                        filter_size=filter_size2,
                                                        num_filters=num_filters2,
                                                        use_pooling=True)

            layer_conv3, weights_conv3 = new_conv_layer(layer_last=layer_conv2,
                                                        num_input_channels=num_filters2,
                                                        filter_size=filter_size3,
                                                        num_filters=num_filters3,
                                                        use_pooling=True)

            layer_conv4, weights_conv4 = new_conv_layer(layer_last=layer_conv3,
                                                        num_input_channels=num_filters3,
                                                        filter_size=filter_size4,
                                                        num_filters=num_filters4,
                                                        use_pooling=True)

        # config
        self.image_input = input_image
        self.feature = layer_conv4
        self.trainable_list = weights_conv1 + weights_conv2 + weights_conv3 + weights_conv4

        self._build_saver()


class Target(Module):
    def __init__(self):
        """
        Code from https://github.com/ZZUTK/Tensorflow-VGG-face, modified by Dick Zhou.
        """

        super(Target, self).__init__(variable_scope="target")

        input_image = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="target_input")

        # hyper-parameters.
        filter_size1 = 3
        num_filters1 = 32

        filter_size2 = 3
        num_filters2 = 64

        filter_size3 = 3
        num_filters3 = 128

        filter_size4 = 3
        num_filters4 = 48

        with tf.variable_scope(self.variable_scope):
            # building layers.
            layer_conv1, weights_conv1 = new_conv_layer(layer_last=input_image,
                                                        num_input_channels=3,
                                                        filter_size=filter_size1,
                                                        num_filters=num_filters1,
                                                        use_pooling=True)

            layer_conv2, weights_conv2 = new_conv_layer(layer_last=layer_conv1,
                                                        num_input_channels=num_filters1,
                                                        filter_size=filter_size2,
                                                        num_filters=num_filters2,
                                                        use_pooling=True)

            layer_conv3, weights_conv3 = new_conv_layer(layer_last=layer_conv2,
                                                        num_input_channels=num_filters2,
                                                        filter_size=filter_size3,
                                                        num_filters=num_filters3,
                                                        use_pooling=True)

            layer_conv4, weights_conv4 = new_conv_layer(layer_last=layer_conv3,
                                                        num_input_channels=num_filters3,
                                                        filter_size=filter_size4,
                                                        num_filters=num_filters4,
                                                        use_pooling=True)

        # config
        self.image_input = input_image
        self.feature = layer_conv4
        self.trainable_list = weights_conv1 + weights_conv2 + weights_conv3 + weights_conv4

        self._init_path = None
        self._init_saver = None

        self._build_saver()

    def override_saver_for_init_by(self, source_model):
        init_config = {}
        for i in range(len(self.trainable_list)):
            w_s = source_model.trainable_list[i]
            w_t = self.trainable_list[i]
            init_config[w_s.name.split(':')[0]] = w_t

        self._init_saver = tf.train.Saver(init_config)
        self._init_path = source_model.variable_scope

    def load_once(self, sess, path):
        if not self.loaded:
            if self._init_path:
                print("init {0} model by copy...".format(self.variable_scope))
                self._init_saver.restore(sess=sess, save_path=os.path.join(path, self._init_path))
            else:
                print("loading saved {0} model...".format(self.variable_scope))
                self.saver.restore(sess=sess, save_path=os.path.join(path, self.variable_scope))
            self.loaded = True


class NnGender(Module):
    def __init__(self, feature, n_hidden=1024):
        super(NnGender, self).__init__(variable_scope="nn_gender")

        with tf.variable_scope(self.variable_scope):
            num_features = feature.get_shape()[1:].num_elements()
            fc_input = tf.reshape(feature, [-1, num_features])

            fc_hidden = new_fc_layer(layer_last=fc_input,
                                     num_inputs=num_features,
                                     num_outputs=n_hidden,
                                     use_relu=True)

            fc_output = new_fc_layer(layer_last=fc_hidden,
                                     num_inputs=n_hidden,
                                     num_outputs=2,
                                     use_relu=False)

            self.label_input = tf.placeholder(dtype=tf.int64)
            self.prediction = tf.argmax(tf.nn.softmax(fc_output), axis=1)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc_output,
                                                                                      labels=self.label_input))

        self._build_saver()


class NnClassification(Module):
    def __init__(self, feature, n_hidden=4096):
        super(NnClassification, self).__init__(variable_scope="nn_classification")

        with tf.variable_scope(self.variable_scope):
            # add noise
            # noise = tf.random_uniform(shape=tf.shape(feature), minval=-5.0, maxval=5, dtype=tf.float32)
            # noise = tf.random_normal(shape=tf.shape(feature), mean=0.0, stddev=10, dtype=tf.float32)
            # feature += noise
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
            self.prediction = tf.nn.sigmoid(fc_output)

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fc_output,
                                                                               labels=self.label_input))

        self._build_saver()


def pre_train(config):
    try:
        os.mkdir(os.path.join(config["save_root"], "adamantite"))
    except OSError:
        pass

    print("==> pre-train started at {0}.".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("---- CONFIG DUMP ----")
    print(json.dumps(config, indent=1))
    print("---- END ----")

    print("--> building models...")
    source_feature_module = Source()
    gender_module = NnGender(feature=source_feature_module.feature)
    image, label = TfReader(data_path=config["source_data"]["path"], size=(256, 256),
                            num_epochs=config["source_data"]["epoch"]) \
        .read(batch_size=config["source_data"]["batch_size"])
    global_step_op = tf.Variable(0, trainable=False, name="global_step")
    var_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, gender_module.variable_scope) + \
                   source_feature_module.trainable_list
    optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"]).minimize(loss=gender_module.loss,
                                                                                       global_step=global_step_op,
                                                                                       var_list=var_to_train,
                                                                                       colocate_gradients_with_ops=True)

    print("optimizer variables:", end='')
    for index, var in enumerate(var_to_train):
        if index % 2 == 0:
            print("\n", end='')
        print("  {0}".format(var), end='')
    print("\n", end='')

    print("--> starting session...")
    if config["checkpointing"]:
        checkpoint = os.path.join(config["save_root"], "adamantite")
    else:
        checkpoint = None
    hooks = [EndSavingHook(module_list=[source_feature_module, gender_module], save_path=config["save_root"])]
    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=checkpoint) as mon_sess:
        global_step = -1
        current_cost = -1
        try:
            while not mon_sess.should_stop():
                image_batch, label_batch = mon_sess.run([image, label])
                _, global_step, current_cost = mon_sess.run([optimizer, global_step_op, gender_module.loss],
                                                            feed_dict={
                                                                source_feature_module.image_input: image_batch,
                                                                gender_module.label_input: label_batch
                                                            })
                if global_step % config["report_rate"] == 0:
                    print("  * step ({0}) cost: {1:8}".format(global_step, current_cost))
        except tf.errors.OutOfRangeError as e:
            print("no more data: {0}".format(repr(e)))
        except KeyboardInterrupt as e:
            print("\ncanceled: {0}".format(repr(e)))

    with open(os.path.join(config["save_root"], "gan_vgg.log"), 'a') as log_file:
        message = "==> pre-train completed at {0} in {1} steps.".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                        global_step)
        log_file.write(message + "\n")
        print(message)

        log_file.write("---- CONFIG DUMP ----\n")
        json.dump(config, log_file, indent=1)
        log_file.write("\n---- END ----\n")

        message = "cost:  {0:8}".format(current_cost)
        log_file.write(message + "\n")
        print(message)

    os.remove(os.path.join(config["save_root"], "checkpoint"))


def adaption(config):
    try:
        os.mkdir(os.path.join(config["save_root"], "adamantite"))
    except OSError:
        pass

    print("==> adaption started at {0}.".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    print("---- CONFIG DUMP ----")
    print(json.dumps(config, indent=1))
    print("---- END ----")

    print("--> building models...")
    source_feature_module = Source()
    target_feature_module = Target()
    target_feature_module.override_saver_for_init_by(source_model=source_feature_module)
    discriminator_module = NnClassification(feature=target_feature_module.feature)
    source_image, _ = TfReader(data_path=config["source_data"]["path"], size=(256, 256),
                               num_epochs=config["source_data"]["epoch"]) \
        .read(batch_size=config["source_data"]["batch_size"])
    target_image, _ = TfReader(data_path=config["target_data"]["path"], size=(256, 256),
                               num_epochs=config["target_data"]["epoch"]) \
        .read(batch_size=config["target_data"]["batch_size"])
    global_step_op = tf.Variable(0, trainable=False, name="global_step")
    var_d = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,
                              scope=discriminator_module.variable_scope)
    optimizer_d = tf.train.AdamOptimizer(learning_rate=config["alternative_learning_rate"]) \
        .minimize(loss=discriminator_module.loss,
                  global_step=global_step_op,
                  var_list=var_d,
                  colocate_gradients_with_ops=True)
    optimizer_m = tf.train.AdamOptimizer(learning_rate=config["learning_rate"]) \
        .minimize(loss=discriminator_module.loss,
                  global_step=global_step_op,
                  var_list=target_feature_module.trainable_list,
                  colocate_gradients_with_ops=True)
    accuracy = tf.reduce_mean(1 - tf.abs(discriminator_module.prediction - discriminator_module.label_input))

    print("optimizer_d variables:", end='')
    for index, var in enumerate(var_d):
        if index % 2 == 0:
            print("\n", end='')
        print("  {0}".format(var), end='')
    print("\n", end='')
    print("optimizer_m variables:", end='')
    for index, var in enumerate(target_feature_module.trainable_list):
        if index % 2 == 0:
            print("\n", end='')
        print("  {0}".format(var), end='')
    print("\n", end='')

    print("--> starting session...")
    if config["checkpointing"]:
        checkpoint = os.path.join(config["save_root"], "adamantite")
    else:
        checkpoint = None

    # no need to init when checkpoint exist
    if checkpoint and os.path.exists(os.path.join(checkpoint, "checkpoint")):
        hooks = [
            EndSavingHook(module_list=[target_feature_module, discriminator_module], save_path=config["save_root"])]
    else:
        hooks = [
            EndSavingHook(module_list=[target_feature_module, discriminator_module], save_path=config["save_root"]),
            LoadInitialValueHook(module_list=[source_feature_module, target_feature_module],
                                 save_path=config["save_root"])]
    with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=checkpoint) as mon_sess:
        global_step = -1
        cost_d = -1
        cost_m = -1
        accuracy_d = -1
        step_for_report = 0
        step_for_header = 0
        try:
            while not mon_sess.should_stop():
                # read image and compute the feature
                source_image_batch, target_image_batch = mon_sess.run([source_image, target_image])
                source_feature_batch, target_feature_batch = mon_sess.run(
                    [source_feature_module.feature, target_feature_module.feature],
                    feed_dict={
                        source_feature_module.image_input: source_image_batch,
                        target_feature_module.image_input: target_image_batch
                    })

                # discriminator
                if "discriminator" in config["adaption_mode"]:
                    import random

                    combined = list(zip(
                        np.concatenate((source_feature_batch, target_feature_batch), axis=0),
                        [1] * config["source_data"]["batch_size"] + [0] * config["target_data"]["batch_size"]
                    ))
                    random.shuffle(combined)

                    features, labels = zip(*combined)

                    _, global_step, cost_d = mon_sess.run(
                        [optimizer_d, global_step_op, discriminator_module.loss],
                        feed_dict={
                            target_feature_module.feature: features,
                            discriminator_module.label_input: labels
                        }
                    )

                # generator
                if "generator" in config["adaption_mode"]:
                    epoch_multiplier_d = 1
                    cost_m = 0
                    accumulated_cost = 0

                    for _ in range(epoch_multiplier_d):
                        _, global_step, current_cost = mon_sess.run(
                            [optimizer_m, global_step_op, discriminator_module.loss],
                            feed_dict={
                                target_feature_module.image_input: target_image_batch,
                                discriminator_module.label_input: [1] * config["target_data"]["batch_size"]
                            }
                        )
                        accumulated_cost += current_cost

                    cost_m = accumulated_cost / epoch_multiplier_d

                # determine accuracy
                features = np.concatenate((source_feature_batch, target_feature_batch), axis=0)
                labels = [1] * config["source_data"]["batch_size"] + [0] * config["target_data"]["batch_size"]
                accuracy_d = mon_sess.run(accuracy,
                                          feed_dict={
                                              target_feature_module.feature: features,
                                              discriminator_module.label_input: labels
                                          })

                # report progress
                if global_step >= step_for_header:
                    step_for_header = global_step + config["report_rate"] * 25
                    print(
                        "\n         step      cost_m      cost_d   accuracy\n            - - - - - - - - - - - - - - -")
                if global_step >= step_for_report:
                    step_for_report = global_step + config["report_rate"]
                    print("  *  {0:8}  {1:10.4f}  {2:10.4f}  {3:8.4f}%".format(global_step, cost_m, cost_d,
                                                                               accuracy_d * 100))

        except tf.errors.OutOfRangeError as e:
            print("no more data: {0}".format(repr(e)))
        except KeyboardInterrupt as e:
            print("\ncanceled: {0}".format(repr(e)))

    with open(os.path.join(config["save_root"], "gan_vgg.log"), 'a') as log_file:
        message = "==> adaption completed at {0} in {1} steps.".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                       global_step)
        log_file.write(message + "\n")
        print(message)

        log_file.write("---- CONFIG DUMP ----\n")
        json.dump(config, log_file, indent=1)
        log_file.write("\n---- END ----\n")

        message = "cost_d:  {0:8}, cost_m:  {1:8}, accuracy: {2:8}".format(cost_d, cost_m, accuracy_d)
        log_file.write(message + "\n")
        print(message)

    os.remove(os.path.join(config["save_root"], "checkpoint"))


def test(config, alex):
    from io import BytesIO
    from PIL import Image
    from zipfile import ZipFile
    import time

    try:
        os.mkdir(os.path.join(config["save_root"], "crimtane"))
    except OSError:
        pass

    print("==> test started at {0} for {1}.".format(datetime.now().strftime("%Y-%m-%d %H:%M"), alex))
    print("---- CONFIG DUMP ----")
    print(json.dumps(config, indent=1))
    print("---- END ----")

    print("--> building models...")
    feature_module = alex()
    gender_module = NnGender(feature=feature_module.feature)
    image, label = TfReader(data_path=config["test_data"]["path"], size=(256, 256),
                            num_epochs=config["test_data"]["epoch"]) \
        .read(batch_size=config["test_data"]["batch_size"])
    statistics = ConfusionMatrix(2)

    # MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    print("--> starting session...")
    if config["keep_zip"] > 0:
        zip_file = ZipFile(
            os.path.join(config["save_root"], "crimtane", "test_result_{0}.zip".format(int(time.time()))),
            'w')
    else:
        zip_file = DummyFile()
    hooks = [LoadInitialValueHook(module_list=[feature_module, gender_module], save_path=config["save_root"])]
    with tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        accumulated_accuracy = 0
        test_step = 0
        file_count = 0
        try:
            while not mon_sess.should_stop():
                test_step += 1
                image_batch, label_batch = mon_sess.run([image, label])
                prediction_value = mon_sess.run(gender_module.prediction, feed_dict={
                    feature_module.image_input: image_batch
                })
                accuracy_value = 1 - np.mean(np.abs(np.transpose(prediction_value) - label_batch))
                accumulated_accuracy += accuracy_value
                statistics.update(predictions=prediction_value, truth=label_batch)
                if test_step % config["report_rate"] == 0:
                    print("  * step ({0}) accuracy: {1:8}".format(test_step, accumulated_accuracy / test_step))
                if config["keep_zip"] > 0:
                    for image_bytes, label_float, prediction_float in zip(image_batch, label_batch, prediction_value):
                        prediction_string = str(int(prediction_float + (0.5 if prediction_float > 0 else -0.5)))
                        label_string = str(int(label_float))
                        jpeg_bytes = BytesIO()
                        Image.fromarray(np.uint8(image_bytes), "RGB").save(jpeg_bytes, format="JPEG")
                        zip_file.writestr("{0}/{1}_{2:05}.jpg".format(label_string, prediction_string, file_count),
                                          jpeg_bytes.getvalue())
                        file_count += 1
        except tf.errors.OutOfRangeError as e:
            print("no more data: {0}".format(repr(e)))
        except KeyboardInterrupt as e:
            print("\ncanceled: {0}".format(repr(e)))

    zip_file.close()
    with open(os.path.join(config["save_root"], "gan_vgg.log"), 'a') as log_file:
        message = "==> test for {2} completed at {0} in {1} steps.".format(datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                                           test_step, alex)
        log_file.write(message + "\n")
        print(message)

        log_file.write("---- CONFIG DUMP ----\n")
        json.dump(config, log_file, indent=1)
        log_file.write("\n---- END ----\n")

        if config["keep_zip"] > 0:
            keep_zip = config["keep_zip"]
            import glob
            zips = glob.glob(os.path.join(config["save_root"], "crimtane", "test_result_*.zip"))
            for a_zip in sorted(zips, reverse=True):
                if keep_zip > 0:
                    keep_zip -= 1
                    continue
                os.remove(a_zip)

            message = "result wrote to {0}.".format(zip_file.filename)
            log_file.write(message + "\n")
            print(message)

        message = "overall result:  {0:8}".format(accumulated_accuracy / test_step)
        log_file.write(message + "\n")
        print(message)

        message = statistics.generate_result()
        log_file.write(message + "\n")
        print(message)


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="gan with alex.")
    parser.add_argument("action", choices=["pretrain", "adaption", "test", "pipeline"],
                        help="action to perform")
    parser.add_argument("-c", "--config", default="gan_vgg.config",
                        help="path to config file")

    parser.add_argument("--adaption-mode", default=None,
                        help="decide whether adaption trains discriminator, generator or both")

    parser.add_argument("--test-using-source", action="store_true",
                        help="test source feature performance on target")
    parser.add_argument("--cuda-devices", default="1",
                        help="cuda device to use")

    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    if args.adaption_mode:
        config["adaption_mode"] = args.adaption_mode

    if args.action == "pretrain":
        pre_train(config)
    elif args.action == "adaption":
        adaption(config)
    elif args.action == "test":
        if args.test_using_source:
            test(config, alex=Source)
        else:
            test(config, alex=Target)
    elif args.action == "pipeline":
        pass


if __name__ == "__main__":
    _main()
