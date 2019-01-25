import os
import re
import numpy as np
import tensorflow as tf

from utils import pp
from model import GAID
from evaluate import get_auc
from preparing_data import preparing_mias_data


flags = tf.app.flags
flags.DEFINE_integer("epoch",100, "Epoch to train [100]")
flags.DEFINE_float("g_learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("d_learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("patch_size", 256, "The size of patch images [256]")
flags.DEFINE_integer("input_height", 256, "The size of image to use (will be center cropped). [256]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 256, "The size of the output images to produce [256]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")

flags.DEFINE_float("w_rec", 4, "alpha to weight reconstruction loss [4]")
flags.DEFINE_float("w_real", 1, "alpha to weight sigmoid cross entropy loss [1]")

flags.DEFINE_boolean("preparing_data", False, "True for preparing patches and create train and test datasets[False]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("dataset", "mias", "The name of dataset [mias]")
flags.DEFINE_string("source_dir", "source", "The name of source dataset (.pmg file for mias)")
flags.DEFINE_string("train_dir", "train", "The name of train data dir.")
flags.DEFINE_string("test_dir", "test", "The name of test data dir.")
flags.DEFINE_string("test_full_image_dir", "full image", "The name of test full image dir.")
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_boolean("test_with_patch", True, "True for testing with image patches, False for testing with full images [True]")

flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("test_result_dir", "test_result", "Directory name to save the anomaly test result [test_result]")

flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("test", False, "True for test in tests directory, not anomaly test [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

FLAGS = flags.FLAGS

def test_with_patch_image(gaid):

    gaid.get_test_data()
    img_name, y_true, res_loss, dis_loss, y_score = gaid.test(FLAGS, True)

    print('[*] testing ...')

    roc_auc = get_auc(y_true, y_score, True)

    print("ROC curve area: %.4f" % roc_auc)

    for idx in range(np.shape(y_true)[0]):
        print("image name: [%s] anomaly score: %.2f, actual label: %.d, generator loss: %.2f, discriminator loss: %.2f" \
              % (str(re.split('/|[.]|\\\\', img_name[idx])[-2]), y_score[idx],
                 y_true[idx], res_loss[idx], dis_loss[idx]))

    test_res = list(zip(y_score,y_true))
    np.savetxt("score"+str(FLAGS.patch_size)+".csv", test_res,header="score,label", delimiter=",")

def test_with_full_image(gaid, FLAGS):
    print('[*] testing ...')
    gaid.test_with_full_image(FLAGS)

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    if not os.path.exists(FLAGS.test_result_dir):
        os.makedirs(FLAGS.test_result_dir)
    if not os.path.exists(os.path.join(FLAGS.test_result_dir, "test_generated")):
        os.makedirs(os.path.join(FLAGS.test_result_dir, "test_generated"))
    if not os.path.exists(os.path.join(FLAGS.test_result_dir, "test_heatmap")):
        os.makedirs(os.path.join(FLAGS.test_result_dir, "test_heatmap"))

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        gaid = GAID(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            data_dir=FLAGS.data_dir,
            train_dir=FLAGS.train_dir,
            test_dir=FLAGS.test_dir,
            patch_size=FLAGS.patch_size,
            c_dim=1,
            w_rec=FLAGS.w_rec,
            w_real=FLAGS.w_real)


        if FLAGS.train:
            gaid.train(FLAGS)
        else:
            if not gaid.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test model")

        if FLAGS.test:
            gaid.anomaly_detector()
            if FLAGS.test_with_patch:
                test_with_patch_image(gaid)
            else:
                test_with_full_image(gaid, FLAGS)

if __name__ == '__main__':

    if FLAGS.dataset == 'mias' and FLAGS.preparing_data:
        preparing_mias_data(
        root_dir=os.path.join(FLAGS.data_dir, FLAGS.dataset),
        source_dir=FLAGS.source_dir,
        train_dir=FLAGS.train_dir,
        test_dir=FLAGS.test_dir,
        patchsize=FLAGS.patch_size)

    tf.app.run()
