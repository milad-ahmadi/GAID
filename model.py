"""
Some codes from https://github.com/Newmu/dcgan_code and https://github.com/LeeDoYup/AnoGAN
"""

import os
import re
import time
import random
import logging
from glob import glob
import seaborn as sns
from six.moves import xrange
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ops import *
from utils import *
from evaluate import get_auc
from preparing_data import extract_patches


def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class GAID(object):
    def __init__(self, sess, input_height=256, input_width=256, crop=True,
         batch_size=64, sample_num = 64, output_height=256, output_width=256, patch_size =256,
         gf_dim=64, df_dim=64,c_dim=1, dataset_name='mias', train_dir='train', test_dir='test',
         input_fname_pattern='*.png', checkpoint_dir=None, data_dir='./data', w_rec=4, w_real=1):

        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          gf_dim: (optional) Dimension of generator (Reconstructor) filters in first conv layer. [64]
          df_dim: (optional) Dimension of discriminator (Representation matching) filters in first conv layer. [64]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.patch_size = patch_size
        self.gf_dim = gf_dim
        self.df_dim = df_dim


        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')


        self.g_bnc1 = batch_norm(name='g_bnc1')
        self.g_bnc2 = batch_norm(name='g_bnc2')
        self.g_bnc3 = batch_norm(name='g_bnc3')
        self.g_bnd1 = batch_norm(name='g_bnd1')
        self.g_bnd2 = batch_norm(name='g_bnd2')
        self.g_bnd3 = batch_norm(name='g_bnd3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.test_dir = test_dir

        self.data = glob(os.path.join(self.data_dir, self.dataset_name, self.train_dir, self.input_fname_pattern))
        np.random.shuffle(self.data)

        if c_dim >= 3: #check if image is a non-grayscale image by checking channel number
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.w_rec = w_rec
        self.w_real = w_real

        self.build_model()

    def build_model(self):
        print(" [*] Build model ...")
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.G = self.generator(inputs, isTrain=True, reuse=False)

        self.D, self.D_logits = self.discriminator(inputs, reuse=False)

        self.sampler = self.generator(inputs, isTrain=False, reuse=True)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # histogram Summary
        self.g_sum = histogram_summary("g", self.G)
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)

        # Loss Fun.
        self.g_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.g_loss_rec = tf.reduce_mean(tf.abs(tf.subtract(inputs , self.G)))
        self.g_loss =  self.w_real * self.g_loss_real + self.w_rec * self.g_loss_rec

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        # Scalar Summary
        self.g_loss_real_sum = scalar_summary("g_loss_real", self.g_loss_real)
        self.g_loss_rec_sum = scalar_summary("g_loss_rec", self.g_loss_rec)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()

        self.anomaly_detector()
        self.get_test_data()

    def train(self, config):

        print(" [*] train model ...")

        # output log
        if os.path.exists("output_log.txt"):
            os.remove("output_log.txt")
        logging.basicConfig(filename='output_log.txt', level=logging.INFO)

        # Optimizer
        d_optim = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta) \
            .minimize(self.g_loss, var_list=self.g_vars)

        # Initialize global var.
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # Merge summary
        self.g_sum = merge_summary([self.d__sum, self.g_sum,self.g_loss_real_sum, self.g_loss_rec_sum, self.g_loss_sum, self.d_loss_fake_sum])
        self.d_sum = merge_summary([self.d_sum, self.d_loss_sum, self.d_loss_real_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        # Load data sample
        sample_files = self.data[0:self.sample_num]
        sample = [get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
        if (self.grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        # Load checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 1
        start_run_time = time.time()
        total_batch_time = 0
        best_auc = 0.0
        sample_save = True

        for epoch in xrange(config.epoch):

            self.data = glob(os.path.join(config.data_dir, config.dataset, config.train_dir, self.input_fname_pattern))

            np.random.shuffle(self.data)

            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):

                start_batch_time = time.time()

                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum ],feed_dict={self.inputs: batch_images})

                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim,self.g_sum], feed_dict={self.inputs: batch_images})

                self.writer.add_summary(summary_str, counter)

                errD = self.d_loss.eval({self.inputs: batch_images})

                errG = self.g_loss.eval({self.inputs: batch_images})

                counter += 1
                end_batch_time = time.time()
                time_batch =  (end_batch_time - start_batch_time)*1000
                total_batch_time += time_batch

                hours, rem = divmod(end_batch_time - start_run_time, 3600)
                minutes, seconds = divmod(rem, 60)

                print(
                    "Epoch: [%2d/%2d] [%4d/%4d] time: %02d:%02d:%02d , G (Reconstructor) loss: %.8f, "
                    "D (Representation matching) loss: %.8f , Avg Run Time (ms/batch): %.8f ,(it/s): %.8f" \
                    % (epoch+1, config.epoch, idx+1, batch_idxs, int(hours), int(minutes),
                       seconds, errD, errG, total_batch_time/counter,counter/(total_batch_time/1000)))

                if np.mod(counter, batch_idxs*2)== 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                                                feed_dict={self.inputs: sample_inputs})

                        save_images(samples, image_manifold_size(samples.shape[0]),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        if sample_save:
                            sample_save = False
                            save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]),
                                        './{}/train_sample_inputs_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] D (Representation matching) loss: %.8f, G (Reconstructor) loss: %.8f" % (d_loss, g_loss))

                    except:
                        print("one pic error!...")

            # test
            _,y_true, g_loss, d_loss, y_score = self.test(config)

            roc_auc = get_auc(y_true, y_score)

            if best_auc < roc_auc:
                best_auc = roc_auc
                self.save(config.checkpoint_dir, epoch)

            logging.info("Epoch: [%2d/%2d] , AUC: %.8f, Best AUC: %.8f,  Avg run time: %.8f" % (epoch+1, config.epoch,
                         roc_auc, best_auc,  total_batch_time/counter))
            print("Epoch: [%2d/%2d], AUC: %.8f, Best AUC: %.8f,  Avg run time: %.8f" % (epoch+1, config.epoch,
                        roc_auc, best_auc,  total_batch_time/counter))
        

    def generator(self, x, isTrain=True, reuse=False, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            conv1 = conv2d(x, self.gf_dim, name='g_1_conv2d')

            conv2 = self.g_bnc1(conv2d(lrelu(conv1), self.gf_dim * 2, name='g_2_conv2d'))

            conv3 = self.g_bnc2(conv2d(lrelu(conv2), self.gf_dim * 4, name='g_3_conv2d'))

            conv4 = self.g_bnc3(conv2d(lrelu(conv3), self.gf_dim * 8, name='g_4_conv2d'))

            deconv1 = self.g_bnd1(deconv2d(tf.nn.relu(conv4), [batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_1_deconv2d',
                                                        with_w=False))
            deconv1 = tf.concat([deconv1, conv3], 3)

            deconv2 = self.g_bnd2(deconv2d(tf.nn.relu(deconv1), [batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_2_deconv2d',
                                                        with_w=False))
            deconv2 = tf.concat([deconv2, conv2], 3)

            deconv3 = self.g_bnd3(deconv2d(tf.nn.relu(deconv2), [batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_3_deconv2d',
                                                        with_w=False))
            deconv3 = tf.concat([deconv3, conv1], 3)

            deconv4 = deconv2d(tf.nn.relu(deconv3), [batch_size, s_h, s_w, self.c_dim], name='g_4_deconv2d', with_w=False)

            return tf.nn.tanh(deconv4)

    def discriminator(self, image, reuse=False, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            conv2 = lrelu(self.d_bn1(conv2d(conv1, self.df_dim * 2, name='d_h1_conv')))
            conv3 = lrelu(self.d_bn2(conv2d(conv2, self.df_dim * 4, name='d_h2_conv')))
            conv4 = lrelu(self.d_bn3(conv2d(conv3, self.df_dim * 8, name='d_h3_conv')))
            lin = linear(tf.reshape(conv4, [batch_size, -1]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(lin), lin

    def feature_match_layer(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            conv1 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            conv2 = lrelu(self.d_bn1(conv2d(conv1, self.df_dim * 2, name='d_h1_conv')))
            conv3 = lrelu(self.d_bn2(conv2d(conv2, self.df_dim * 4, name='d_h2_conv')))
            conv4 = lrelu(self.d_bn3(conv2d(conv3, self.df_dim * 8, name='d_h3_conv')))
            return conv4

    def anomaly_detector(self, ano_para=0.4, feature_match_layer = True):
        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:  # for test
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.test_inputs = tf.placeholder(tf.float32, [1] + image_dims, name='test_images')
        test_inputs = self.test_inputs

        self.ano_G = self.generator(test_inputs, isTrain=False, reuse=True, batch_size=1)

        # Anomaly Loss
        if feature_match_layer:
            dis_f_generated = self.feature_match_layer(self.ano_G, reuse=True)

            dis_f_input = self.feature_match_layer(test_inputs, reuse=True)

            self.dis_loss = tf.reduce_mean(tf.pow(tf.subtract(dis_f_generated, dis_f_input),2))
        else:
            test_D, test_D_logits_ = self.discriminator(self.ano_G, reuse=True, batch_size=1)

            self.dis_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(test_D_logits_, tf.ones_like(test_D)))

        self.rec_loss = tf.reduce_mean(tf.pow(tf.subtract(test_inputs, self.ano_G),2))

        self.anomaly_score = (1. - ano_para) * self.rec_loss + ano_para * self.dis_loss

    def get_test_data(self):

        normal_data_names = glob(
            os.path.join(self.data_dir, self.dataset_name, self.test_dir, 'normal' , self.input_fname_pattern))
        normal_data_label = np.zeros((np.shape(normal_data_names)[0]))

        abnormal_data_names = glob(
            os.path.join(self.data_dir, self.dataset_name, self.test_dir, 'abnormal', self.input_fname_pattern))
        abnormal_data_label = np.ones((np.shape(abnormal_data_names)[0]))

        self.test_data_names = normal_data_names + abnormal_data_names
        self.test_data_label = np.concatenate((normal_data_label, abnormal_data_label),axis=0)
        zip_names_label = list(zip(self.test_data_names, self.test_data_label))
        random.shuffle(zip_names_label)
        self.test_data_names, self.test_data_label = zip(*zip_names_label)
        self.test_data_label = np.array(self.test_data_label).astype(int)
        batch = [
            get_image(name, input_height=self.input_height, input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width, crop=self.crop, grayscale=self.grayscale) for name in
            self.test_data_names]

        if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
            batch_images = np.array(batch).astype(np.float32)

        self.test_data = batch_images

        print(" [*] test data for anomaly detection is loaded")

    def test(self,config, save_generated=False):

        num_test_image = np.shape(self.test_data)[0]

        test_image_ano_score = []
        test_image_rec_loss = []
        test_image_dis_loss = []
        for idx in range(num_test_image):
            input_image = np.expand_dims(self.test_data[idx], axis=0)

            generated_image, rec_loss, dis_loss, anomaly_score = self.sess.run(
                [self.ano_G, self.rec_loss, self.dis_loss, self.anomaly_score],
                feed_dict={self.test_inputs: input_image})
            generated_image = np.squeeze(generated_image, axis=0)
            generated_image = np.squeeze(generated_image, axis=2)

            input_image = np.squeeze(input_image, axis=0)
            input_image = np.squeeze(input_image, axis=2)

            test_image_ano_score.append(anomaly_score)
            test_image_rec_loss.append(rec_loss)
            test_image_dis_loss.append(dis_loss)

            inverse_transform(generated_image)
            generated_image = np.concatenate((generated_image, input_image), axis=1)

            if save_generated:
                plt.imsave(os.path.join(config.test_result_dir,"test_generated",
                            str(re.split('/|[.]|\\\\',self.test_data_names[idx])[-2]) + '.png')
                           , inverse_transform(generated_image), cmap=cm.gray)

        test_image_ano_score = np.array(test_image_ano_score)

        normalized_test_image_ano_score = (test_image_ano_score - min(test_image_ano_score)) \
                                          / (max(test_image_ano_score) - min(test_image_ano_score))
        normalized_test_image_dis_loss = (test_image_dis_loss - min(test_image_dis_loss)) \
                                         / (max(test_image_dis_loss) - min(test_image_dis_loss))
        normalized_test_image_rec_loss = (test_image_rec_loss - min(test_image_rec_loss)) \
                                         / (max(test_image_rec_loss) - min(test_image_rec_loss))

        return self.test_data_names,self.test_data_label, normalized_test_image_rec_loss, normalized_test_image_dis_loss,\
               normalized_test_image_ano_score

    def test_with_full_image(self, config):
        self.full_images_names = glob(
            os.path.join(self.data_dir, self.dataset_name, self.test_dir, config.test_full_image_dir, self.input_fname_pattern))
        self.full_images = [imread(name, grayscale=self.grayscale) for name in self.full_images_names]
        num_images = np.shape(self.full_images_names)[0]
        patch_size = [config.patch_size, config.patch_size]
        sns.set()
        for idx in range(num_images):

            print("Testing full image "+str(idx)+" --image name: "+str(re.split('/|[.]|\\\\', self.full_images_names[idx])[-2]))
            patches, regions = extract_patches(self.full_images[idx], patch_size, overlap_allowed=0.2, cropvalue=None,
                                               crop_fraction_allowed=1)

            num_patches = np.shape(patches)[0]
            if self.grayscale:
                patches = np.array(patches).astype(np.float32)[:, :, :, None]
            else:
                patches = np.array(patches).astype(np.float32)
            active = np.ones((np.shape(self.full_images[idx])[0], np.shape(self.full_images[idx])[1]))
            image_scores = np.zeros((np.shape(self.full_images[idx])[0], np.shape(self.full_images[idx])[1]))
            for jdx in range(num_patches):
                active[regions[jdx]] += 1

            ano_scores = []
            for jdx in range(num_patches):
                input_image = np.expand_dims(patches[jdx], axis=0)
                generated_image, rec_loss, dis_loss, anomaly_score = self.sess.run(
                    [self.ano_G, self.rec_loss, self.dis_loss, self.anomaly_score],
                    feed_dict={self.test_inputs: input_image})
                ano_scores.append(anomaly_score)

            norm_ano_scores = (ano_scores - min(ano_scores)) \
                              / (max(ano_scores) - min(ano_scores))
            for jdx in range(num_patches):
                image_scores[regions[jdx]] += norm_ano_scores[jdx] / active[regions[jdx]]

            ax = sns.heatmap(image_scores, vmin=0, vmax=1, xticklabels=False,
                             yticklabels=False, cbar= (idx == 0), square=True)
            figure = ax.get_figure()
            figure.savefig(os.path.join(config.test_result_dir, "test_heatmap",
                                        str(re.split('/|[.]|\\\\', self.full_images_names[idx])[
                                                -2]) + '-' + 'heatmap' + '.png'), dpi=400)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "GAID.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
