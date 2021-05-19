""""
Copyright 2021 the Norwegian Computing Center

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
"""

from __future__ import division, print_function

try:
    from StringIO import StringIO
    from BytesIO import BytesIO
except ImportError:
    from io import StringIO, BytesIO

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import io
import os
import sys
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from shutil import copyfile
from distutils.dir_util import mkpath
from tensorboard.plugins.pr_curve import summary as pr_summary
#To do confusion matrix
import seaborn as sn
import pandas as pd
#To get pip freeze
try:
    from pip._internal.operations import freeze
except ImportError:
    from pip.operations import freeze

tf.disable_v2_behavior()

class TensorboardLogger():
    "A tensorboard connector"

    #Placeholders for tensorboard summaries that cannot be written without tensorboard
    placeholder_string = tf.placeholder('string')
    placeholder_pr_curve_labels =  tf.placeholder('bool',shape=[None,None])
    placeholder_pr_curve_predictions = tf.placeholder('float32',shape=[None,None])

    @staticmethod
    def generate_string(objects):
        """ Collect class names from objects and make string """
        strings = [type(obj).__name__ if type(obj) is not str else obj for obj in objects ]
        return ' - '.join(strings)

    def __init__(self,
                 name='',
                 verbose=1,
                 log_dir='log/',
                 copy_source = True):

        self.verbose = verbose

        #Make directory
        if type(name) in [list, tuple]:
            name = TensorboardLogger.generate_string(name)

        self.log_dir = log_dir + '/' + name + ' - ' + datetime.datetime.now().strftime("%H.%M.%S, %B %d, %Y") + '/'
        self.log_dir = self.log_dir.replace('//', '/')
        os.makedirs(self.log_dir)

        #Make writer
        self.writer = tf.summary.FileWriter(self.log_dir)

        self.timeseries = {}
        self.img_pre_processing = None

        self.summaries = {}

        if verbose: print('Starting TensorboardLogger -', self.log_dir)

        # Copy python code
        if copy_source:
            if verbose: print(' - Copying source for backup')
            os.makedirs(self.log_dir + 'code')

            #Save pip freeze
            with open(self.log_dir+'REQUIREMENTS.txt','w') as f:
                f.writelines('\n'.join(freeze.freeze()))

            #Get folders in which to search
            ignore_patterns = ['lib/python','/.pycharm_helpers/']
            paths = list(set([p for p in sys.path if not any([ignore in p for ignore in ignore_patterns])]))
            folders_2_ignore = ['/log', 'lib/python']

            for path in paths:
                for root, dirs, files in os.walk(path):
                    base_folder = path.split('/')[-1]

                    if '__ignore__.py' in files or any([p2i in root for p2i in folders_2_ignore]):
                        folders_2_ignore.append(root)
                        continue

                    for file in files:
                        if file.endswith(".py"):
                            old = os.path.join(root, file)
                            new = self.log_dir + 'code/' + base_folder + old.split(base_folder)[-1]
                            try:
                                mkpath('/'.join(new.split('/')[0:-1]) + '/')
                            except:
                                None
                            try:
                                copyfile(old, new)
                            except:
                                return
            if verbose: print(' - Copying source for backup - FINISHED')
            if verbose: print('')

        pass

    def log_image(self, tag, images,
                  step=0,
                  max_imgs = 12,
                  cm='gray',
                  preprocessing = None,
                  heatmap_overlay=None,
                  alpha_mode=True):

        #If input is numpy array
        if type(images) in [np.array, np.ndarray]:

            #Make list of images
            shape = images.shape

            # Single image: H x W
            if len(shape)==2:
                images = [images]

            # Single image 2D: H x W x 3
            elif len(shape) == 3:
                if not(shape[-1] == 3 or shape[-1]==1):
                    images = images[:,:,0]
                images = [images]

            # Batch 2D: B x C x H x W
            elif len(shape) == 4:
                images = np.moveaxis(images, 1, -1)
                images = list(images)

            # Batch 3D: B x C x H x W x D
            elif len(shape) == 5:
                new_images_ts = [          images[i, 0, :, :, shape[4] // 2]      for i in range(shape[0])]
                new_images_il = [ np.rot90(images[i, 0, :, shape[3] // 2, :], -1) for i in range(shape[0])]
                new_images_cl = [ np.rot90(images[i, 0, shape[2] // 2, :, :], -1) for i in range(shape[0])]

                self.log_image('timeslice_' + tag, new_images_ts, step,  max_imgs=max_imgs, cm=cm, preprocessing=preprocessing)
                self.log_image('inline_' + tag , new_images_il, step,  max_imgs=max_imgs, cm=cm, preprocessing=preprocessing)
                self.log_image('crossline_' + tag, new_images_cl, step,  max_imgs=max_imgs, cm=cm, preprocessing=preprocessing)
                return

        #Check input format
        assert type(images) in [list,tuple]

        #Select half images from start and half from the end
        if len(images) > max_imgs:
            tmp0 = images[:max_imgs // 2]
            tmp1 = images[-max_imgs // 2:]
            images = tmp0+tmp1

        im_summaries = []
        for nr, img in enumerate(images):

            #Heatmap on top of image
            if heatmap_overlay is not None:
                # Add color dimention to image
                if len(img.shape) == 2:
                    img = np.expand_dims(img, -1)

                # Make gray-scale
                if img.shape[2] == 1:
                    img = np.repeat(img, 3, 2)

                #Normalize
                img = img - np.min(img)
                img = img / np.max(img)

                # Convert heatmap to RGB
                cmap = plt.get_cmap('jet')
                rgba_heatmap = cmap(heatmap_overlay.astype('float'))[:, :, 0:3]  # Remove alpha channel

                # Blend
                if alpha_mode:
                    h_fraction = np.expand_dims(heatmap_overlay * 0.8, 2)
                else:
                    h_fraction = np.expand_dims(np.round(heatmap_overlay) * 0.8, 2)

                img = rgba_heatmap * h_fraction + img * (1 - h_fraction)

                img = (img * 255).astype(np.uint8)

            #Pre processing
            if preprocessing is not None:
                img = preprocessing(img)

            #Change ignore label
            if np.sum(-100<img[img<0]) == 0 and np.sum(np.abs(img.astype('int8') - img)) == 0:
                img[img == -100] = -1;

            #Normalize
            img = img.astype('float')
            img = img - np.min(img)
            img = img / np.max(img + 0.0001)

            #Grayscale ?
            if (cm == 'gray' or cm == 'grey') and ((len(img.shape)>2 and img.shape[2] == 1) or len(img.shape)==2 ):
                if len(img.shape)==2 :
                    img = np.expand_dims(img,2)
                img = np.repeat(img,3,2)

            # Write the image to a string
            s = BytesIO()
            plt.imsave(s, img.squeeze(), format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

            #Limit number of images to plot
            if nr == max_imgs-1:
                break

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_scalar(self, tag, value, step=0):
        def to_unicode(s):
            try:
                return unicode(s)
            except:
                return str(s)
        #Dump to text file
        with io.open(self.log_dir + tag + '.txt','a',encoding="utf-8") as f:
            f.write(to_unicode(str(step) + ' ' + str(value) + '\n'))

        #Write to summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_pr_curve(self, tag, pred_probability, labels, step=0, N = 10000):
        labels = labels.astype('int32')

        # If values are images, then image dimensions must be flattened with the batch dimension
        if len(pred_probability.shape) > 2:
            Nb = pred_probability.shape[0]
            Nc = pred_probability.shape[1]
            Ns = np.prod(pred_probability.shape[2:])
            pred_probability = np.reshape(pred_probability, [Nb, Nc, Ns])
            pred_probability = np.transpose(pred_probability, [0, 2, 1])
            pred_probability = np.reshape(pred_probability, [Nb * Ns, Nc])
            labels = labels.flatten()

        # Remove ignore-labels
        keep = np.logical_not(labels == -100)
        pred_probability = pred_probability[keep, :]
        labels = labels[keep]

        #Get size of array
        Nb = pred_probability.shape[0]
        Nc = pred_probability.shape[1]

        # Decimate
        if Nb > N:
            inds = np.linspace(0, Nb-1, N).astype('int64')
            labels = labels[inds]
            pred_probability = pred_probability[inds,:]
            Nb = pred_probability.shape[0]

        # Make one-hot bool array of true labels
        tmp = np.zeros([Nb, Nc])
        for i in range(Nb):
            if labels[i] >= 0:
                tmp[i, labels[i]] = 1
        labels = tmp.astype('bool')

        #Add summary to dict
        if tag not in self.summaries.keys():
            self.summaries[tag] = pr_summary.op(name=tag, labels=self.placeholder_pr_curve_labels, predictions=self.placeholder_pr_curve_predictions, num_thresholds=21)

        #Make summary
        merged_summary_op = tf.summary.merge([self.summaries[tag]])
        with self._tf_session() as sess:
            merged_summary = sess.run(merged_summary_op, feed_dict={self.placeholder_pr_curve_labels: labels, self.placeholder_pr_curve_predictions: pred_probability})

        #Write
        self.writer.add_summary(merged_summary, step)

    def log_scatter_plot(self, tag, pred, true, step=0, N = 10000):

        # If values are images, then image dimensions must be flattened with the batch dimension
        if len(pred.shape) > 2:
            Nb = pred.shape[0]
            Nc = pred.shape[1]
            Ns = np.prod(pred.shape[2:])
            pred = np.reshape(pred, [Nb, Nc, Ns])
            pred = np.transpose(pred, [0, 2, 1])
            pred = np.reshape(pred, [Nb * Ns, Nc])
            true = np.reshape(true, [Nb * Ns, Nc])

        #Remove ignore-labels
        keep = np.logical_not(np.any(true == -100, axis=1))
        pred = pred[keep, :]
        true = true[keep, :]

        #Get size of array
        Nb = pred.shape[0]
        Nc = pred.shape[1]

        # Decimate
        if Nb > N:
            inds = np.linspace(0, Nb-1, N).astype('int64')
            true = true[inds, :]
            pred = pred[inds, :]

        for i in range(Nc):
            plt.clf()
            plt.plot([np.min(true),np.min(true)],[np.max(true),np.max(true)],c="k")
            plt.grid()
            plt.scatter(true[:,i], pred[:,i], c="g", alpha=0.2, marker=r'.')
            plt.xlabel("True")
            plt.ylabel("Prediction")

            self.log_matplotlib_plt('scatter_' + tag + '_' + str(i), plt, step)
            plt.clf()

    def log_confusion_matrix(self, tag, pred_class, labels, step=0, N = 10000):
        classes = np.unique(labels)
        max_class = int(np.max(classes))
        label_names = [str(i) for i in range(max_class+1)]

        #Reshape to vector
        pred_class = pred_class.flatten()
        labels = labels.flatten()

        # Remove ignore-labels
        keep = np.logical_not(labels == -100)
        pred_class = pred_class[keep]
        labels = labels[keep]

        #Decimate
        if labels.shape[0] > N:
            inds = np.linspace(0,labels.shape[0]-1,N).astype('int64')
            labels = labels[inds]
            pred_class = pred_class[inds]

        #Remove unknowns
        knowns = labels >= 0
        labels = labels[knowns]
        pred_class = pred_class[knowns]

        #Get confusion
        from sklearn.metrics import confusion_matrix

        cf = confusion_matrix(labels, pred_class)

        #Make plt
        df_cm = pd.DataFrame(cf, index=label_names,columns=label_names)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)

        #Log
        self.log_matplotlib_plt('confusion_' + tag, plt, step, size=[7,10])

    def log_matplotlib_plt(self, tag, plt, step=0, size = [10,10]):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue(), height=size[0], width=size[1])

        # Create a Summary value
        sum = tf.Summary.Value(tag=tag, image=img_sum)

        # Create and write Summary
        self.writer.add_summary(tf.Summary(value=[sum]), step)

    def log_string(self, string, step=0):
        if self.verbose: print(string)

        #Make summary op
        if 'text' not in self.summaries.keys():
            self.summaries['text'] = tf.summary.text('text', self.placeholder_string)
        merged_summary_op = tf.summary.merge([self.summaries['text']])


        with self._tf_session() as sess:
            merged_summary = sess.run(merged_summary_op, feed_dict={self.placeholder_string: string})
        self.writer.add_summary(merged_summary, step)

    def log_histogram(self, tag, values, step=0, bins=1000, range =None ):

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values.flatten(), bins=bins, range = range)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_save_model(self, model_adapter, step = 0, best_acc = None):
        def save():
            model_adapter.save(self.log_dir + 'saved_model_' + str(step))
            print('Model saved')

        if best_acc is not None:
            if self.best_acc is None:
                self.best_acc = None
                save()
            else:
                if self.best_acc >= best_acc:
                    self.best_acc = best_acc
                    save()
                else:
                    print('Not saved')
        else:
            save()

    def _tf_session(self):
        session_conf = tf.ConfigProto(
            device_count={'CPU': 1, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        return tf.Session(config=session_conf)



