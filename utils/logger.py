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

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import datetime
import io
import os
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import torch
from shutil import copyfile
from distutils.dir_util import mkpath
from tensorboard.plugins.pr_curve import summary as pr_summary
from PIL import Image

# To get pip freeze
try:
    from pip._internal.operations import freeze
except ImportError:
    from pip.operations import freeze

class TensorboardLogger():
    "A tensorboard connector"

    #Placeholders for tensorboard summaries that cannot be written without tensorboard
    placeholder_string = tf.compat.v1.placeholder('string')
    placeholder_pr_curve_labels =  tf.compat.v1.placeholder('bool',shape=[None,None])
    placeholder_pr_curve_predictions = tf.compat.v1.placeholder('float32',shape=[None,None])

    @staticmethod
    def generate_string(objects):
        """ Collect class names from objects and make string """
        strings = [type(obj).__name__ if type(obj) is not str else obj for obj in objects ]
        return '_'.join(strings)

    def __init__(self,
                 name='',
                 log_dir='log/',
                 include_datetime=False,
                 verbose=1,
                 copy_source = False):

        self.include_datetime = include_datetime
        self.verbose = verbose
        self.current_lowest_score = np.inf
        self.current_highest_score = -np.inf

        #Make directory
        if type(name) in [list, tuple]:
            name = TensorboardLogger.generate_string(name)
        if self.include_datetime:
            self.log_dir = log_dir + '/' + name + '_' + datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + '/'
        else:
            self.log_dir = log_dir + '/' + name + '/'

        self.log_dir = self.log_dir.replace('//', '/')
        os.makedirs(self.log_dir, exist_ok=True)

        #Make writer
        self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

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
            paths = [os.getcwd()]
            #Todo: Should we use current working directory only, or all folders in python-path?
            #list(set([p for p in sys.path    if not any([ignore in p for ignore in ignore_patterns])]))


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

    def log_scalar(self, tag, value, step=0):
        def to_unicode(s):
            try:
                return unicode(s)
            except:
                return str(s)
        #Dump to text file
        with io.open(self.log_dir + tag + '.txt', 'a', encoding="utf-8") as f:
            f.write(to_unicode(str(step) + ' ' + str(value) + '\n'))

        #Write to summary
        summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_save_model(self, model, filename, step = 0, save_on_max=None, save_on_min=None):
        # Changed model_adapter to model
        def save():
            #if type(model_adapter) == nn.Module:
            torch.save(model.state_dict(),self.log_dir + filename + '_' + str(step) + '.pt')
            torch.save(model.state_dict(), self.log_dir + filename + '.pt')
            #else:
            #    model_adapter.save(self.log_dir + 'saved_model_' + str(step))
            #    model_adapter.save(self.log_dir + 'saved_model')
            print(filename + ' saved')
            return True

        if save_on_max is not None and save_on_min is not None:
            raise Warning('Can not save on both max and min - will save model regardless of min/max acc')
            return save()
        elif save_on_max is None and save_on_min is None:
            return save()
        elif save_on_max is not None and self.current_highest_score < save_on_max:
            self.current_highest_score = save_on_max
            return save()
        elif save_on_min is not None and self.current_lowest_score > save_on_min:
            self.current_lowest_score = save_on_min
            return save()
        else:
            print(filename + ' not saved')
            return False

    def log_plot(self, tag, figure, step=0):
        plot_buf = io.BytesIO()
        figure.savefig(plot_buf, format='png')
        plot_buf.seek(0)
        img = Image.open(plot_buf)
        img_ar = np.array(img)

        img_summary = tf.compat.v1.Summary.Image(encoded_image_string=plot_buf.getvalue(),
                                   height=img_ar.shape[0],
                                   width=img_ar.shape[1])

        summary = tf.compat.v1.Summary()
        summary.value.add(tag=tag, image=img_summary)
        self.writer.add_summary(summary, global_step=step)
        self.writer.flush()

    def _tf_session(self):
        session_conf = tf.compat.v1.ConfigProto(
            device_count={'CPU': 1, 'GPU': 0},
            allow_soft_placement=True,
            log_device_placement=False
        )
        return tf.compat.v1.Session(config=session_conf)