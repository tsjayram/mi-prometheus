#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
mnist.py: contains code for loading the `MNIST` dataset using ``torchvision``.
"""
__author__ = "Younes Bouhadjar & Vincent Marois"

import os
import torch
from torchvision import datasets, transforms

from miprometheus.utils.data_dict import DataDict
from miprometheus.problems.image_to_class.image_to_class_problem import ImageToClassProblem


class MNIST(ImageToClassProblem):
    """
    Classic MNIST classification problem.

    Please see reference here: http://yann.lecun.com/exdb/mnist/

    .. warning::

        The dataset is not originally split into a training set, validation set and test set; only\
        training and test set. It is recommended to use a validation set.

        ``torch.utils.data.SubsetRandomSampler`` is recommended.

    """

    def __init__(self, params_):
        """
        Initializes MNIST problem:

            - Calls ``problems.problem.ImageToClassProblem`` class constructor,
            - Sets following attributes using the provided ``params``:

                - ``self.data_folder`` (`string`) : Root directory of dataset where ``processed/training.pt``\
                    and  ``processed/test.pt`` will be saved,
                - ``self.use_train_data`` (`bool`, `optional`) : If True, creates dataset from ``training.pt``,\
                    otherwise from ``test.pt``
                - ``self.resize`` : (optional) resize the images to `[h, w]` if set,
                - ``self.defaut_values`` :

                    >>> self.default_values = {'num_classes': 10,
                    >>>            'num_channels': 1,
                    >>>            'width': self.width, # (DEFAULT: 28)
                    >>>            'height': self.height} # (DEFAULT: 28)

                - ``self.data_definitions`` :

                    >>> self.data_definitions = {'images': {'size': [-1, 1, self.height, self.width], 'type': [torch.Tensor]},
                    >>>                          'targets': {'size': [-1], 'type': [torch.Tensor]},
                    >>>                          'targets_label': {'size': [-1, 1], 'type': [list, str]}
                    >>>                         }

        .. warning::

            Resizing images might cause a significant slow down in batch generation.

        .. note::

            The following is set by default:

            >>> self.params.add_default_params({'data_folder': '~/data/mnist',
            >>>           'use_train_data': True})

        :param params_: Dictionary of parameters (read from configuration ``.yaml`` file).

        """

        # Call base class constructors.
        super(MNIST, self).__init__(params_, 'MNIST')

        # Set default parameters.
        self.params.add_default_params({'data_folder': '~/data/mnist',
                                        'use_train_data': True
                                        })

        # Get absolute path.
        data_folder = os.path.expanduser(self.params['data_folder'])

        # Retrieve parameters from the dictionary.
        self.use_train_data = self.params['use_train_data']

        # Add transformations depending on the resizing option.
        if 'resize' in self.params:
            # Check the desired size.
            if len(self.params['resize']) != 2:
                self.logger.error("'resize' field must contain 2 values: the desired height and width")
                exit(-1)

            # Output image dimensions.
            self.height = self.params['resize'][0]
            self.width = self.params['resize'][1]
            self.num_channels = 1

            # Up-scale and transform to tensors.
            transform = transforms.Compose([transforms.Resize((self.height, self.width)), transforms.ToTensor()])

            self.logger.warning('Upscaling the images to [{}, {}]. Slows down batch generation.'.format(
                self.width, self.height))

        else:
            # Default MNIST settings.
            self.width = 28
            self.height = 28
            self.num_channels = 1
            # Simply turn to tensor.
            transform = transforms.Compose([transforms.ToTensor()])

        # Define the default_values dict: holds parameters values that a model may need.
        self.default_values = {'num_classes': 10,
                               'num_channels': self.num_channels,
                               'width': self.width,
                               'height': self.height}

        self.data_definitions = {'images': {'size': [-1, self.num_channels, self.height, self.width], 'type': [torch.Tensor]},
                                 'targets': {'size': [-1], 'type': [torch.Tensor]},
                                 'targets_label': {'size': [-1, 1], 'type': [list, str]}
                                 }

        # load the dataset
        self.dataset = datasets.MNIST(root=data_folder, train=self.use_train_data, download=True,
                                      transform=transform)

        # Set length.
        self.length = len(self.dataset)

        # Class names.
        self.labels = 'Zero One Two Three Four Five Six Seven Eight Nine'.split(' ')

    def __getitem__(self, index):
        """
        Getter method to access the dataset and return a sample.

        :param index: index of the sample to return.
        :type index: int

        :return: ``DataDict({'images','targets', 'targets_label'})``, with:

            - images: Image, resized if ``self.resize`` is set,
            - targets: Index of the target class
            - targets_label: Label of the target class (cf ``self.labels``)
        """
        # Get image and target.
        img, target = self.dataset.__getitem__(index)
  
        # Digit label.
        label = self.labels[target.data]

        # Return data_dict.
        data_dict = self.create_data_dict()
        data_dict['images'] = img
        data_dict['targets'] = target
        data_dict['targets_label'] = label
        return data_dict

    def collate_fn(self, batch):
        """
        Combines a list of ``DataDict`` (retrieved with ``__getitem__`` ) into a batch.

        .. note::

            This function wraps a call to ``default_collate`` and simply returns the batch as a ``DataDict``\
            instead of a dict.
            Multi-processing is supported as the data sources are small enough to be kept in memory\
            (`training.pt` has a size of 47.5 MB).

        :param batch: list of individual ``DataDict`` samples to combine.

        :return: ``DataDict({'images','targets', 'targets_label'})`` containing the batch.

        """

        return DataDict({key: value for key, value in zip(self.data_definitions.keys(),
                                                          super(MNIST, self).collate_fn(batch).values())})


if __name__ == "__main__":
    """ Tests sequence generator - generates and displays a random sample"""

    # Load parameters.
    from miprometheus.utils.param_interface import ParamInterface
    params = ParamInterface()  # using the default values

    # Test different options.
    params.add_config_params({'data_folder': '~/data/mnist',
                                    'use_train_data': True,
                                    #'resize': [32, 32],
                                    })

    batch_size = 64

    # Create problem.
    mnist = MNIST(params)

    # get a sample
    sample = mnist[10]
    print(type(sample))
    print('__getitem__ works.')

    # wrap DataLoader on top of this Dataset subclass
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=mnist, collate_fn=mnist.collate_fn,
                            batch_size=batch_size, shuffle=True, num_workers=0)

    # try to see if there is a speed up when generating batches w/ multiple workers
    #print("Measuring generation time. Please wait...") 
    #import time
    #s = time.time()
    #for i, batch in enumerate(dataloader):
        #print('Batch # {} - {}'.format(i, type(batch)))
        #pass
    #print('Number of workers: {}'.format(dataloader.num_workers))
    #print('Time taken to exhaust the dataset for a batch size of {}: {}s'.format(batch_size, time.time()-s))

    # Display single sample (0) from batch.
    batch = next(iter(dataloader))
    mnist.show_sample(batch, 0)
