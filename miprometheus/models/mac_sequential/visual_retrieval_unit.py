#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2018 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
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
read_unit.py: Implementation of the ``VisualRetrievalUnitt`` for the VWM network. Cf https://arxiv.org/abs/1803.03067 for the \
reference paper.
"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.utils_mac import linear
from miprometheus.models.mac_sequential.attention_module import Attention_Module


class VisualRetrievalUnit(Module):
    """
    Implementation of the ``VisualReadUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the ``VisualReadUnit``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        """

        # call base constructor
        super(VisualRetrievalUnit, self).__init__()

        # define linear layer for the projection of the previous memory state
        self.summary_proj_layer = linear(dim, dim, bias=True)

        # linear layer to define I'(i,h,w) elements (r2 equation)
        self.aggregator = linear(2 * dim, dim, bias=True)

        # instantiate attention module
        self.attention_module = Attention_Module(dim)

        # linear layer to compute attention weights
        self.attn = linear(dim, 1, bias=True)

        # define linear layer for the projection of the knowledge base
        self.feature_maps_proj_layer = linear(dim, dim, bias=True)

    def forward(self, summary_object, feature_maps, ctrl_state):
        """
        Forward pass of the ``VisualRetrievalUnit``. Assuming 1 scalar attention weight per \
        knowledge base elements.

        :param memory_states: list of all previous memory states, each of shape [batch_size x mem_dim]
        :type memory_states: torch.tensor

        :param feature_maps: image representation (output of CNN), shape [batch_size x nb_kernels x (feat_H * feat_W)]
        :type feature_maps: torch.tensor

        :param ctrl_states: All previous control state, each of shape [batch_size x ctrl_dim].
        :type ctrl_states: list


        :return: current read vector, shape [batch_size x read_dim]

        """
        # assume mem_dim = ctrl_dim = nb_kernels = dim

        # pass memory state through linear layer
        summary_object = self.summary_proj_layer(summary_object).unsqueeze(2)
        # memory_state: [batch_size x dim x 1]

        # pass feature maps through linear layer
        feature_maps_proj = self.feature_maps_proj_layer(
            feature_maps.permute(0, 2, 1)).permute(0, 2, 1)

        # compute I(i,h,w) elements (r1 equation)
        # [batch_size x dim x 1] * [batch_size x dim x (H*W)] -> [batch_size x dim x (H*W)]
        I_elements = summary_object * feature_maps_proj

        # compute I' elements (r2 equation)
        concat = self.aggregator(
            torch.cat([I_elements,feature_maps],
                      dim=1).permute(0, 2, 1))  # [batch_size x (H*W) x dim]

        # compute attention weights
        visual_output, visual_attention = self.attention_module(ctrl_state,concat,feature_maps.permute(0, 2, 1))


        return visual_output, visual_attention
