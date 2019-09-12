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
reasoning_unit.py: Implementation of the ``ReasoningUnit`` for the VWM network.

"""
__author__ = "Vincent Albouy, T.S. Jayram"

import torch
from torch.nn import Module
from miprometheus.models.vwm_model.utils_VWM import linear
from miprometheus.utils.app_state import AppState


class ReasoningUnit(Module):
    """
    Implementation of the `` ReasoningUnit`` of the VWM network.
    """

    def __init__(self, dim):
        """
        Constructor for the `` ReasoningUnit``.
        :param dim: dimension of feature vectors
        :type dim: int
        """

        # call base constructor
        super(ReasoningUnit, self).__init__()

        def reasoning_net():
            return torch.nn.Sequential(linear(6, 12, bias=True),
                                       torch.nn.ELU(),
                                       linear(12, 12, bias=True),
                                       torch.nn.ELU(),
                                       linear(12, 3, bias=True),
                                       torch.nn.Softmax())

        self.match_gate_net = reasoning_net()
        self.memory_gate_net = reasoning_net()

    def forward(self, control_state, visual_attention, read_head, temporal_class_weights):
        """
        Forward pass of the ``ReasoningUnit``.

        :param control_state: last control state
        :param visual_attention: visual attention
        :param read_head: read head
        :param temporal_class_weights

        :return: image_match, memory_match, do_replace, do_add_new
        """

        # Compute a summary of each attention vector in [0,1]
        # The more the attention is localized, the more the summary will be closer to 1

        va_aggregate = (visual_attention * visual_attention).sum(dim=-1, keepdim=True)
        rh_aggregate = (read_head * read_head).sum(dim=-1, keepdim=True)

        reasoning_input = torch.cat([temporal_class_weights, va_aggregate, rh_aggregate], dim=-1)

        match_gate_out = self.match_gate_net(reasoning_input)
        image_match = match_gate_out[..., 0]
        memory_match = match_gate_out[..., 1]

        memory_gate_out = self.memory_gate_net(reasoning_input)
        do_replace = memory_gate_out[..., 0]
        do_add_new = memory_gate_out[..., 1]

        return image_match, memory_match, do_replace, do_add_new
