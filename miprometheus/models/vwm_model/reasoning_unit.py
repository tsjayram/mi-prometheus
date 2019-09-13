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

        def validator_net():
            return torch.nn.Sequential(linear(2 * dim, 2 * dim, bias=True),
                                       torch.nn.ReLU(),
                                       linear(2 * dim, 1, bias=True),
                                       torch.nn.Sigmoid())

        def gate_net():
            return torch.nn.Sequential(linear(6, 12, bias=True),
                                       torch.nn.ELU(),
                                       linear(12, 12, bias=True),
                                       torch.nn.ELU(),
                                       linear(12, 3, bias=True),
                                       torch.nn.Softmax(dim=-1))

        self.visual_object_validator = validator_net()
        self.memory_object_validator = validator_net()

        self.match_gate_net = gate_net()
        self.memory_gate_net = gate_net()

    def forward(self, control_state, visual_object, visual_attention,
                memory_object, read_head, temporal_class_weights):
        """
        Forward pass of the ``ReasoningUnit``.

        :param control_state: last control state
        :param visual_object: visual object
        :param visual_attention: visual attention
        :param memory_object: memory object
        :param read_head: read head
        :param temporal_class_weights

        :return: image_match, memory_match, do_replace, do_add_new
        """

        # Compute a summary of each attention vector in [0,1]
        # The more the attention is localized, the more the summary will be closer to 1

        va_aggregate = (visual_attention * visual_attention).sum(dim=-1, keepdim=True)
        rh_aggregate = (read_head * read_head).sum(dim=-1, keepdim=True)

        # the visual object validator
        concat_visual = torch.cat([control_state, visual_object], dim=-1)
        valid_vo = self.visual_object_validator(concat_visual)

        # the memory object validator
        concat_memory = torch.cat([control_state, memory_object], dim=-1)
        valid_mo = self.memory_object_validator(concat_memory)

        gate_input = torch.cat([temporal_class_weights, valid_vo, valid_mo], dim=-1)

        match_out = self.match_gate_net(gate_input)
        image_match = match_out[..., 0]
        memory_match = match_out[..., 1]

        memory_out = self.memory_gate_net(gate_input)
        do_replace = memory_out[..., 0]
        do_add_new = memory_out[..., 1]

        return image_match, memory_match, do_replace, do_add_new
