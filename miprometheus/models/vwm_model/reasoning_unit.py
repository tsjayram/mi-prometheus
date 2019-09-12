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

        self.va_scale = torch.nn.Parameter(torch.ones(1).type(AppState().dtype))
        self.va_shift = torch.nn.Parameter(torch.ones(1).type(AppState().dtype))

        self.rh_scale = torch.nn.Parameter(torch.ones(1).type(AppState().dtype))
        self.rh_shift = torch.nn.Parameter(torch.ones(1).type(AppState().dtype))

        # self.visual_object_validator = torch.nn.Sequential(
        #     linear(1, 1, bias=True),
        #     torch.nn.Sigmoid())
        #
        # self.memory_object_validator = torch.nn.Sequential(
        #     linear(1, 1, bias=True),
        #     torch.nn.Sigmoid())

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

        va_aggregate = (visual_attention * visual_attention).sum(dim=-1, keepdim=False)
        rh_aggregate = (read_head * read_head).sum(dim=-1, keepdim=False)

        va_scale = 1 + torch.nn.functional.softplus(self.va_scale)
        va_shift = torch.nn.functional.sigmoid(self.va_shift)
        valid_vo = torch.nn.functional.sigmoid(va_scale*(va_aggregate-va_shift))
        # valid_vo = valid_vo.squeeze(-1)

        rh_scale = 1 + torch.nn.functional.softplus(self.rh_scale)
        rh_shift = torch.nn.functional.sigmoid(self.rh_shift)
        valid_mo = torch.nn.functional.sigmoid(rh_scale*(rh_aggregate-rh_shift))
        # valid_mo = valid_mo.squeeze(-1)

        # get t_now, t_last, t_latest, t_none from temporal_class_weights
        t_now = temporal_class_weights[:, 0]
        t_last = temporal_class_weights[:, 1]
        t_latest = temporal_class_weights[:, 2]

        # check if temporal context is last or latest
        temporal_test_1 = (t_last + t_latest) * (1 - t_now)

        # conditioned on temporal context,
        # check if we should replace existing memory object
        do_replace = valid_mo * valid_vo * temporal_test_1

        # otherwise, conditioned on temporal context,
        # check if we should add a new one to VWM
        do_add_new = (1 - valid_mo) * valid_vo * temporal_test_1

        # check if temporal context is now or latest
        temporal_test_2 = (t_now + t_latest) * (1 - t_last)

        # conditioned on temporal context, check if we have a valid visual object
        image_match = valid_vo * temporal_test_2

        # check if temporal context is either last, or latest without a visual object
        temporal_test_3 = (t_last + t_latest * (1 - valid_vo)) * (1 - t_now)

        # conditioned on temporal context, check if we have a valid memory object
        memory_match = valid_mo * temporal_test_3

        return image_match, memory_match, do_replace, do_add_new
