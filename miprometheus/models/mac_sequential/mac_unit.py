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
mac_unit.py: Implementation of the MAC Unit for the MAC network. Cf https://arxiv.org/abs/1803.03067 for the \
reference paper.
"""
__author__ = "Vincent Marois"

import torch
from torch.nn import Module

from miprometheus.models.mac_sequential.control_unit import ControlUnit
from miprometheus.models.mac_sequential.read_unit import ReadUnit
from miprometheus.models.mac_sequential.write_unit import WriteUnit
from miprometheus.models.mac_sequential.read_memory import ReadMemory
from miprometheus.models.mac_sequential.utils_mac import linear
from miprometheus.models.dwm.tensor_utils import circular_conv
from miprometheus.utils.app_state import AppState
app_state = AppState()


class MACUnit(Module):
    """
    Implementation of the ``MACUnit`` (iteration over the MAC cell) of the MAC network.
    """

    def __init__(self, dim, max_step=2, self_attention=False,
                 memory_gate=False, dropout=0.15, slots=0):
        """
        Constructor for the ``MACUnit``, which represents the recurrence over the \
        MACCell.

        :param dim: global 'd' hidden dimension.
        :type dim: int

        :param max_step: maximal number of MAC cells. Default: 12
        :type max_step: int

        :param self_attention: whether or not to use self-attention in the ``WriteUnit``. Default: ``False``.
        :type self_attention: bool

        :param memory_gate: whether or not to use memory gating in the ``WriteUnit``. Default: ``False``.
        :type memory_gate: bool

        :param dropout: dropout probability for the variational dropout mask. Default: 0.15
        :type dropout: float

        """

        # call base constructor
        super(MACUnit, self).__init__()

        # instantiate the units
        self.control = ControlUnit(dim=dim, max_step=max_step)
        self.read = ReadUnit(dim=dim)
        self.read_memory = ReadMemory(dim=dim)
        self.write = WriteUnit(
            dim=dim, self_attention=self_attention, memory_gate=memory_gate)

        self.slots = slots
        print(self.slots)

        # initialize hidden states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim).type(app_state.dtype))
        self.control_0 = torch.nn.Parameter(
            torch.zeros(1, dim).type(app_state.dtype))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

        self.cell_state_history = []

        self.W = torch.zeros(48, 1, self.slots).type(app_state.dtype)


        self.linear_layer = linear(128, 1, bias=True)

        self.linear_layer_history = linear(128, 1, bias=True)

        self.linear_layer_mix_context = torch.nn.Sequential(linear(dim, dim, bias=True),
                                               torch.nn.ELU(),
                                               linear(dim, 3, bias=True))


        if slots==4:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 1.], [1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]).type(app_state.dtype)

        elif slots==6:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 0., 0., 1.], [1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
                 [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.]]).type(app_state.dtype)

        elif slots==8:
            self.convolution_kernel = torch.tensor(
                [[0., 0., 0., 0., 0., 0., 0., 1.], [1., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 1., 0.]]).type(app_state.dtype)

        else:
            exit()


        self.concat_contexts = torch.zeros(48, 128, requires_grad=False).type(app_state.dtype)


        self.linear_read = torch.nn.Sequential(linear(2*dim,2*dim, bias=True),
                                              torch.nn.ELU(),
                                              linear(2*dim, 1, bias=True))

        self.linear_read_history = torch.nn.Sequential(linear(2*dim,2*dim, bias=True),
                                               torch.nn.ELU(),
                                               linear(2*dim, 1, bias=True))




    def get_dropout_mask(self, x, dropout):
        """
        Create a dropout mask to be applied on x.

        :param x: tensor of arbitrary shape to apply the mask on.
        :type x: torch.tensor

        :param dropout: dropout rate.
        :type dropout: float

        :return: mask.

        """
        # create a binary mask, where the probability of 1's is (1-dropout)
        mask = torch.empty_like(x).bernoulli_(
            1 - dropout).type(app_state.dtype)

        # normalize the mask so that the average value is 1 and not (1-dropout)
        mask /= (1 - dropout)

        return mask

    def forward(self, context, question, knowledge, kb_proj, controls, memories, control_pass, memory_pass, control, memory, history,  Wt_sequential ):
        """
        Forward pass of the ``MACUnit``, which represents the recurrence over the \
        MACCell.

        :param context: contextual words, shape [batch_size x maxQuestionLength x dim]
        :type context: torch.tensor

        :param question: questions encodings, shape [batch_size x 2*dim]
        :type question: torch.tensor

        :param knowledge: knowledge_base (feature maps extracted by a CNN), shape \
        [batch_size x nb_kernels x (feat_H * feat_W)].
        :type knowledge: torch.tensor

        :return: list of the memory states.

        """
        batch_size = question.size(0)

        if not control_pass:

            controls = [control]

        if not memory_pass:

            memories = [memory]

        #empty state history
        self.cell_state_history = []



        # main loop of recurrence over the MACCell
        for i in range(self.max_step):

            # control unit
            control = self.control(
                step=i,
                contextual_words=context,
                question_encoding=question,
                ctrl_state=control)


            # save new control state
            controls.append(control)


            # read unit
            read ,attention = self.read(memory_states=memories, knowledge_base=knowledge,
                             ctrl_states=controls, kb_proj=kb_proj)

            # read memory

            read_history,rvi_history  = self.read_memory(memory_states=memories, history=history,
                             ctrl_states=controls)

            # calculate two gates gKB and gM gates

            concat_read=torch.cat([control, read], dim=1)

            gkb = self.linear_read(concat_read)
            gkb = torch.sigmoid(gkb)


            concat_read_history = torch.cat([control, read_history], dim=1)


            gmem = self.linear_read_history(concat_read_history)
            gmem = torch.sigmoid(gmem)

            # history update equation


            # print(self.Wt_sequential.size())

            W = (gmem * rvi_history.squeeze(1) + Wt_sequential.squeeze(1)*(1-gmem))*gkb
            W= W.unsqueeze(1)


            ######## Update history #########

            #take the read vector and add one dimension [batch size, hidden dim, 1]

            read_unsqueezed=read.unsqueeze(2)
            added_object = read_unsqueezed.matmul(W)

            unity_matrix = torch.ones(batch_size, self.dim, 1).type(app_state.dtype)
            J = torch.ones(batch_size, self.dim,self.slots).type(app_state.dtype)

            history=history*(J-unity_matrix.matmul(W))+added_object
            #print(added_object[0],added_object.size(),i)

            ####### Update Wt_sequential ########

            #calculate constants terms
            first_term=(torch.ones(batch_size,1).type(app_state.dtype)-gmem)*gkb
            second_term=torch.ones(batch_size,1).type(app_state.dtype)-first_term

            #get convolved tensor

            convolved_Wt_sequential=Wt_sequential.squeeze(1).matmul(self.convolution_kernel).unsqueeze(1)

            #final expression to update Wt_sequential

            Wt_sequential = (convolved_Wt_sequential.squeeze(1)*first_term).unsqueeze(1)+(Wt_sequential.squeeze(1)*second_term).unsqueeze(1)
            #print(Wt_sequential[0],i)


            # choose between now, last, latest context to built the final read vector
            now_context = gkb*read
            last_context =  gmem*read_history
            latest_context = (1-gkb)*last_context + now_context



            #obtain neural network that mixes the 3 context (v1,v2,v3)  #### COULD BE 2 LAYERS ? ######
            context_weighting_vector = self.linear_layer_mix_context(control)
            context_weighting_vector = torch.nn.functional.softmax(context_weighting_vector,dim=1)
            context_weighting_vector = context_weighting_vector.unsqueeze(1)


            #final read vector
            context_read_vector = context_weighting_vector[:,:,0]*now_context + context_weighting_vector[:,:,1]*last_context + context_weighting_vector[:,:,2]*latest_context


            # write unit
            memory = self.write(memory_states=memories,
                                read_vector= context_read_vector , ctrl_states=controls)


            # save new memory state
            memories.append(memory)


            # store attention weights for visualization
            if app_state.visualize:
                self.cell_state_history.append(
                    (self.read.rvi.cpu().detach(), self.control.cvi.cpu().detach(),history.detach(), rvi_history.detach(),gmem,gkb, Wt_sequential, context_weighting_vector))

        return memory, controls, memories, self.cell_state_history, attention, history, Wt_sequential
