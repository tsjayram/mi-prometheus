#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ntm_cell.py: pytorch module implementing single (recurrent) cell of Neural Turing Machine"""
__author__ = "Tomasz Kornuta"

import torch 
import collections
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controllers'))
from controller_factory import ControllerFactory
from misc.app_state import AppState

from models.encoder_solver.mae_interface import MAEInterface

# Helper collection type.
_MAECellStateTuple = collections.namedtuple('MAECellStateTuple', ('ctrl_state', 'interface_state',  'memory_state'))

class MAECellStateTuple(_MAECellStateTuple):
    """Tuple used by MAE Cells for storing current/past state information"""
    __slots__ = ()


class MAECell(torch.nn.Module):
    """ Class representing a single Memory-Augmented Encoder cell. """

    def __init__(self, params):
        """ Cell constructor.
        Cell creates controller and interface.
        It also initializes memory "block" that will be passed between states.
            
        :param params: Dictionary of parameters.
        """
        # Call constructor of base class.
        super(NTMCell, self).__init__() 
        
        # Parse parameters.
        # Set input and output sizes. 
        self.input_size = params["num_control_bits"] + params["num_data_bits"]
        try:
            self.output_size  = params['num_output_bits']
        except KeyError:
            self.output_size = params['num_data_bits']

        # Get controller hidden state size.
        self.controller_hidden_state_size = params['controller']['hidden_state_size']



        # Controller - entity that processes input and produces hidden state of the ntm cell.        
        ext_controller_inputs_size = self.input_size
        # Create dictionary wirh controller parameters.
        controller_params = {
           "name":  params['controller']['name'],
           "input_size": ext_controller_inputs_size,
           "output_size": self.controller_hidden_state_size,
           "non_linearity": params['controller']['non_linearity'], 
           "num_layers": params['controller']['num_layers']
        }
        # Build the controller.
        self.controller = ControllerFactory.build_model(controller_params)  

        # Interface - entity responsible for accessing the memory.
        self.interface = MAEInterface(params)
        
        
    def init_state(self,  batch_size,  num_memory_addresses, num_memory_content_bits):
        """
        Returns 'zero' (initial) state:
        * memory  is reset to random values.
        * read & write weights are set to 1e-6.
        * read_vectors are initialize as 0s.
        
        :param batch_size: Size of the batch in given iteraction/epoch.
        :param num_memory_addresses: Number of memory addresses.
        :param num_memory_content_bits: Number of memory content bits.
        :returns: Initial state tuple - object of NTMCellStateTuple class.
        """
        dtype = AppState().dtype

        # Initialize controller state.
        ctrl_init_state =  self.controller.init_state(batch_size)

        # Initialize interface state. 
        interface_init_state =  self.interface.init_state(batch_size,  num_memory_addresses)

        # Memory [BATCH_SIZE x MEMORY_ADDRESSES x CONTENT_BITS] 
        init_memory_BxAxC = torch.zeros(batch_size,  num_memory_addresses,  num_memory_content_bits).type(dtype)
        
        # Pack and return a tuple.
        return MAECellStateTuple(ctrl_init_state, interface_init_state,  init_memory_BxAxC)


    def forward(self, inputs_BxI,  prev_cell_state):
        """
        Forward function of NTM cell.
        
        :param inputs_BxI: a Tensor of input data of size [BATCH_SIZE  x INPUT_SIZE]
        :param  prev_cell_state: a MAECellStateTuple tuple, containing previous state of the cell.
        :returns: MAECellStateTuple tuple containing current cell state.
        """
        # Unpack previous cell  state.
        (prev_ctrl_state_tuple, prev_interface_state_tuple,  prev_memory_BxAxC) = prev_cell_state

        controller_input = inputs_BxI
        # Execute controller forward step.
        ctrl_output_BxH,  ctrl_state_tuple = self.controller(controller_input,  prev_ctrl_state_tuple)
       
        # Execute interface forward step.
        memory_BxAxC, interface_state_tuple = self.interface(ctrl_output_BxH, prev_memory_BxAxC,  prev_interface_state_tuple)
        
        # Pack current cell state.
        cell_state_tuple = MAECellStateTuple(ctrl_state_tuple, interface_state_tuple,  memory_BxAxC)
        
        # Return logits and current cell state.
        return cell_state_tuple
    
