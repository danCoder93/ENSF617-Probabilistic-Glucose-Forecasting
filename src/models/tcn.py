import os
import warnings
import math
from typing import Optional, Union, List, Tuple, Literal, cast, Sequence
from collections.abc import Iterable

# Numpy
from numpy.typing import ArrayLike, NDArray
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm

from utils.config import TCNConfig


class BufferIO():
    def __init__(
            self,
            in_buffers: Optional[ Iterable ] = None,
            ):
        if in_buffers is not None:
            in_buffers = list( in_buffers )
            self.in_buffers_length = len( in_buffers )
            self.in_buffers = iter( in_buffers )
        else:
            self.in_buffers_length = None
            self.in_buffers = None
        
        self.out_buffers = []
        self.internal_buffers = []
        return
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.in_buffers is not None:
            return next( self.in_buffers )
        else:
            return None
        
    def append_out_buffer(
            self,
            x: Tensor,
            ):
        self.out_buffers.append(x)
        return
    
    def append_internal_buffer(
            self,
            x: Tensor,
            ):
        self.internal_buffers.append(x)
        return
        
    def next_in_buffer(
            self,
            ):
        return self.__next__()
        
    def step(self):
        # If in_buffers is None, then the internal buffers are used as input
        # After the first step, the operation will continue as usual
        if self.in_buffers is None:
            self.in_buffers_length = len( self.internal_buffers)
        if len( self.out_buffers ) != self.in_buffers_length:
            raise ValueError(
                """
                Number of out buffers does not match number of in buffers.
                """
                )
        self.in_buffers = iter( self.out_buffers )
        self.out_buffers = []
        return

# Padding modes
PADDING_MODES = [
    'zeros',
    'reflect',
    'replicate',
    'circular',
]

class TemporalPad1d(nn.Module):
    def __init__(
            self,
            padding: int,
            in_channels: int,
            buffer: Optional[ Union[ float, Tensor ] ] = None,
            padding_mode: str = 'zeros',
            causal: bool = False,
            ):
        super(TemporalPad1d, self).__init__()

        if not isinstance(padding, int):
            raise ValueError(
                f"""
                padding must be an integer, but got {type(padding)}.
                padding must not be a tuple, because the TemporalPadding
                will automatically determine the amount of left and right
                padding based on the causal flag.
                """
                )

        self.pad_len = padding
        self.causal = causal

        if causal:
            # Padding is only on the left side
            self.left_padding = self.pad_len
            self.right_padding = 0
        else:
            # Padding is on both sides
            self.left_padding = self.pad_len // 2
            self.right_padding = self.pad_len - self.left_padding
        
        if padding_mode == 'zeros':
            self.pad = nn.ConstantPad1d(
                (self.left_padding, self.right_padding),
                0.0,
                )
        elif padding_mode == 'reflect':
            self.pad = nn.ReflectionPad1d(
                (self.left_padding, self.right_padding),
                )
        elif padding_mode == 'replicate':
            self.pad = nn.ReplicationPad1d(
                (self.left_padding, self.right_padding),
                )
        elif padding_mode == 'circular':
            self.pad = nn.CircularPad1d(
                (self.left_padding, self.right_padding),
                )
        else:
            raise ValueError(
                f"""
                padding_mode must be one of {PADDING_MODES},
                but got {padding_mode}.
                """
                )
        
        # Buffer is used for streaming inference
        if buffer is None:
            if in_channels is None:
                buffer = torch.zeros(
                    1,
                    self.pad_len,
                    )
            else:
                buffer = torch.zeros(
                    1,
                    in_channels,
                    self.pad_len,
                    )
        elif isinstance(buffer, (int, float)):
            if in_channels is None:
                buffer = torch.full(
                    size = (1, self.pad_len),
                    fill_value = buffer,
                    )
            else:
                buffer = torch.full(
                    size = (1, in_channels, self.pad_len),
                    fill_value = buffer,
                    )
        elif not isinstance(buffer, Tensor):
            raise ValueError(
                f"""
                The argument 'buffer' must be None or of type float,
                int, or Tensor, but got {type(buffer)}.
                """
                )
        
        # Register buffer as a persistent buffer which is available as self.buffer
        self.register_buffer(
            'buffer',
            buffer,
            )
        
        return
    
    def pad_inference(
            self,
            x: Tensor,
            buffer_io: Optional[ BufferIO ] = None,
            ):

        if not self.causal:
            raise ValueError(
                """
                Streaming inference is only supported for causal convolutions.
                """
                )

        if x.shape[0] != 1:
            raise ValueError(
                f"""
                Streaming inference requires a batch size
                of 1, but batch size is {x.shape[0]}.
                """
                )
        
        if buffer_io is None:
            in_buffer = self.buffer
        else:
            in_buffer = buffer_io.next_in_buffer()
            if in_buffer is None:
                in_buffer = self.buffer
                buffer_io.append_internal_buffer( in_buffer )

        x = torch.cat(
            (in_buffer, x),
            -1,
            )

        out_buffer = x[ ..., -self.pad_len: ]
        if buffer_io is None:
            self.buffer = out_buffer
        else:
            buffer_io.append_out_buffer(out_buffer)

        return x
    
    def forward(
            self,
            x: Tensor,
            inference: bool = False,
            buffer_io: Optional[ BufferIO ] = None,
            ):
        if inference:
            x = self.pad_inference(x, buffer_io=buffer_io)
        else:
            x = self.pad(x)
        return x
    
    def reset_buffer(self):
        self.buffer.zero_()
        if self.buffer.shape[-1] != self.pad_len:
            raise ValueError(
                f"""
                Buffer shape {self.buffer.shape} does not match the expected
                shape (1, {self.in_channels}, {self.pad_len}).
                """
                )

class TemporalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = True,
            padding_mode='zeros',
            device=None,
            dtype=None,
            buffer = None,
            causal = True,
            lookahead = 0,
            ):
        super(TemporalConv1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0, # Padding is reimplemented in this class
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode='zeros', # Padding is reimplemented in this class
            device=device,
            dtype=dtype,
            )
        
        # Padding is computed internally
        if padding != 0:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    The value of 'padding' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    If you want to suppress this error in order to use the layer as drop-in replacement
                    for nn.Conv1d, set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN' to '0'
                    (will reduce error to a warning) or '1' (will suppress the error/warning entirely).
                    """
                    )

        # Lookahead is only kept for legacy reasons, ensure it is zero
        if lookahead != 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                The parameter will be removed in a future version.
                """
                )

        self.pad_len = (kernel_size - 1) * dilation
        self.causal = causal
        
        self.padder = TemporalPad1d(
            padding = self.pad_len,
            in_channels = in_channels,
            buffer = buffer,
            padding_mode = padding_mode,
            causal = causal,
            )
        
        return
    
    # In pytorch-tcn >= 1.2.2, buffer is moved to TemporalPad1d
    # We keep the property for backwards compatibility, e.g. in
    # case one wants to load old model weights.
    @property
    def buffer(self):
        return self.padder.buffer
    
    @buffer.setter
    def buffer(self, value):
        self.padder.buffer = value
        return

    def forward(
            self,
            x: Tensor,
            inference: bool = False,
            in_buffer: Tensor | None = None,
            buffer_io: Optional[ BufferIO ] = None,
            ):
        if in_buffer is not None:
            raise ValueError(
                """
                The argument 'in_buffer' was removed in pytorch-tcn >= 1.2.2.
                Instead, you should pass the input buffer as a BufferIO object
                to the argument 'buffer_io'.
                """
                )
        x = self.padder(x, inference=inference, buffer_io=buffer_io)
        x = super().forward(x)
        return x
    
    def inference(self, *args, **kwargs):
        raise NotImplementedError(
            """
            The function "inference" was removed in pytorch-tcn >= 1.2.2.
            Instead, you should use the modules forward function with the
            argument "inference=True" enabled.
            """
            )
        return
    
    def reset_buffer(self):
        self.padder.reset_buffer()
        return


class TemporalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            padding = 0,
            output_padding = 0,
            groups = 1,
            bias = True,
            dilation = 1,
            padding_mode = 'zeros',
            device=None,
            dtype=None,
            buffer = None,
            causal = True,
            lookahead = 0,
            ):
        
        # Padding is computed internally
        if padding != 0:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    The value of 'padding' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    If you want to suppress this error in order to use the layer as drop-in replacement
                    for nn.Conv1d, set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN' to '0'
                    (will reduce error to a warning) or '1' (will suppress the error/warning entirely).
                    """
                    )
            
        # dilation rate should be 1
        if dilation != 1:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'dilation' must be 1 for TemporalConvTranspose1d, other values are
                    not supported. The value of 'dilation' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'dilation' must be 1 for TemporalConvTranspose1d, other values are
                    not supported. If you want to suppress this error in order to use the layer as drop-in
                    replacement for nn.ConvTranspose1d, set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN'
                    to '0' (will reduce error to a warning) or '1' (will suppress the error/warning entirely).
                    """
                    )
            
        # output_padding should be 0
        if output_padding != 0:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'output_padding' must be 0 for TemporalConvTranspose1d, because the correct
                    amount of padding is calculated automatically based on the kernel size and stride. The value
                    of 'output_padding' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'output_padding' must be 0 for TemporalConvTranspose1d, because the correct
                    amount of padding is calculated automatically based on the kernel size and stride. If you want
                    to suppress this error in order to use the layer as drop-in replacement for nn.ConvTranspose1d,
                    set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN' to '0' (will reduce error to a warning)
                    or '1' (will suppress the error/warning entirely).
                    """
                    )

        # Lookahead is only kept for legacy reasons, ensure it is zero
        if lookahead != 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                The parameter will be removed in a future version.
                """
                )

        # This implementation only supports kernel_size == 2 * stride
        if kernel_size != 2 * stride:
            raise ValueError(
                f"""
                This implementation of TemporalConvTranspose1d only
                supports kernel_size == 2 * stride, but got 
                kernel_size = {kernel_size} and stride = {stride}.
                """
                )


        self.causal = causal                      
        self.upsampling_factor = stride
        self.buffer_size = (kernel_size // stride) - 1

        if self.causal:
            #self.pad_left = self.buffer_size
            #self.pad_right = 0
            self.implicit_padding = 0
        else:
            #self.pad_left = 0
            #self.pad_right = 0
            self.implicit_padding = (kernel_size-stride)//2

        super(TemporalConvTranspose1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = self.implicit_padding, # Padding is patially reimplemented in this class
            output_padding = 0, # Output padding is not supported
            groups = groups,
            bias = bias,
            dilation = 1, # Dilation is not supported
            padding_mode = 'zeros', # Padding mode is reimplemented in this class
            device=device,
            dtype=dtype,
            )
        
        self.padder = TemporalPad1d(
            padding = self.buffer_size,
            in_channels = in_channels,
            padding_mode = padding_mode,
            causal = causal,
            )

        # Deprecated in pytorch-tcn >= 1.2.2
        # Keep for backwards compatibility to load old model weights
        if buffer is None:
            buffer = torch.zeros(
                1,
                in_channels,
                self.buffer_size,
                )
        self.register_buffer(
            'buffer',
            buffer,
            )

        return
        
    def forward(
            self,
            x: Tensor,
            inference: bool = False,
            in_buffer: Tensor | None = None,
            buffer_io: Optional[ BufferIO ] = None,
            ):
        if in_buffer is not None:
            raise ValueError(
                """
                The argument 'in_buffer' was removed in pytorch-tcn >= 1.2.2.
                Instead, you should pass the input buffer as a BufferIO object
                to the argument 'buffer_io'.
                """
                )
        if self.causal:
            x = self.padder(x, inference=inference, buffer_io=buffer_io)
            x = super().forward(x)
            x = x[:, :, self.upsampling_factor : -self.upsampling_factor]
        else:
            x = super().forward(x)
            # if stride is odd, remove last element due to padding
            if self.upsampling_factor % 2 == 1:
                x = x[..., :-1]
        return x
    
    def inference(self, *args, **kwargs):
        raise NotImplementedError(
            """
            The function "inference" was removed in pytorch-tcn >= 1.2.2.
            Instead, you should use the modules forward function with the
            argument "inference=True" enabled.
            """
            )
        return
    
    def reset_buffer(self):
        self.padder.reset_buffer()


GainNonlinearity = Literal["linear", "conv1d", "conv2d", "conv3d",
                           "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
                           "sigmoid", "tanh", "relu", "leaky_relu", "selu"]


activation_fn = dict(
    relu=nn.ReLU,
    tanh=nn.Tanh,
    leaky_relu=nn.LeakyReLU,
    sigmoid=nn.Sigmoid,
    elu=nn.ELU,
    gelu=nn.GELU,
    selu=nn.SELU,
    softmax=nn.Softmax,
    log_softmax=nn.LogSoftmax,
)

kernel_init_fn = dict(
    xavier_uniform=nn.init.xavier_uniform_,
    xavier_normal=nn.init.xavier_normal_,
    kaiming_uniform=nn.init.kaiming_uniform_,
    kaiming_normal=nn.init.kaiming_normal_,
    normal=nn.init.normal_,
    uniform=nn.init.uniform_,
)

def _check_activation_arg(
        activation,
        arg_name,
        ):
    if activation is None and arg_name == 'output_activation':
        return
    if isinstance( activation, str ):
        if activation not in activation_fn.keys():
            raise ValueError(
                f"""
                If argument '{arg_name}' is a string, it must be one of:
                {activation_fn.keys()}. However, you may also pass any
                torch.nn.Module object as the 'activation' argument.
                """
                )
    else:
        try:
            if not isinstance( activation(), nn.Module ):
                raise ValueError(
                    f"""
                    The argument '{arg_name}' must either be a valid string or
                    a torch.nn.Module object, but an object of type {type(activation())}
                    was passed.
                    """
                    )
        except:
            raise ValueError(
                f"""
                The argument '{arg_name}' must either be a valid string or
                a torch.nn.Module object, but an object of type {type(activation)}
                was passed.
                """
                )
    return

def _check_generic_input_arg(
        arg,
        arg_name,
        allowed_values,
        ):
    if arg not in allowed_values:
        raise ValueError(
            f"""
            Argument '{arg_name}' must be one of: {allowed_values},
            but {arg} was passed.
            """
            )
    return

def get_kernel_init_fn(
        name: str,
        activation: str,
        ):
    try:
        if isinstance( activation, str ):
            activation_fn_obj = activation_fn[ activation ]()
        else:
            activation_fn_obj = activation()
        if isinstance( activation_fn_obj, nn.Module ):
            return kernel_init_fn[ name ], dict()
        # TODO: this means no gain is used for custom activation functions
    except:
        pass
        
    if name not in kernel_init_fn.keys():
        raise ValueError(
            f"Argument 'kernel_initializer' must be one of: {kernel_init_fn.keys()}"
            )
    if name in [ 'xavier_uniform', 'xavier_normal' ]:
        if activation in [ 'gelu', 'elu', 'softmax', 'log_softmax' ]:
            warnings.warn(
                f"""
                Argument 'kernel_initializer' {name}
                is not compatible with activation {activation} in the
                sense that the gain is not calculated automatically.
                Here, a gain of sqrt(2) (like in ReLu) is used.
                This might lead to suboptimal results.
                """
                )
            gain = np.sqrt( 2 )
        else:
            gain = nn.init.calculate_gain( cast(GainNonlinearity, activation) )
        kernel_init_kw = dict( gain=gain )
    elif name in [ 'kaiming_uniform', 'kaiming_normal' ]:
        if activation in [ 'gelu', 'elu', 'softmax', 'log_softmax' ]:
            raise ValueError(
                f"""
                Argument 'kernel_initializer' {name}
                is not compatible with activation {activation}.
                It is recommended to use 'relu' or 'leaky_relu'.
                """
                )
        else:
            nonlinearity = activation
        kernel_init_kw = dict( nonlinearity=nonlinearity )
    else:
        kernel_init_kw = dict()
    
    return kernel_init_fn[ name ], kernel_init_kw



class BaseTCN(nn.Module):
    def __init__(
            self,
            ):
        super(BaseTCN, self).__init__()
        return
    
    def inference(
            self,
            *args,
            **kwargs,
            ):
        
        return self( *args, inference=True, **kwargs )
    
    def init_weights(self):
        
        def _init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d) ):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_init_weights)

        return
    
    def reset_buffers(self):
        def _reset_buffer(x):
            if isinstance(x, (TemporalPad1d,) ):
                x.reset_buffer()
        self.apply(_reset_buffer)
        return
    
    def get_buffers(self):
        """
        Get all buffers of the network in the order they were created.
        """
        buffers = []
        def _get_buffers(x):
            if isinstance(x, (TemporalPad1d,) ):
                buffers.append(x.buffer)
        self.apply(_get_buffers)
        return buffers
    
    def get_in_buffers(self, *args, **kwargs):
        """
        Get all buffers of the network in the order they are used in
        the forward pass. This is important for external buffer io, e.g.
        with ONNX inference.
        """
        # Get the internal buffer state
        buffers = self.get_buffers()
        # Get the in_buffers via dummy forward pass
        buffer_io = BufferIO( in_buffers=None )
        self(
            *args,
            inference=True,
            buffer_io=buffer_io,
            **kwargs,
            )
        in_buffers = buffer_io.internal_buffers
        # Restore the internal buffer state
        self.set_buffers( buffers )
        return in_buffers
    
    def set_buffers(self, buffers):
        """
        Set all buffers of the network in the order they were created.
        """
        def _set_buffers(x):
            if isinstance(x, (TemporalPad1d,) ):
                x.buffer = buffers.pop(0)
        self.apply(_set_buffers)
        return
    

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)




class TemporalBlock(BaseTCN):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            dropout,
            causal,
            use_norm,
            activation,
            kerner_initializer,
            embedding_shapes,
            embedding_mode,
            use_gate
            ):
        super(TemporalBlock, self).__init__()
        self.use_norm = use_norm
        self.activation = activation
        self.kernel_initializer = kerner_initializer
        self.embedding_shapes = embedding_shapes
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal

        if self.use_gate:
            conv1d_n_outputs = 2 * n_outputs
        else:
            conv1d_n_outputs = n_outputs


        self.conv1 = TemporalConv1d(
            in_channels=n_inputs,
            out_channels=conv1d_n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=self.causal
            )

        self.conv2 = TemporalConv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=self.causal
            )
        
        if use_norm == 'batch_norm':
            if self.use_gate:
                self.norm1 = nn.BatchNorm1d(2 * n_outputs)
            else:
                self.norm1 = nn.BatchNorm1d(n_outputs)
            self.norm2 = nn.BatchNorm1d(n_outputs)
        elif use_norm == 'layer_norm':
            if self.use_gate:
                self.norm1 = nn.LayerNorm(2 * n_outputs)
            else:
                self.norm1 = nn.LayerNorm(n_outputs)
            self.norm2 = nn.LayerNorm(n_outputs)
        elif use_norm == 'weight_norm':
            self.norm1 = None
            self.norm2 = None
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
        elif use_norm is None:
            self.norm1 = None
            self.norm2 = None

        if isinstance( self.activation, str ):
            self.activation1 = activation_fn[ self.activation ]()
            self.activation2 = activation_fn[ self.activation ]()
            self.activation_final = activation_fn[ self.activation ]()
        else:
            self.activation1 = self.activation()
            self.activation2 = self.activation()
            self.activation_final = self.activation()

        if self.use_gate:
            self.activation1 = nn.GLU(dim=1)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, padding=0) if n_inputs != n_outputs else None

        if self.embedding_shapes is not None:
            if self.use_gate:
                embedding_layer_n_outputs = 2 * n_outputs
            else:
                embedding_layer_n_outputs = n_outputs

            self.embedding_projection_1 = nn.Conv1d(
                in_channels = sum( [ shape[0] for shape in self.embedding_shapes ] ),
                out_channels = embedding_layer_n_outputs,
                kernel_size = 1,
                )
            
            self.embedding_projection_2 = nn.Conv1d(
                in_channels = 2 * embedding_layer_n_outputs,
                out_channels = embedding_layer_n_outputs,
                kernel_size = 1,
                )
        
        self.init_weights()
        return

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
            )
        initialize(
            self.conv1.weight,
            **kwargs # pyright: ignore[reportArgumentType]
            )
        initialize(
            self.conv2.weight,
            **kwargs # pyright: ignore[reportArgumentType]
            )

        if self.downsample is not None:
            initialize(
                self.downsample.weight,
                **kwargs # pyright: ignore[reportArgumentType]
                )
        return
    
    def apply_norm(
            self,
            norm_fn,
            x,
        ):
        if self.use_norm == 'batch_norm':
            x = norm_fn(x)
        elif self.use_norm == 'layer_norm':
            x = norm_fn( x.transpose(1, 2) )
            x = x.transpose(1, 2)
        return x
    
    def apply_embeddings(
            self,
            x,
            embeddings,
            ):
        
        if not isinstance( embeddings, list ):
            embeddings = [ embeddings ]

        e = []
        for embedding, expected_shape in zip( embeddings, self.embedding_shapes ):
            if embedding.shape[1] != expected_shape[0]:
                raise ValueError(
                    f"""
                    Embedding shape {embedding.shape} passed to 'forward' does not 
                    match the expected shape {expected_shape} provided as input to
                    argument 'embedding_shapes'.
                    """
                    )
            if len( embedding.shape ) == 2:
                # unsqueeze time dimension of e and repeat it to match x
                e.append( embedding.unsqueeze(2).repeat(1, 1, x.shape[2]) )
            elif len( embedding.shape ) == 3:
                # check if time dimension of embedding matches x
                if embedding.shape[2] != x.shape[2]:
                    raise ValueError(
                        f"""
                        Embedding time dimension {embedding.shape[2]} does not
                        match the input time dimension {x.shape[2]}
                        """
                        )
                e.append( embedding )
        e = torch.cat( e, dim=1 )
        e = self.embedding_projection_1( e )
        #print('shapes:', e.shape, x.shape)
        if self.embedding_mode == 'concat':
            x = self.embedding_projection_2(
                torch.cat( [ x, e ], dim=1 )
                )
        elif self.embedding_mode == 'add':
            x = x + e

        return x
    
    def forward(
            self,
            x,
            embeddings,
            inference,
            in_buffers=None,
            ):
        
        if in_buffers:
            in_buffer_1, in_buffer_2 = in_buffers
        else:
            in_buffer_1, in_buffer_2 = None, None

        out = self.conv1(x, inference=inference, in_buffer = in_buffer_1)
        out = self.apply_norm( self.norm1, out )

        if embeddings is not None:
            out = self.apply_embeddings( out, embeddings )

        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out, inference=inference, in_buffer = in_buffer_2)
        out = self.apply_norm( self.norm2, out )
        out = self.activation2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.activation_final(out + res), out



class TCN(BaseTCN):
    def __init__(
            self,
            config : TCNConfig,
            ):
        super(TCN, self).__init__()

        if config.lookahead > 0:
            # Only lookahead of 0 is supported, parameter is kept for compatibility
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                The parameter will be removed in a future version.
                """
                )


        if config.dilations is not None and len(config.dilations) != len(config.num_channels):
            raise ValueError("Length of dilations must match length of num_channels")
        
        self.allowed_norm_values = ['batch_norm', 'layer_norm', 'weight_norm', None]
        self.allowed_input_shapes = ['NCL', 'NLC']

        _check_generic_input_arg( config.causal, 'causal', [True, False] )
        _check_generic_input_arg( config.use_norm, 'use_norm', self.allowed_norm_values )
        _check_activation_arg(config.activation, 'activation')
        _check_generic_input_arg( config.kernel_initializer, 'kernel_initializer', kernel_init_fn.keys() )
        _check_generic_input_arg( config.use_skip_connections, 'use_skip_connections', [True, False] )
        _check_generic_input_arg( config.input_shape, 'input_shape', self.allowed_input_shapes )
        _check_generic_input_arg( config.embedding_mode, 'embedding_mode', ['add', 'concat'] )
        _check_generic_input_arg( config.use_gate, 'use_gate', [True, False] )
        _check_activation_arg(config.output_activation, 'output_activation')

        if config.dilations is None:
            if config.dilation_reset is None:
                dilations = [ 2 ** i for i in range( len( config.num_channels ) ) ]
            else:
                # Calculate after which layers to reset
                dilation_reset = int( np.log2( config.dilation_reset * 2 ) )
                dilations = [
                    2 ** (i % dilation_reset)
                    for i in range( len( config.num_channels ) )
                    ]
            
        self.dilations = dilations
        self.activation = config.activation
        self.kernel_initializer = config.kernel_initializer
        self.use_skip_connections = config.use_skip_connections
        self.input_shape = config.input_shape
        self.embedding_shapes = config.embedding_shapes
        self.embedding_mode = config.embedding_mode
        self.use_gate = config.use_gate
        self.causal = config.causal
        self.output_projection = config.output_projection
        self.output_activation = config.output_activation

        if config.embedding_shapes is not None:
            if isinstance(config.embedding_shapes, Iterable):
                for shape in config.embedding_shapes:
                    if not isinstance( shape, tuple ):
                        try:
                            shape = tuple( shape ) if isinstance(shape, Iterable) and not isinstance(shape, str) else (shape,)
                        except Exception as e:
                            raise ValueError(
                                f"""
                                Each shape in argument 'embedding_shapes' must be an Iterable of tuples.
                                Tried to convert {shape} to tuple, but failed with error: {e}
                                """
                                )
                    if len( shape ) not in [ 1, 2 ]:
                        raise ValueError(
                            f"""
                            Tuples in argument 'embedding_shapes' must be of length 1 or 2.
                            One-dimensional tuples are interpreted as (embedding_dim,) and
                            two-dimensional tuples as (embedding_dim, time_steps).
                            """
                            )
            else:
                raise ValueError(
                    f"""
                    Argument 'embedding_shapes' must be an Iterable of tuples,
                    but is {type(config.embedding_shapes)}.
                    """
                    )
            

        if config.use_skip_connections:
            self.downsample_skip_connection = nn.ModuleList()
            for i in range( len( config.num_channels ) ):
                # Downsample layer output dim to network output dim if needed
                if config.num_channels[i] != config.num_channels[-1]:
                    self.downsample_skip_connection.append(
                        nn.Conv1d( config.num_channels[i], config.num_channels[-1], 1 )
                        )
                else:
                    self.downsample_skip_connection.append( None ) # pyright: ignore[reportArgumentType]
            self.init_skip_connection_weights()
            if isinstance( self.activation, str ):
                self.activation_skip_out = activation_fn[ self.activation ]()
            else:
                self.activation_skip_out = self.activation()
        else:
            self.downsample_skip_connection = None
        
        layers = []
        num_levels = len(config.num_channels)
        
        for i in range(num_levels):
            dilation_size = self.dilations[i]

            in_channels = config.num_inputs if i == 0 else config.num_channels[i-1]
            out_channels = config.num_channels[i]

            layers += [
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=config.kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=config.dropout,
                    causal=config.causal,
                    use_norm=config.use_norm,
                    activation=config.activation,
                    kerner_initializer=self.kernel_initializer,
                    embedding_shapes=self.embedding_shapes,
                    embedding_mode=self.embedding_mode,
                    use_gate=self.use_gate
                    )
                ]

        self.network = nn.ModuleList(layers)

        if self.output_projection is not None:
            self.projection_out = nn.Conv1d(
                in_channels=config.num_channels[-1],
                out_channels=self.output_projection,
                kernel_size=1,
                )
        else:
            self.projection_out = None

        if self.output_activation is not None:
            if isinstance( self.output_activation, str ):
                self.activation_out = activation_fn[ self.output_activation ]()
            else:
                self.activation_out = self.output_activation()
        else:
            self.activation_out = None #nn.Identity()

        if self.causal:
            self.reset_buffers()
        return
    
    def init_skip_connection_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
            )
        for layer in self.downsample_skip_connection: # pyright: ignore[reportOptionalIterable]
            if layer is not None:
                initialize(
                    layer.weight, # pyright: ignore[reportArgumentType]
                    **kwargs # pyright: ignore[reportArgumentType]
                    )
        return

    def forward(
            self,
            x,
            embeddings=None,
            inference=False,
            in_buffers=None,
            ):
        if inference and not self.causal:
            raise ValueError(
                """
                This streaming inference mode is made for blockwise causal
                processing and thus, is only supported for causal networks.
                However, you selected a non-causal network.
                """
                )
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
        if self.use_skip_connections:
            skip_connections = []
            # Adding skip connections from each layer to the output
            # Excluding the last layer, as it would not skip trainable weights
            for index, layer in enumerate( self.network ):
                
                if in_buffers:
                    layer_in_buffers = in_buffers[ 2*index: ]
                else:
                    layer_in_buffers = None

                x, skip_out = layer(
                    x,
                    embeddings=embeddings,
                    inference=inference,
                    in_buffers=layer_in_buffers,
                    )
                if self.downsample_skip_connection[ index ] is not None: # pyright: ignore[reportOptionalSubscript]
                    skip_out = self.downsample_skip_connection[ index ]( skip_out ) # pyright: ignore[reportOptionalSubscript]
                if index < len( self.network ) - 1:
                    skip_connections.append( skip_out )
            skip_connections.append( x )
            x = torch.stack( skip_connections, dim=0 ).sum( dim=0 )
            x = self.activation_skip_out( x )
        else:
            for index, layer in enumerate( self.network ):
                
                if in_buffers:
                    layer_in_buffers = in_buffers[ 2*index: ]
                else:
                    layer_in_buffers = None
                #print( 'TCN, embeddings:', embeddings.shape )
                x, _ = layer(
                    x,
                    embeddings=embeddings,
                    inference=inference,
                    in_buffers=layer_in_buffers,
                    )
        if self.projection_out is not None:
            x = self.projection_out( x )
        if self.activation_out is not None:
            x = self.activation_out( x )
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
        return x
