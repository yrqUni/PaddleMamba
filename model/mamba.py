import math
from dataclasses import dataclass
from typing import Union

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform, Constant

from .pscan import pscan

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = True # apply layernorms to internal activations

    pscan: bool = True # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Layer):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.LayerList([ResidualBlock(config) for _ in range(config.n_layers)])
        #self.norm_f = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        #x = self.norm_f(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Layer):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Layer):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(
            config.d_model, 2 * config.d_inner, 
            weight_attr=ParamAttr(initializer=Uniform(low=-0.1, high=0.1)), 
            bias_attr=ParamAttr(initializer=Constant(value=0)) if config.bias else False
        )

        self.conv1D = nn.Conv1D(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias_attr=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias_attr=False)

        # projects Δ from dt_rank to d_inner
        # dt initialization
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        dt_proj_weight_init = Uniform(low=-dt_init_std, high=dt_init_std) if config.dt_init == "random" else Constant(value=dt_init_std)
        dt_proj_bias_init = Constant(value=config.dt_init_floor)
        self.dt_proj = nn.Linear(
            config.dt_rank, config.d_inner, 
            weight_attr=ParamAttr(initializer=dt_proj_weight_init),
            bias_attr=ParamAttr(initializer=dt_proj_bias_init)
        )
        
        # dt bias
        dt = paddle.exp(paddle.rand([config.d_inner]) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)).clip(min=config.dt_init_floor)

        inv_dt = dt + paddle.log(-paddle.expm1(-dt)) 
        with paddle.no_grad():
            self.dt_proj.bias.set_value(-inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = paddle.arange(1, config.d_state + 1, dtype=paddle.float32)
        A = paddle.tile(A.unsqueeze(0), [config.d_inner, 1])
        self.A_log = paddle.create_parameter(shape=A.shape, dtype=str(A.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(A)) # why store A in log ? to keep A < 0 (cf -paddle.exp(...)) ? for gradient stability ?
        self.D = paddle.create_parameter(shape=[config.d_inner], dtype='float32', default_initializer=paddle.nn.initializer.Constant(1.0))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias_attr=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, axis=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose([0, 2, 1]) # (B, ED, L)
        x = self.conv1D(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose([0, 2, 1]) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -paddle.exp(self.A_log.astype('float32')) # (ED, N)
        D = self.D.astype('float32')
        # TODO remove .astype('float32')

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        delta, B, C = paddle.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], axis=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, L, ED)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D)
            
        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = paddle.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = paddle.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = paddle.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = paddle.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, paddle.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The paddle.zeros() isn't a problem (it's same as just feeding the input, because the conv1D is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, axis=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1D(paddle.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = paddle.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -paddle.exp(self.A_log.astype('float32')) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.astype('float32')
        # TODO remove .astype('float32')

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = paddle.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = paddle.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = paddle.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        # todo : pq h.squeeze(1) ??
        return y, h.squeeze(1)

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(paddle.nn.Layer):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[d_model],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(value=1.0)
        )

    def forward(self, x):
        output = x * paddle.rsqrt(paddle.mean(paddle.square(x), axis=-1, keepdim=True) + self.eps) * self.weight
        return output
