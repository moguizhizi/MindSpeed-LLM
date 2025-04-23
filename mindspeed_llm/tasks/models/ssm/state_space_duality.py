from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange, repeat


@dataclass
class ProcessInputs:
    """Input data structure for main processing flow"""
    x: torch.Tensor        # (B, L, H, P)
    dt: torch.Tensor       # (B, L, H)
    A: torch.Tensor        # (H,)
    B: torch.Tensor        # (B, L, G, S)
    C: torch.Tensor        # (B, L, G, S)
    D: torch.Tensor        # Residual matrix


class StateOptions:
    def __init__(self, initial_states=None, return_final_state=False, cached_start=False):
        self.initial_states = initial_states
        self.return_final_state = return_final_state
        self.cached_start = cached_start
        self.final_state = None  # Added for state persistence

    @property
    def should_return_final(self):
        return self.return_final_state or self.cached_start


class StateSpaceProcessor:
    def __init__(self, config):
        """
        Configuration should contain:
        - nheads_local: Number of local heads
        - ngroups_local: Number of local groups
        - dt_min/dt_max: Time step constraints
        - dt_bias: Time step bias term
        - headdim: Dimension per head
        - d_state: State dimension
        - chunk_size: Processing chunk size
        - D_has_hdim: Dimension for D matrix
        """
        self.config = self._validate_config(config)

    @property
    def h_ratio(self) -> int:
        return self.config['nheads_local'] // self.config['ngroups_local']

    def _validate_config(self, config) -> dict:
        """Configuration validation"""
        required_keys = {'nheads_local', 'ngroups_local', 'dt_min', 'dt_max', 'dt_bias',
                        'headdim', 'd_state', 'chunk_size', 'D_has_hdim'}
        missing = required_keys - config.keys()
        if missing:
            raise ValueError(f"Missing config keys: {missing}")
        return config

    def process(self, inputs: ProcessInputs, state_opts: StateOptions = StateOptions()):
        """
        Main processing pipeline
        Args:
            inputs: Input data
            state_opts: State options
        Returns:
            y: (B, L, H, P) Output features
        """
        # Unpack inputs
        x, dt, A, B, C, D = inputs.x, inputs.dt, inputs.A, inputs.B, inputs.C, inputs.D

        # Parameter initialization
        initial_states = self._prepare_initial_states(state_opts.initial_states)
        seq_len = x.size(1)
        pad_size = self._calculate_padding(seq_len)

        # Dimension transformations
        x, dt, A, B, C = self._expand_dims(x, A, dt, B, C)
        B_exp, C_exp = self._expand_groups_to_heads(B, C)
        dt_proc = self._process_time_step(dt)
        D = self._prepare_residual(D, x, pad_size)

        # Chunk processing
        x_pad, A_pad, B_pad, C_pad = self._chunk_and_pad(x, dt_proc, A, B_exp, C_exp, pad_size)

        # Core computations
        Y_diag, states, A_cum = self._compute_diagonal_blocks(A_pad, B_pad, C_pad, x_pad)
        Y_off, final_state = self._compute_inter_chunk_blocks(A_cum, C_pad, states, initial_states)

        # Output synthesis
        state_opts.final_state = final_state
        return self._synthesize_output((Y_diag, Y_off, D), (pad_size, seq_len), state_opts)

    def _expand_dims(self, x, A, dt, B, C):
        x = rearrange(x, "b l (h p) -> b l h p", p=self.config['headdim']).contiguous()
        dt = dt.contiguous()
        A = A.contiguous()
        B = rearrange(B, "b l (g n) -> b l g n", n=self.config['d_state']).contiguous()
        C = rearrange(C, "b l (g n) -> b l g n", n=self.config['d_state']).contiguous()
        return x, dt, A, B, C

    def _prepare_initial_states(self, states: Optional[torch.Tensor]) -> torch.Tensor:
        """State initialization"""
        return rearrange(states, "b n h p -> b 1 n h p") if states is not None else None

    def _calculate_padding(self, seq_len: int) -> int:
        """Calculate padding length"""
        return self.config['chunk_size'] - (seq_len % self.config['chunk_size'])

    def _expand_groups_to_heads(self, B, C):
        """Dimension expansion: groups -> heads"""
        B_exp = repeat(B, "b l g d -> b l (g h) d", h=self.h_ratio)
        C_exp = repeat(C, "b l g d -> b l (g h) d", h=self.h_ratio)
        return B_exp, C_exp

    def _process_time_step(self, dt):
        """Time step parameter processing"""
        dt_proc = nn.functional.softplus(dt + self.config['dt_bias'])
        return torch.clamp(dt_proc, self.config['dt_min'], self.config['dt_max'])

    def _prepare_residual(self, D, x, pad_size):
        """Residual connection preparation"""
        D = rearrange(D.float(), "(h p) -> h p", p=self.config['headdim']) \
                    if self.config['D_has_hdim'] else D
        return rearrange(D, "h -> 1 1 h 1") * self._pad_sequence(x, pad_size)

    def _chunk_and_pad(self, x, dt, A, B, C, pad_size):
        """Chunking and padding operations"""
        # Discretization
        x = x * dt.unsqueeze(-1)
        A = A * dt

        # Padding and chunking
        x, A, B, C = [
            rearrange(
                tensor=self._pad_sequence(tensor, pad_size),
                pattern="b (c l) ... -> b c l ...",
                l=self.config['chunk_size'] 
            )
            for tensor in (x, A, B, C)
        ]
        return x, A, B, C
        
    def _compute_diagonal_blocks(self, A, B, C, x):
        """Diagonal block computation"""
        A = rearrange(A, "b c l h -> b h c l")
        A_cum = torch.cumsum(A, dim=-1)
        L = torch.exp(self._segmented_sum(A)).to(torch.bfloat16)

        # Diagonal term calculation
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, x)

        # State initialization
        decay = torch.exp(A_cum[:, :, :, -1:] - A_cum).to(torch.bfloat16)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay, x)
        return Y_diag, states, A_cum

    def _compute_inter_chunk_blocks(self, A, C, states, initial_states):
        """Inter-chunk computation"""
        # State propagation
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay = torch.exp(self._segmented_sum(nn.functional.pad(A[:, :, :, -1], (1, 0)))).to(torch.bfloat16)
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

        # State transformation to output
        state_decay = torch.exp(A).to(torch.bfloat16)
        return torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay), final_state

    def _synthesize_output(
        self,
        y_parts: tuple,          # (Y_diag, Y_off, D)
        seq_meta: tuple,         # (pad_size, seq_len)
        state_opts: StateOptions # Contains final_state and return controls
    ):
        """Unpack tuples"""
        Y_diag, Y_off, D = y_parts
        pad_size, seq_len = seq_meta

        """Get parameters from state options"""
        final_state = state_opts.final_state
        return_final = state_opts.should_return_final

        """Output synthesis"""
        y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        y = y + D

        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        if not return_final:
            return y
        else:
            return y, final_state

    def _pad_sequence(self, x, pad_size=0):
        """Padding handling"""
        if not 2 < len(x.shape) < 5:
            raise AssertionError('len(x.shape) must in range(2, 5)')
        
        pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(x.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

        return nn.functional.pad(x, pad_shape, mode="constant", value=0)

    def _segmented_sum(self, x):
        """Numerically stable segmented summation"""
        T = x.size(-1)
        x = repeat(x, "... d -> ... d e", e=T)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
        x = x.masked_fill(~mask, 0)
        x_segsum = torch.cumsum(x, dim=-2)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum