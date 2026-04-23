import torch
import numpy as np
import os

from torch._C import device

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions,
                 input_device, output_device='cpu', frame_stack=4):
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")

        if override in ["cpu", "cuda:0", "cuda:1"]:
            print("Received replay buffer memory override.")
            self.input_device = override
        else:
            self.input_device  = input_device
        
        print(f"Replay buffer memory on: {self.input_device}")

        self.output_device = output_device

        # States (uint8 saves ~4× RAM vs float32)
        self.state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )
        self.next_state_memory      = torch.zeros(
            (max_size, *input_shape), dtype=torch.uint8, device=self.input_device
        )

        # Actions as scalar indices for torch.gather
        self.action_memory  = torch.zeros(max_size, dtype=torch.int64,
                                          device=self.input_device)
        self.reward_memory  = torch.zeros(max_size, dtype=torch.float32,
                                          device=self.input_device)
        self.terminal_memory = torch.zeros(max_size, dtype=torch.bool,
                                           device=self.input_device)

    # ------------------------------------------------------------------ #

    def can_sample(self, batch_size: int) -> bool:
        """Require at least 5×batch_size transitions before sampling."""
        return self.mem_ctr >= batch_size * 10

    # ------------------------------------------------------------------ #

    def store_transition(self, state, action, reward, next_state, done):
        """Write a transition in-place on `input_device`."""
        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx]      = torch.as_tensor(
            state, dtype=torch.uint8, device=self.input_device)
        self.next_state_memory[idx] = torch.as_tensor(
            next_state, dtype=torch.uint8, device=self.input_device)

        self.action_memory[idx]   = int(action)
        self.reward_memory[idx]   = float(reward)
        self.terminal_memory[idx] = bool(done)

        self.mem_ctr += 1

    # ------------------------------------------------------------------ #

    def sample_buffer(self, batch_size):
        """Return tensors ready for training (on `output_device`)."""
        max_mem = min(self.mem_ctr, self.mem_size)
        batch   = torch.randint(0, max_mem, (batch_size,),
                                device=self.input_device, dtype=torch.int64)

        # Cast / move once, right here
        states      = self.state_memory[batch]     \
                        .to(self.output_device, dtype=torch.float32)
        next_states = self.next_state_memory[batch]\
                        .to(self.output_device, dtype=torch.float32)
        rewards     = self.reward_memory[batch].to(self.output_device)
        dones       = self.terminal_memory[batch].to(self.output_device)

        # **Return actions as 1-D (B,) LongTensor — caller will unsqueeze**
        actions     = self.action_memory[batch].to(self.output_device)

        return states, actions, rewards, next_states, dones

    def can_sample_sequences(self, batch_size: int, seq_len: int) -> bool:
        """Require enough consecutive transitions to fill at least 10 sequences."""
        filled = min(self.mem_ctr, self.mem_size)
        return filled >= seq_len * 10

    def sample_sequences(self, batch_size: int, seq_len: int):
        """
        Sample batch_size sequences of consecutive transitions.

        Consecutive transitions are needed to train the RSSM: the GRU must
        process a real temporal sequence so gradients flow backward through
        time (BPTT) and the hidden state learns to carry information forward.

        Sampling avoids the write-head boundary so sequences never mix data
        from opposite ends of the circular buffer (which would be temporally
        discontinuous).

        Args:
            batch_size : number of sequences to return
            seq_len    : number of consecutive timesteps per sequence

        Returns:
            states      : (batch_size, seq_len, C, H, W)  uint8 → float32
            actions     : (batch_size, seq_len)            int64
            rewards     : (batch_size, seq_len)            float32
            next_states : (batch_size, seq_len, C, H, W)  uint8 → float32
            dones       : (batch_size, seq_len)            bool
        """
        filled = min(self.mem_ctr, self.mem_size)

        if filled < seq_len:
            raise ValueError(
                f"Buffer has {filled} transitions but seq_len={seq_len} required."
            )

        if self.mem_ctr <= self.mem_size:
            # Buffer has not yet wrapped — all indices 0…filled-1 are valid
            # and temporally ordered, so any contiguous window is safe.
            max_start_index = filled - seq_len
            start_indices   = torch.randint(
                0, max_start_index + 1, (batch_size,), device=self.input_device
            )
        else:
            # Buffer is full and wrapping.  The write head sits at the oldest
            # data boundary.  To avoid sampling across that boundary we start
            # all sequences at or after the write head and stay within
            # (mem_size - seq_len) steps forward, which guarantees no sequence
            # crosses the boundary.
            write_head_index = self.mem_ctr % self.mem_size
            valid_range      = self.mem_size - seq_len
            random_offsets   = torch.randint(
                0, valid_range, (batch_size,), device=self.input_device
            )
            start_indices = (write_head_index + random_offsets) % self.mem_size

        # Build all per-step indices: shape (batch_size, seq_len)
        step_offsets = torch.arange(seq_len, device=self.input_device)
        all_indices  = (start_indices.unsqueeze(1) + step_offsets.unsqueeze(0)) % self.mem_size

        # Gather and move to output device in one shot
        states      = self.state_memory[all_indices].to(self.output_device, dtype=torch.float32)
        next_states = self.next_state_memory[all_indices].to(self.output_device, dtype=torch.float32)
        actions     = self.action_memory[all_indices].to(self.output_device)
        rewards     = self.reward_memory[all_indices].to(self.output_device)
        dones       = self.terminal_memory[all_indices].to(self.output_device)

        return states, actions, rewards, next_states, dones

    def print_stats(self):
        filled = min(self.mem_ctr, self.mem_size)
        tensors = [self.state_memory, self.next_state_memory,
                   self.action_memory, self.reward_memory, self.terminal_memory]
        used_bytes  = sum(t.element_size() * t.numel() * filled / self.mem_size for t in tensors)
        total_bytes = sum(t.element_size() * t.numel() for t in tensors)
        print(f"{filled} memories loaded | "
              f"used: {used_bytes / 1e9:.3f} GB / {total_bytes / 1e9:.3f} GB")
