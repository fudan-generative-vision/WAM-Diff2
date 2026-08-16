import torch
import time

def unsqueeze_to_match(source: torch.Tensor, target: torch.Tensor, how: str = "suffix") -> torch.Tensor:
    return source

def expand_tensor_like(input_tensor: torch.Tensor, expand_to: torch.Tensor):
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)

class GPUTimer:
    def __init__(self, enabled=False, rank=0):
        self.enabled = enabled
        self.rank = rank
        self.start_event = torch.cuda.Event(enable_timing=True) if enabled else None
        self.end_event = torch.cuda.Event(enable_timing=True) if enabled else None

    def start(self, label):
        if not self.enabled: return
        self.label = label
        self.start_event.record()

    def stop(self):
        if not self.enabled: return
        self.end_event.record()
        torch.cuda.synchronize()  # 强制 GPU 完成计算，以便立即打印
        elapsed = self.start_event.elapsed_time(self.end_event)
        if self.rank == 0:
            print(f"[Profiling] {self.label:<40} : {elapsed:>10.2f} ms")