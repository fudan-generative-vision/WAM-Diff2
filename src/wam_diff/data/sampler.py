import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class BalanceSampler(Sampler):
    def __init__(
        self,
        lengths,
        v_tokens,
        local_batch_size,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed:int = 0,
        drop_last=False,
        **kwargs,
    ):
        self.lengths = np.array(lengths)
        self.v_tokens = np.array(v_tokens)
        self.local_batch_size = local_batch_size

        self.num_replicas = num_replicas
        if self.num_replicas is None:
            self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1

        self.rank = rank
        if self.rank is None:
            self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.world_batch_size = local_batch_size * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._build_steps()

    def _build_steps(self):
        # 1. 区分 vl 和 text samples.
        is_vl = self.v_tokens > 0
        vl_indices = np.where(is_vl)[0]
        tx_indices = np.where(~is_vl)[0]

        # 2. 桶内按长度排序
        vl_indices_sorted = vl_indices[np.argsort(self.lengths[vl_indices])]
        tx_indices_sorted = tx_indices[np.argsort(self.lengths[tx_indices])]

        # 3. 分别打包成超级批次 (super-batch)
        all_super_batches = []

        def pack_to_super_batches(indices):
            num_extra = len(indices) % self.world_batch_size
            if num_extra > 0:
                if self.drop_last:
                    indices = indices[:-num_extra]
                else:
                    # 桶内补齐，防止类型混杂
                    indices = np.concatenate([indices, indices[:self.world_batch_size - num_extra]])
            return indices.reshape(-1, self.world_batch_size)

        if len(vl_indices_sorted) > 0:
            all_super_batches.append(pack_to_super_batches(vl_indices_sorted))
        if len(tx_indices_sorted) > 0:
            all_super_batches.append(pack_to_super_batches(tx_indices_sorted))

        # 垂直拼接所有超级批次
        self.global_steps = np.vstack(all_super_batches)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 打乱的是步的顺序。这样 step 1 可能是全员 vl samples，step 2 可能是全员 text samples.
        if self.shuffle:
            order = torch.randperm(len(self.global_steps), generator=g).tolist()
        else:
            order = range(len(self.global_steps))

        for step_idx in order:
            global_batch = self.global_steps[step_idx]
            start = self.rank * self.local_batch_size
            yield global_batch[start : start + self.local_batch_size].tolist()

    def __len__(self):
        return len(self.global_steps)

    def set_epoch(self, epoch):
        self.epoch = epoch
