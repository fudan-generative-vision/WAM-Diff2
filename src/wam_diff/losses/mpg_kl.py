import torch
import torch.nn as nn
from wam_diff.utils.scheduler import CondOTScheduler
from wam_diff.utils.mixture import MixtureDiscreteProbPath

# MixturePathGeneralizeKL
class MixturePathGeneralizeKL(nn.Module):
    def __init__(
        self,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
        reduction: str = "mean",
        # path: MixtureDiscreteProbPath = None,
    ):
        super().__init__()
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index
        self.reduction = reduction
        # self.path = path
        # HARDCODE
        scheduler = CondOTScheduler()
        self.path = MixtureDiscreteProbPath(scheduler=scheduler)

    @staticmethod
    def compute_macro_average_loss(per_token_loss, loss_mask, response_mask):
        """计算 Sequence Packing 下的 Macro-Average Loss
        利用response_mask的连续性, 通过检测 0->1 的跳变来区分不同的sample片段。

        Args:
            per_token_loss: [B, L] 或 [N], reduction='none'的loss
            loss_mask: [B, L] 或 [N], 0/1 或 Bool, 1 表示有效的计算loss token
            response_mask:    [B, L] 或 [N], 0/1 或 Bool, 1 表示有效的response token
        Returns:
            final_loss: Scalar
        """
        if per_token_loss.dim() == 1:
            per_token_loss = per_token_loss.unsqueeze(0)
            response_mask = response_mask.unsqueeze(0)

        B, L = per_token_loss.shape
        device = per_token_loss.device
        mask_bool = response_mask.bool()

        # =========================================================
        # 1. 自动生成 Sample IDs
        # =========================================================
        # 我们需要识别每个连续 '1' 片段的起点。
        # 起点定义为：当前是 1，且前一个是 0 (上升沿)；或者是一行的开头且为 1。

        # 1.1 计算同一行内的跳变 (当前位 > 前一位) => 0->1 变 True, 1->0 变 False
        # shape: [B, L-1]
        inner_transitions = mask_bool[:, 1:] & (~mask_bool[:, :-1])

        # 1.2 处理每行的第一列
        # 如果第一列是1，它也是一个片段的开始, shape: [B, 1]
        first_col_starts = mask_bool[:, 0:1]

        # 1.3 拼接得到完整的“片段起点”掩码, shape: [B, L]
        # is_start[b, l]为True 表示这里是一个新sample的开始
        is_start = torch.cat([first_col_starts, inner_transitions], dim=1)

        # 1.4 生成全局唯一ID
        # 展平后计算累加和, 每遇到一个True(新起点)，ID就+1。
        # 这样同一个连续片段内的ID会保持不变 (因为中间全是False, cumsum不增加)。
        # padding(0)部分的ID会跟随上一个片段，但会在后续被mask过滤掉，不影响。
        flat_ids = is_start.view(-1).cumsum(dim=0)

        # 2. 准备数据
        flat_loss = per_token_loss.reshape(-1)
        flat_mask = mask_bool.reshape(-1).float()
        flat_loss_mask = loss_mask.reshape(-1).bool()

        # 3. 确保loss只计算mask=True的部分
        # (虽然padding部分的ID可能重复，但乘上mask后变为 0，不会产生贡献)
        flat_loss = flat_loss * flat_mask
        flat_loss[~flat_loss_mask] = 0

        # 4. 准备scatter容器
        max_id = flat_ids.max().item()
        # 防止全 0 mask 导致 max_id 为负数
        if max_id < 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        num_bins = int(max_id + 1)
        loss_sum_per_sample = torch.zeros(num_bins, device=device)
        valid_len_per_sample = torch.zeros(num_bins, device=device)

        # 5. 分组累加
        loss_sum_per_sample.scatter_add_(0, flat_ids, flat_loss)
        valid_len_per_sample.scatter_add_(0, flat_ids, flat_mask)

        # 6. 计算每个子序列的平均loss, 只有长度>0的才是有效样本
        valid_samples_mask = valid_len_per_sample > 0

        # 计算每个样本内部的mean loss
        loss_mean_per_sample = loss_sum_per_sample / (valid_len_per_sample + 1e-9)

        # 7.取所有有效样本的均值(Macro Average)
        # 如果整个batch都是padding，valid_samples_mask全False，需要处理
        if valid_samples_mask.sum() == 0:
             return torch.tensor(0.0, device=device, requires_grad=True)

        # final_loss = loss_mean_per_sample[valid_samples_mask].mean()
        final_loss = loss_mean_per_sample[valid_samples_mask].sum()

        return final_loss

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,   # gt
        x_t: torch.Tensor,      # sample
        t: torch.Tensor,
        response_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        num_label_tokens: int,
        num_samples: int = 1,
        block_size: int = 0,
    ) -> torch.Tensor:
        x_1 = labels
        b, l = x_1.shape

        if self.fp32_upcast:
            logits = logits.float()

        # extract x_1 value of log(p_{1|t}(x|x_t))
        # 模型觉得变回真值x1的可能性有多大
        log_p_1t = torch.log_softmax(logits, dim=-1) # [b, l, vocab_size].
        log_p_1t_x1 = torch.gather(log_p_1t, dim=-1, index=x_1.unsqueeze(-1)).view(b, l) # log prob [b, l].

        # extract x_t value of p_{1|t}(x|x_t).
        # 模型觉得当前这个词应该保留的概率有多大
        p_1t = torch.exp(log_p_1t)
        p_1t_xt = torch.gather(p_1t, dim=-1, index=x_t.unsqueeze(-1)).view(b, l) # prob [b, l].

        # 计算在时间t，token变化的速率
        # jump_coeff是权重系数。随着时间t变化，模型对于“去噪”的紧迫程度不同，这个系数用来缩放loss的大小。
        scheduler_output = self.path.scheduler(t)
        d_alpha_t = scheduler_output.d_alpha_t
        alpha_t = scheduler_output.alpha_t
        # jump_coeff = (d_alpha_t / (1 - alpha_t))[(...,) + (None,) * (x_1.dim() - 1)] # [b, 1].
        # jump_coeff = jump_coeff.repeat(1, l) # [b, l].
        jump_coeff = d_alpha_t / (1 - alpha_t)
        delta_x1_xt = (x_t == x_1).to(logits.dtype) # 当前词x_t是目标词x_1，则为1，否则为0

        # 1. 当前词x_t已经是正确答案x_1, L \propto 1 - p(x_t|x_t), 如果现在的词是对的，模型应该极度自信地保持现状，不要乱变。
        # 2. 当前词x_t是噪声，不是正确答案x_1, L \propto -p(x_t|x_t) - \log p(x_1|x_t)，把概率都集中到正确答案x_1$上，如果实在没把握猜出x_1，就老实待在原地x_t，不要把概率浪费在其他无关的词y上，不要瞎编。
        # 因为log函数的特性，去噪（奔向x_1）的动力远大于保持（留在x_t）的动力。p(x_t|x_t)这一项更多是起到一个正则化（Regularization）的作用，防止概率泄漏到无关词汇。
        per_token_loss = -jump_coeff * (p_1t_xt - delta_x1_xt + (1 - delta_x1_xt) * log_p_1t_x1)

        if self.reduction == "none":
            return per_token_loss

        # calculate average loss.
        if self.reduction == "mean": # sample-level average loss
            # return self.compute_macro_average_loss(per_token_loss, loss_mask) / accumulation_steps
            return self.compute_macro_average_loss(per_token_loss, loss_mask, response_mask) / num_samples
        elif self.reduction == "sum": # token-level average loss
            per_token_loss = per_token_loss * loss_mask
            loss = per_token_loss.sum() / num_label_tokens
            return loss
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")
