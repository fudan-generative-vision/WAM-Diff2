from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

# try:
#     from .scheduler.scheduler import ConvexScheduler, SchedulerOutput
#     from .utils import expand_tensor_like, unsqueeze_to_match
# except:
#     from dllm.pipelines.qwen.scheduler.scheduler import ConvexScheduler, SchedulerOutput
#     from dllm.pipelines.qwen.utils import expand_tensor_like, unsqueeze_to_match

from wam_diff.utils.scheduler import ConvexScheduler, SchedulerOutput
from wam_diff.utils.helper import expand_tensor_like, unsqueeze_to_match

@dataclass
class PathSample:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample :math:`t`.
        x_t (Tensor): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (Tensor): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).
    """
    x_1: torch.Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: torch.Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: torch.Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: torch.Tensor = field(metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."})
    dx_t: torch.Tensor = field(metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."})


@dataclass
class DiscretePathSample:
    """
    Represents a sample of a conditional-flow generated discrete probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample  :math:`t`.
        x_t (Tensor): the sample along the path  :math:`X_t ~ p_t`.
    """
    x_1: torch.Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: torch.Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: torch.Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: torch.Tensor = field(metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."})


class ProbPath(ABC):
    r"""Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        for x_0, x_1 in dataset:
            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Optimizes the model. The loss function varies, depending on model and path.
            loss(path_sample, my_model(x_t, t)).backward()
    """

    @abstractmethod
    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> PathSample:
        r"""Sample from an abstract probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        | returns :math:`X_0, X_1, X_t \sim p_t(X_t)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample.
        """
        ...

    def assert_sample_shape(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        # assert (
        #     t.ndim == 1
        # ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"


class MixtureDiscreteProbPath(ProbPath):
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    .. math::

        P(X_t = X_0) = \sigma_t \quad \text{and} \quad  P(X_t = X_1) = 1 - \sigma_t,

    where :math:`\sigma_t` is provided by the scheduler.

    Example:

    .. code-block:: python

        >>> x_0 = torch.zeros((1, 3, 3))
        >>> x_1 = torch.ones((1, 3, 3))

        >>> path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.1])).x_t
        >>> result
        tensor([[[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.5])).x_t
        >>> result
        tensor([[[1.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([1.0])).x_t
        >>> result
        tensor([[[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]]])

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\sigma_t`.
    """

    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(
            scheduler, ConvexScheduler
        ), "Scheduler for ConvexProbPath must be a ConvexScheduler."

        self.scheduler = scheduler

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> DiscretePathSample:
        r"""Sample from the affine probability path:
            | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
            | return :math:`X_0, X_1, t`, and :math:`X_t \sim p_t`.
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t
        if sigma_t.ndim == 1:
            sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1) # [B, L]

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t # [B, L]
        x_t = torch.where(source_indices, x_0, x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(self, posterior_logits: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Convert the factorized posterior to velocity.

        | given :math:`p(X_1|X_t)`. In the factorized case: :math:`\prod_i p(X_1^i | X_t)`.
        | return :math:`u_t`.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0, 1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1) # [b, l, vocab_size]
        vocab_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocab_size) # [b, l]
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)
        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)


# class MixtureDiscreteSoftmaxProbPath(ProbPath):
#     def __init__(
#         self,
#         embedding_path,
#         device: str = "cuda",
#         mode: str = "text",
#     ):
#         self.a = 1.0
#         self.c = 2.5
#         self.eps = 0.03
#         self.device = device
#         assert mode in ['image', 'text'], f"Unsupported mode probability path: {mode}"
#         self.embedding = self.get_embedding(embedding_path).to(self.device) # [vocab_size, feat_dim]
#         if isinstance(self.embedding, torch.Tensor):
#             self.embedding = F.normalize(self.embedding, p=2, dim=-1)
#         elif isinstance(self.embedding, nn.Embedding):
#             self.embedding.weight.requires_grad = False
#             self.embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
#         torch.cuda.empty_cache()

#     def get_embedding(self, embedding_path):
#         embedding = torch.load(embedding_path, map_location="cpu")
#         return embedding

#     # def metric(self, emb_z):
#     #     emb_z_flat = emb_z.view(-1, emb_z.shape[-1]) # [b * l, feat_dim]
#     #     emb_z_norm = F.normalize(emb_z_flat, p=2, dim=-1) # [b * l, feat_dim]
#     #     # [b * l, 1] + [vocab_size] = [b * l, vocab_size]
#     #     # [b * l, vocab_size] - [b * l, vocab_size] = [b * l, vocab_size]
#     #     distance = (torch.sum(emb_z_norm ** 2, dim=1, keepdim=True) + \
#     #                 torch.sum(self.embedding ** 2, dim=1) - \
#     #                 torch.einsum('bd,dn->bn', emb_z_norm, torch.einsum('n d->d n', self.embedding)) * 2) ** 1
#     #     return distance

#     def metric(self, emb_z):
#         """
#         [Kinetic-Optimal 核心]
#         距离必须定义为 Squared Euclidean Distance (||x-y||^2)。
#         这是 Gaussian Path 的对数概率项，也是 Kinetic Energy 的形式。
#         """
#         emb_z_flat = emb_z.view(-1, emb_z.shape[-1]) # [b*l, feat_dim]
#         emb_z_norm = F.normalize(emb_z_flat, p=2, dim=-1, eps=1e-6)

#         # Squared Euclidean = 2 - 2 * Cosine
#         #[b*l, feat_dim] * [feat_dim, vocab_size] = [b*l, vocab_size]
#         cosine_sim = torch.matmul(emb_z_norm, self.embedding.transpose(0, 1))
#         dist_squared = 2 * (1 - cosine_sim)

#         # return torch.clamp(dist_squared, min=0.0)
#         return dist_squared ** 2

#     def scheduler(self, t: torch.Tensor):
#         # 1. 计算 beta_t (温度倒数)
#         # beta_t = c * (t / (1-t))^a
#         # term = t / (1 - t)
#         # beta_t = self.c * (term ** self.a)

#         # 2. 计算 d_beta_t (权重/速率)
#         # d_beta/dt = c * a * t^(a-1) / (1-t)^(a+1)
#         numerator = self.c * self.a * (t ** (self.a - 1)) * (1 + self.eps)
#         denominator = (1 - t + self.eps) ** (self.a + 1)
#         d_beta_t = numerator / denominator

#         return SchedulerOutput(d_alpha_t=d_beta_t, alpha_t=torch.zeros_like(t))

#     def get_prob_distribution(self, emb, t, return_beta=False):
#         b, l = emb.shape[:2]
#         dist = self.metric(emb).reshape(b, l, -1) # [b, l, vocab_size]

#         beta_t = self.c * ((t / (1 - t + self.eps)) ** self.a)
#         if beta_t.shape[0] == b:
#             beta_t = beta_t.reshape(b, 1, 1) # [b, 1, 1]
#         dist = torch.softmax(dist * (-1) * beta_t, dim=-1) # [b, l, vocab_size]
#         if return_beta:
#             return dist, beta_t
#         else:
#             return dist

#     def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> DiscretePathSample:
#         x_1_embed = self.embedding[x_1] # [b, l, feat_dim]
#         prob_x_t = self.get_prob_distribution(x_1_embed, t) # [b, l, vocab_size]
#         b, l = prob_x_t.shape[:2]
#         x_t = torch.multinomial(prob_x_t.reshape(b * l, -1), num_samples=1, replacement=False).reshape(b, l) # [b, l]
#         return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)


class MixtureDiscreteSoftmaxProbPath(ProbPath):
    def __init__(
        self,
        embedding_path,
        device: str = "cuda",
        mode: str = "text",
    ):
        self.a = 1.0
        self.c = 2.5
        self.eps = 0.03
        self.device = device
        assert mode in ['image', 'text'], f"Unsupported mode probability path: {mode}"
        self.embedding = self.get_embedding(embedding_path).to(self.device) # [vocab_size, feat_dim]
        if isinstance(self.embedding, torch.Tensor):
            self.embedding = F.normalize(self.embedding, p=2, dim=-1)
        elif isinstance(self.embedding, nn.Embedding):
            self.embedding.weight.requires_grad = False
            self.embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        torch.cuda.empty_cache()

    def get_embedding(self, embedding_path):
        embedding = torch.load(embedding_path, map_location="cpu")
        return embedding

    # def metric(self, emb_z):
    #     emb_z_flat = emb_z.view(-1, emb_z.shape[-1]) # [b * l, feat_dim]
    #     emb_z_norm = F.normalize(emb_z_flat, p=2, dim=-1) # [b * l, feat_dim]
    #     # [b * l, 1] + [vocab_size] = [b * l, vocab_size]
    #     # [b * l, vocab_size] - [b * l, vocab_size] = [b * l, vocab_size]
    #     distance = (torch.sum(emb_z_norm ** 2, dim=1, keepdim=True) + \
    #                 torch.sum(self.embedding ** 2, dim=1) - \
    #                 torch.einsum('bd,dn->bn', emb_z_norm, torch.einsum('n d->d n', self.embedding)) * 2) ** 1
    #     return distance

    def metric(self, emb_z):
        """
        [Kinetic-Optimal 核心]
        距离必须定义为 Squared Euclidean Distance (||x-y||^2)。
        这是 Gaussian Path 的对数概率项，也是 Kinetic Energy 的形式。
        """
        emb_z_flat = emb_z.view(-1, emb_z.shape[-1]) # [b*l, feat_dim]
        emb_z_norm = F.normalize(emb_z_flat, p=2, dim=-1, eps=1e-6)

        # Squared Euclidean = 2 - 2 * Cosine
        #[b*l, feat_dim] * [feat_dim, vocab_size] = [b*l, vocab_size]
        cosine_sim = torch.matmul(emb_z_norm, self.embedding.transpose(0, 1))
        dist_squared = 2 * (1 - cosine_sim)

        # return torch.clamp(dist_squared, min=0.0)
        return dist_squared ** 2

    def scheduler(self, t: torch.Tensor):
        # 1. 计算 beta_t (温度倒数)
        # beta_t = c * (t / (1-t))^a
        # term = t / (1 - t)
        # beta_t = self.c * (term ** self.a)

        # 2. 计算 d_beta_t (权重/速率)
        # d_beta/dt = c * a * t^(a-1) / (1-t)^(a+1)
        numerator = self.c * self.a * (t ** (self.a - 1)) * (1 + self.eps)
        denominator = (1 - t + self.eps) ** (self.a + 1)
        d_beta_t = numerator / denominator

        return SchedulerOutput(d_alpha_t=d_beta_t, alpha_t=torch.zeros_like(t))

    def get_prob_distribution(self, emb, t, return_beta=False):
        b, l = emb.shape[:2]
        dist = self.metric(emb).reshape(b, l, -1) # [b, l, vocab_size]

        # beta_t = self.c * ((t / (1 - t + self.eps)) ** self.a)
        beta_t = 32 ** t
        if beta_t.shape[0] == b:
            beta_t = beta_t.reshape(b, 1, 1) # [b, 1, 1]
        dist = torch.softmax(dist * (-1) * beta_t, dim=-1) # [b, l, vocab_size]
        if return_beta:
            return dist, beta_t
        else:
            return dist

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> DiscretePathSample:
        x_1_embed = self.embedding[x_1] # [b, l, feat_dim]
        prob_x_t = self.get_prob_distribution(x_1_embed, t) # [b, l, vocab_size]
        b, l = prob_x_t.shape[:2]
        x_t = torch.multinomial(prob_x_t.reshape(b * l, -1), num_samples=1, replacement=False).reshape(b, l) # [b, l]
        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)
