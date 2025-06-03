import torch
from torch import nn


def activation_function(h):
    # # Compute log(1e13) accurately
    # log1e13 = 13 * \
    #     torch.log(torch.tensor(10.0, dtype=h.dtype, device=h.device))

    # # Numerator in log-space
    # numerator = h + log1e13

    # # Denominator in log-space using logaddexp for numerical stability
    # denominator = torch.logaddexp(h, log1e13)

    # # Compute the log of the function
    # log_f = numerator - denominator

    # # Exponentiate to get the final result
    return torch.exp(torch.clamp(h, 0, 10))


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=True):
    # Func torch.log or torch.diagamma in DBF
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * \
        kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


class EDLMSELoss(nn.Module):
    def __init__(self, num_classes, annealing_step, device):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
        self.ohe = torch.eye(self.num_classes)

    def forward(self, output, y, epoch_num):
        self.ohe = self.ohe.to(output.device)
        target = self.ohe[y]
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        )
        # evidence = nn.functional.relu(output)
        evidence = activation_function(output)
        alpha = evidence + 1.
        loss = edl_loss(torch.digamma, target, alpha, epoch_num,
                        self.num_classes, self.annealing_step, self.device)

        # prev_loss = self.squared_error_bayes_risk(evidence, target) + annealing_coef * 0 * self.kl_divergence_loss(evidence,
        #                                                                                                            target)

        # print(
        #     f"{prev_loss}, mean loss {torch.mean(loss)}")
        return torch.mean(loss)

    def squared_error_bayes_risk(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        p = alpha / strength[:, None]
        err = (target - p) ** 2
        var = p * (1 - p) / (strength[:, None] + 1)
        loss = (err + var).sum(dim=-1)
        return loss.mean()

    def kl_divergence_loss(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        n_classes = evidence.shape[-1]
        alpha_tilde = target + (1 - target) * alpha
        strength_tilde = alpha_tilde.sum(dim=-1)
        first = (torch.lgamma(alpha_tilde.sum(dim=-1))
                 - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
                 - (torch.lgamma(alpha_tilde)).sum(dim=-1))

        second = (
            (alpha_tilde - 1) *
            (torch.digamma(alpha_tilde) -
             torch.digamma(strength_tilde)[:, None])
        ).sum(dim=-1)

        loss = first + second

        return loss.mean()


def kl_divergence_loss(evidence: torch.Tensor, target: torch.Tensor):
    alpha = evidence + 1.
    n_classes = evidence.shape[-1]
    alpha_tilde = target + (1 - target) * alpha
    strength_tilde = alpha_tilde.sum(dim=-1)
    first = (torch.lgamma(alpha_tilde.sum(dim=-1))
             - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
             - (torch.lgamma(alpha_tilde)).sum(dim=-1))

    second = (
        (alpha_tilde - 1) *
        (torch.digamma(alpha_tilde) -
         torch.digamma(strength_tilde)[:, None])
    ).sum(dim=-1)

    loss = first + second

    return loss.mean()


class EDLCELoss(nn.Module):
    def __init__(self, num_classes, annealing_step):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.ohe = torch.eye(self.num_classes)

    def forward(self, output, y, epoch_num):
        self.ohe = self.ohe.to(output.device)
        target = self.ohe[y]
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32),
        )
        evidence = nn.functional.relu(output)
        loss = self.cross_entropy_bayes_risk(evidence, target)
        return loss

    def cross_entropy_bayes_risk(self, evidence: torch.Tensor, target: torch.Tensor):
        alpha = evidence + 1.
        strength = alpha.sum(dim=-1)
        loss = (target * (torch.digamma(strength)
                [:, None] - torch.digamma(alpha))).sum(dim=-1)
        return loss.mean()
