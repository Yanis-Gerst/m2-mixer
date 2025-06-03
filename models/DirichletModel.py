from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy


class DirichletModel(pl.LightningModule):
    def __init__(self, model, num_classes=42, dropout=0.):
        super(DirichletModel, self).__init__()
        self.num_classes = num_classes
        self.model = model(num_classes=num_classes,
                           monte_carlo=False, dropout=dropout, dirichlet=True)
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.criterion = AvgTrustedLoss(num_views=3)
        self.aleatoric_uncertainties = None
        self.epistemic_uncertainties = None

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.log('train_loss', loss)
        acc = self.train_acc(output, target)
        self.log('train_acc_step', acc, prog_bar=True)
        return loss

    def shared_step(self, batch):
        image, audio, text, target = batch
        output_a, output = self((image, audio, text))
        output = torch.stack(output)
        loss = self.criterion(output, target, output_a)
        return loss, output_a, target

    def validation_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.val_acc(output, target)
        alphas = output + 1
        probs = alphas / alphas.sum(dim=-1, keepdim=True)
        entropy = self.num_classes / alphas.sum(dim=-1)
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        aleatoric_uncertainty = - \
            torch.sum(probs * (torch.digamma(alphas + 1) -
                      torch.digamma(alpha_0 + 1)), dim=-1)
        return loss, output, target, entropy, aleatoric_uncertainty

    def test_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch)
        self.test_acc(output, target)
        alphas = output + 1
        probs = alphas / alphas.sum(dim=-1, keepdim=True)
        entropy = self.num_classes / alphas.sum(dim=-1)
        alpha_0 = alphas.sum(dim=-1, keepdim=True)
        aleatoric_uncertainty = - \
            torch.sum(probs * (torch.digamma(alphas + 1) -
                      torch.digamma(alpha_0 + 1)), dim=-1)
        return loss, output, target, entropy, aleatoric_uncertainty

    def training_epoch_end(self, outputs):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.criterion.annealing_step += 1

    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.val_acc.compute(), prog_bar=True)
        self.log('val_loss', np.mean(
            [x[0].detach().cpu().numpy() for x in outputs]), prog_bar=True)
        self.log('val_entropy', torch.cat(
            [x[3] for x in outputs]).mean(), prog_bar=True)
        self.log('val_sigma', torch.cat([x[4]
                 for x in outputs]).mean(), prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.log('test_entropy_epi', torch.cat([x[3] for x in outputs]).mean())
        self.log('test_ale', torch.cat([x[4] for x in outputs]).mean())
        self.aleatoric_uncertainties = torch.cat(
            [x[4] for x in outputs]).detach().cpu().numpy()
        self.epistemic_uncertainties = torch.cat(
            [x[3] for x in outputs]).detach().cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class MCDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(MCDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return torch.nn.functional.dropout(x, p=self.p, training=True)


class AleatoricClassificationLoss(torch.nn.Module):
    def __init__(self, num_samples=100):
        super(AleatoricClassificationLoss, self).__init__()
        self.num_samples = num_samples

    def forward(self, logits, targets, log_std):
        return aleatoric_loss(logits, targets, log_std, num_samples=self.num_samples)


def aleatoric_loss(logits, targets, log_std, num_samples=100):
    # std = torch.exp(log_std)
    std = log_std
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples,
                        device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise

    targets = targets.unsqueeze(-1).repeat(*
                                           [1] * len(logits.shape), num_samples).squeeze(0)
    mc_x = torch.nn.functional.cross_entropy(prd, targets, reduction='none')
    # mean across mc samples
    mc_x = mc_x.mean(-1)
    # mean across every thing else
    mc_x_mean = mc_x.mean()
    # assert is not inf or nan
    assert not torch.isfinite(mc_x_mean).sum(
    ) == 0, f"Loss is inf: {mc_x_mean}"
    return mc_x.mean()


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num,
                    num_classes, annealing_step, device)
    return torch.mean(loss)


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


def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        # (num_views, batch_size)
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum


class AvgTrustedLoss(nn.Module):
    def __init__(self, num_views: int, annealing_start=50, gamma=1):
        super(AvgTrustedLoss, self).__init__()
        self.num_views = num_views
        self.annealing_step = 0
        self.annealing_start = annealing_start
        self.gamma = gamma

    def forward(self, evidences, target, evidence_a, **kwargs):
        num_classes = evidences.shape[-1]
        target = F.one_hot(target, num_classes)
        alphas = evidences + 1
        loss_acc = edl_digamma_loss(alphas, target, self.annealing_step, num_classes, self.annealing_start,
                                    evidence_a.device)
        for v in range(len(evidences)):
            alpha = evidences[v] + 1
            loss_acc += edl_digamma_loss(alpha, target, self.annealing_step, num_classes, self.annealing_start,
                                         evidence_a.device)
        loss_acc = loss_acc / (len(evidences) + 1)
        loss = loss_acc + self.gamma * \
            get_dc_loss(evidences, evidence_a.device)
        return loss


def sampling_softmax(logits, log_sigma, num_samples=100):
    std = torch.exp(log_sigma)
    mu_mc = logits.unsqueeze(-1).repeat(*[1] * len(logits.shape), num_samples)
    # hard coded the known shape of the data
    noise = torch.randn(*logits.shape, num_samples,
                        device=logits.device) * std.unsqueeze(-1)
    prd = mu_mc + noise
    return torch.softmax(prd, dim=0).mean(-1)


def compute_uncertainty(outputs, log_sigmas_ale, log_sigmas_ep, num_samples=100):
    p_ale = sampling_softmax(outputs, log_sigmas_ale, num_samples)
    entropy_ale = -torch.sum(p_ale * torch.log(p_ale + 1e-6), dim=-1)
    p_ep = sampling_softmax(outputs, log_sigmas_ep, num_samples)
    entropy_ep = -torch.sum(p_ep * torch.log(p_ep + 1e-6), dim=-1)
    return entropy_ale, entropy_ep
