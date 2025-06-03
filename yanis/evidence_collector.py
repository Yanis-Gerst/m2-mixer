import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes, aggregation='average', activation='softplus'):
        super(EvidenceCollector, self).__init__()
        self.aggregation = aggregation
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        self.activation = activation
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))
        # self.net.append(nn.Softplus())

    def activation_function(self, h):
        if self.activation == 'softplus':
            return softplus(h)
        # Compute log(1e13) accurately
        log1e13 = 13 * \
            torch.log(torch.tensor(10.0, dtype=h.dtype, device=h.device))

        # Numerator in log-space
        numerator = h + log1e13

        # Denominator in log-space using logaddexp for numerical stability
        denominator = torch.logaddexp(h, log1e13)

        # Compute the log of the function
        log_f = numerator - denominator

        # Exponentiate to get the final result
        return torch.exp(log_f)
