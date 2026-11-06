"""Paper-based MIL comparators. See SOURCES.md for versions, licenses, and adaptations.

CLAM logic is adapted from Mahmood Lab's GPL-3.0 implementation. DTFD-MIL and
ProMIL formulations follow their MIT-licensed author implementations. This
combined adaptation is distributed under GPL-3.0; upstream notices accompany it.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GatedAttention(nn.Module):
    def __init__(self, width, hidden, dropout=0.0):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(width, hidden), nn.Sigmoid(), nn.Dropout(dropout))
        self.score = nn.Linear(hidden, 1)

    def forward(self, features):
        weights = self.score(self.v(features) * self.u(features)).transpose(0, 1).softmax(-1)
        return weights @ features, weights.squeeze(0)


def smooth_top1(logits, labels):
    """Smooth top-1 SVM (margin=1, temperature=1), using stable logsumexp."""
    margin = 1 - F.one_hot(labels, logits.shape[-1]).to(logits.dtype)
    scores = logits + margin - logits.gather(1, labels[:, None])
    return torch.logsumexp(scores, dim=1).mean()


class CLAM(nn.Module):
    """CLAM-SB with gated attention and class-specific instance clustering."""
    def __init__(self, num_classes, width=1024, k=8, dropout=.25, bag_weight=.7):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttention(512, 256, dropout)
        self.clf = nn.Linear(512, num_classes)
        self.instance_clfs = nn.ModuleList(nn.Linear(512, 2) for _ in range(num_classes))
        self.k, self.bag_weight = k, bag_weight
        self.subtyping = num_classes > 2

    def forward(self, features, label=None):
        h = self.project(features)
        pooled, weights = self.attention(h)
        logits = self.clf(pooled)
        auxiliary = h.sum() * 0
        # Disjoint top/bottom subsets prevent contradictory pseudo-labels in small bags.
        k = min(self.k, len(h) // 2)
        if label is not None and k:
            order = torch.argsort(weights, descending=True, stable=True)
            top, bottom = order[:k], order[-k:]
            instance_losses = []
            for class_id, clf in enumerate(self.instance_clfs):
                if class_id == int(label):
                    instances = torch.cat([h[top], h[bottom]])
                    targets = torch.cat([torch.ones(k, device=h.device), torch.zeros(k, device=h.device)]).long()
                elif self.subtyping:
                    instances, targets = h[top], torch.zeros(k, device=h.device, dtype=torch.long)
                else:
                    continue
                instance_losses.append(smooth_top1(clf(instances), targets))
            auxiliary = torch.stack(instance_losses).mean()
        return logits, auxiliary, k

    def objective(self, features, label):
        logits, instance_loss, k = self(features, label)
        bag_loss = F.cross_entropy(logits, label.reshape(1))
        loss = self.bag_weight * bag_loss + (1-self.bag_weight) * instance_loss if k else bag_loss
        return loss, logits.softmax(-1), dict(bag_loss=bag_loss.item(), instance_loss=instance_loss.item(), selected_k=k)


class DTFDAFS(nn.Module):
    """Double-tier MIL with aggregated feature selection (AFS), never MaxS/MaxMinS."""
    def __init__(self, num_classes, width=1024, groups=4):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(width, 512, bias=False), nn.ReLU())
        self.attention1 = GatedAttention(512, 128)
        self.clf1 = nn.Linear(512, num_classes)
        self.attention2 = GatedAttention(512, 128)
        self.clf2 = nn.Linear(512, num_classes)
        self.groups = groups

    def tier1_parameters(self):
        return list(self.project.parameters()) + list(self.attention1.parameters()) + list(self.clf1.parameters())

    def tier2_parameters(self):
        return list(self.attention2.parameters()) + list(self.clf2.parameters())

    def distill(self, features, generator=None):
        # Match randomized pseudo-bags; cap group count to avoid empty groups.
        order = torch.randperm(len(features), generator=generator, device='cpu').to(features.device)
        groups = torch.tensor_split(order, min(self.groups, len(features)))
        h = self.project(features)
        distilled = torch.cat([self.attention1(h[indices])[0] for indices in groups])
        return self.clf1(distilled), distilled

    def tier2(self, distilled):
        return self.clf2(self.attention2(distilled)[0])

    def train_step(self, features, label, optimizers, clip=5.0):
        lower, upper = optimizers
        lower.zero_grad(set_to_none=True)
        upper.zero_grad(set_to_none=True)
        pseudo_logits, distilled = self.distill(features)
        pseudo_loss = F.cross_entropy(pseudo_logits, label.expand(len(pseudo_logits)))
        if not torch.isfinite(pseudo_loss):
            raise ValueError('Non-finite DTFD pseudo-bag loss')
        # The author's optimizer updates the first tier only from its pseudo-bag loss.
        # Detaching makes that separation explicit and avoids stepping a retained graph.
        upper_features = distilled.detach()
        pseudo_loss.backward()
        for module in (self.project, self.attention1, self.clf1):
            nn.utils.clip_grad_norm_(module.parameters(), clip)
        lower.step()
        logits = self.tier2(upper_features)
        bag_loss = F.cross_entropy(logits, label.reshape(1))
        if not torch.isfinite(bag_loss):
            raise ValueError('Non-finite DTFD bag loss')
        bag_loss.backward()
        nn.utils.clip_grad_norm_(self.tier2_parameters(), clip)
        upper.step()
        return bag_loss + pseudo_loss.detach(), logits.softmax(-1).detach(), dict(bag_loss=bag_loss.item(), pseudo_loss=pseudo_loss.item())

    def predict(self, features, seed, repeats=1):
        generator = torch.Generator().manual_seed(seed)
        outputs = []
        for _ in range(repeats):
            _, distilled = self.distill(features, generator)
            outputs.append(self.tier2(distilled).softmax(-1))
        return torch.stack(outputs).mean(0)


class LeNet(nn.Module):
    """LeNet-style MNIST encoder (20/50 filters, 500-dimensional embedding)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(20, 50, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(50*4*4, 500), nn.ReLU())

    def forward(self, images):
        return self.fc(self.conv(images))


def bernstein_logs(instance_logits, estimator):
    """Author-code ordering: ascending scores, alpha^(n-k)*(1-alpha)^k.

    Both probability and complement are aggregated in log space. This avoids
    subtracting a rounded probability of one and supports bags of length one.
    """
    ordered = instance_logits.sort(dim=0).values
    length = len(ordered)
    k = torch.arange(length, dtype=ordered.dtype, device=ordered.device)[:, None]
    n = ordered.new_tensor(length - 1)
    log_choose = torch.lgamma(n+1) - torch.lgamma(k+1) - torch.lgamma(n-k+1)
    log_weights = log_choose + (n-k)*F.logsigmoid(estimator) + k*F.logsigmoid(-estimator)
    log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
    log_p = torch.logsumexp(log_weights + F.logsigmoid(ordered), dim=0, keepdim=True)
    log_not_p = torch.logsumexp(log_weights + F.logsigmoid(-ordered), dim=0, keepdim=True)
    return log_p, log_not_p


class ProMIL(nn.Module):
    def __init__(self, encoder, width, num_classes=2, mnist=False, dropout=.2):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        outputs = 1 if num_classes == 2 else num_classes
        self.clf = (nn.Linear(width, outputs) if mnist else
                    nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, outputs)))
        self.estimator = nn.Parameter(torch.full((outputs,), .2))
        self.encoder_frozen = False
        for layer in self.clf.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def set_encoder_frozen(self, frozen):
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, images):
        logits = self.clf(self.encoder(images))
        return bernstein_logs(logits, self.estimator)

    def objective(self, images, label):
        log_p, log_not_p = self(images)
        if self.num_classes == 2:
            target = label.reshape(1, 1).to(log_p.dtype)
            probability = torch.cat([log_not_p.exp(), log_p.exp()], dim=-1)
        else:
            # Explicit one-vs-rest extension; original ProMIL is binary.
            target = F.one_hot(label.reshape(1), self.num_classes).to(log_p.dtype)
            probability = log_p.softmax(-1)
        loss = -(target*log_p + (1-target)*log_not_p).mean()
        return loss, probability, dict(alpha=self.estimator.detach().sigmoid().mean().item())
