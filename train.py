import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Conformer

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # z: (B, D) embeddings
        # y: (B,) class labels
        z = F.normalize(z, dim=1)
        B = z.size(0)

        sim = (z @ z.t()) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values

        logits_mask = torch.ones((B, B), device=z.device, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)

        y = y.view(-1, 1)
        pos_mask = (y == y.t()) & logits_mask

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_cnt = pos_mask.sum(dim=1).clamp_min(1)
        loss = -(log_prob * pos_mask).sum(dim=1) / pos_cnt
        return loss.mean()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conformer().to(device)
    model.train()

    B = 32
    n_chans = 64
    samples = 1250
    num_classes = 4

    # Raw EEG input: (batch, electrodes, time)
    x = torch.randn(B, n_chans, samples, device=device)
    y = torch.randint(0, num_classes, (B,), device=device)

    supcon_lambda = 0.1
    supcon_tau = 0.07

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ce_loss = nn.CrossEntropyLoss()
    supcon_loss = SupConLoss(temperature=supcon_tau)

    optimizer.zero_grad()

    logits, feat = model(x)

    loss_ce = ce_loss(logits, y)
    loss_supcon = supcon_loss(feat, y)
    loss = loss_ce + supcon_lambda * loss_supcon

    loss.backward()
    optimizer.step()

    print(
        f"loss={loss.item():.4f} "
        f"ce={loss_ce.item():.4f} "
        f"supcon={loss_supcon.item():.4f}"
    )


if __name__ == "__main__":
    main()
