import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out_1, out_2, label):
        euclidean_distance = nn.functional.pairwise_distance(out_1, out_2, keepdim=True)

        # calculation is here:
        # https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
        # https://lilianweng.github.io/posts/2021-05-31-contrastive/
        # if label = 1 it means images are not similar
        contrastiveLoss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) *
            torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return contrastiveLoss
