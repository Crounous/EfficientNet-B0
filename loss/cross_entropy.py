from torch import nn, Tensor
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """ [https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html] """

    def __init__(self, reduction='mean', label_smoothing=0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(prediction, target, reduction=self.reduction, label_smoothing=self.label_smoothing)
