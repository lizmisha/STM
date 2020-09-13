import torch
from segmentation_models_pytorch.utils.losses import DiceLoss
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    __name__ = 'focal_loss'

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


class DiceFocalLoss(nn.Module):
    __name__ = 'dice_focal_loss'

    def __init__(self, eps=1, gamma=2, channel_main: list = None):
        super().__init__()
        self.dice = DiceLoss(eps=eps, activation='sigmoid')
        self.focal = FocalLoss(gamma=gamma)
        self.channel_main = channel_main

    def forward(self, y_pr, y_gt):
        if self.channel_main is not None:
            channels_not_main = [channel for channel in range(y_gt.shape[1]) if channel not in set(self.channel_main)]

            y_pr_main = torch.index_select(y_pr, dim=1, index=torch.tensor(self.channel_main).to(y_pr.device))
            y_gt_main = torch.index_select(y_gt, dim=1, index=torch.tensor(self.channel_main).to(y_gt.device))

            y_pr_not_main = torch.index_select(y_pr, dim=1, index=torch.tensor(channels_not_main).to(y_pr.device))
            y_gt_not_main = torch.index_select(y_gt, dim=1, index=torch.tensor(channels_not_main).to(y_gt.device))

            dice_main = self.dice(y_pr_main, y_gt_main)
            focal_main = self.focal(y_pr_main, y_gt_main)
            dice_not_main = self.dice(y_pr_not_main, y_gt_not_main)
            focal_not_main = self.focal(y_pr_not_main, y_gt_not_main)

            return 0.4 * dice_main + 0.4 * focal_main + 0.1 * dice_not_main + 0.1 * focal_not_main

        dice = self.dice(y_pr, y_gt)
        focal = self.focal(y_pr, y_gt)
        return 0.5 * dice + 0.5 * focal