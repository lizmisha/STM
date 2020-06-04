from typing import List

import torch


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _take_channel(pr, gt, main_ch=None):
    if main_ch is None:
        return pr, gt

    return (torch.index_select(pr, dim=1, index=torch.tensor(main_ch).to(pr.device)),
            torch.index_select(gt, dim=1, index=torch.tensor(main_ch).to(gt.device)))


class IoU:

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7, main_ch: List[int] = None):
        self.threshold = threshold
        self.eps = eps
        self.main_ch = main_ch

    def __call__(self, pr, gt):
        pr = _threshold(pr, self.threshold)
        pr, gt = _take_channel(pr, gt, self.main_ch)

        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + self.eps
        return (intersection + self.eps) / union
