from typing import List

import torch
import numpy as np
from scipy import ndimage as ndi
from scipy.special import softmax
from skimage.morphology import remove_small_objects, watershed, remove_small_holes
from src.modelling.constants import MASK_VALUES


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


def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def map_score(pr, gt, thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05)):
    true_objects = len(np.unique(gt))
    pred_objects = len(np.unique(pr))

    # Compute intersection between all objects
    intersection = np.histogram2d(gt.flatten(), pr.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(gt, bins=true_objects)[0]
    area_pred = np.histogram(pr, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Loop over IoU thresholds
    prec = []
    for t in thresholds:
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        prec.append(p)

    return np.mean(prec)


def my_watershed(result, threshold=0.3):
    mask_border = result[MASK_VALUES['main_mask']].copy()
    m = result[MASK_VALUES['main_mask']] * (1 - result[MASK_VALUES['border']])
    mask_border[m <= threshold + 0.35] = 0
    mask_border[m > threshold + 0.35] = 1
    mask_border = mask_border.astype(np.bool)
    mask_border = remove_small_objects(mask_border, 10).astype(np.uint8)

    mask = (result[MASK_VALUES['main_mask']] > threshold).astype(np.bool)
    mask = remove_small_holes(mask, 1000)
    mask = remove_small_objects(mask, 8).astype(np.uint8)

    markers = ndi.label(mask_border, output=np.uint32)[0]
    return watershed(mask, markers, mask=mask, watershed_line=True)


class MAPScore:
    def __init__(self, thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05), use_postproc: bool = False):
        self.thresholds = thresholds
        self.use_postproc = use_postproc

    def __call__(self, pr, gt):
        pr = pr.cpu().detach().numpy()
        gt = gt.numpy()

        scores = []
        for curr_pr, curr_gt in zip(pr, gt):
            if self.use_postproc:
                curr_pr = softmax(curr_pr, axis=0)
                curr_pr = my_watershed(curr_pr)

            curr_score = map_score(curr_pr, curr_gt, self.thresholds)
            scores.append(curr_score)

        return np.array(scores)


class IoU:

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7, main_ch: List[int] = None):
        self.threshold = threshold
        self.eps = eps
        self.main_ch = main_ch

    def __call__(self, pr, gt):
        pr = torch.sigmoid(pr)
        pr = _threshold(pr, self.threshold)
        pr, gt = _take_channel(pr, gt, self.main_ch)

        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + self.eps
        return (intersection + self.eps) / union
