from typing import Dict, Tuple, Union

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO


def make_transforms(transform: Dict[str, str], mode: str):
    _transform = albu.load(transform[mode], 'yaml')
    _transform.transforms = albu.Compose([*_transform.transforms, ToTensorV2()])
    return _transform


def make_df_coco(data_folder: str, dataset_name: str, mode: str, split_data: bool = True) -> Tuple[pd.DataFrame, COCO]:
    pass


def get_mask_border_from_coco(data: COCO, img_id: int) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    ann_ids = data.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = data.loadAnns(ann_ids)

    if not anns:
        return None, None

    mask = None
    borders = np.zeros(data.annToMask(anns[0]).shape, dtype=np.uint8)
    for num_ann in range(len(anns)):
        curr_mask = data.annToMask(anns[num_ann])
        if mask is None:
            mask = curr_mask.copy()
        else:
            mask += curr_mask

        contours, _ = cv2.findContours(curr_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(borders, contours, -1, 1, 3)

    mask = mask.astype(np.float32)
    mask -= borders

    mask = np.clip(mask, 0, 1).astype(np.uint8)
    borders = np.clip(borders, 0, 1).astype(np.uint8)

    return mask, borders


def get_mask_from_coco(data: COCO, img_id: int) -> Union[np.ndarray, None]:
    ann_ids = data.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = data.loadAnns(ann_ids)

    if not anns:
        return None

    mask = None
    for num_ann in range(len(anns)):
        curr_mask = data.annToMask(anns[num_ann])
        if mask is None:
            mask = curr_mask.copy()
        else:
            mask += curr_mask

    return np.clip(mask, 0, 1).astype(np.uint8)
