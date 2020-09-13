import os
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


def get_img_list(data: COCO) -> list:
    img_list = []
    imgIds = data.getImgIds()
    for idx in imgIds:
        img_list.append([data.loadImgs(idx)[0]['file_name'], idx])

    return img_list


def make_df_coco(
        dataset_folder: str,
        dataset_name: str,
        mode: str,
        split_data: bool = True
) -> Tuple[pd.DataFrame, COCO]:
    coco_data = COCO(os.path.join(dataset_folder, dataset_name))
    df = pd.DataFrame(get_img_list(coco_data), columns=['image', 'id'])
    if not split_data:
        return df, coco_data

    train_df, valid_df = train_test_split(df, test_size=0.1, random_state=69)
    if mode == 'train':
        return train_df, coco_data

    return valid_df, coco_data


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


def get_mask_border_background_from_coco(
        data: COCO,
        imd_id: int
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
    mask, borders = get_mask_border_from_coco(data, imd_id)
    if mask is None or borders is None:
        return None, None, None

    mask_background = np.ones_like(mask, dtype=float)
    mask_background -= mask
    mask_background -= borders
    mask_background = np.clip(mask_background, 0, 1).astype(np.uint8)
    return mask, borders, mask_background


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


def get_labeled_mask_from_coco(data: COCO, img_id: int) -> Union[np.ndarray, None]:
    ann_ids = data.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = data.loadAnns(ann_ids)

    if not anns:
        return None

    mask = None
    for num_ann in range(len(anns)):
        curr_mask = data.annToMask(anns[num_ann]) * (num_ann + 1)
        if mask is None:
            mask = curr_mask.copy()
        else:
            mask += curr_mask

    labels = set(np.arange(0, len(anns) + 1))
    mask_labels = np.unique(mask)
    for lbl in mask_labels:
        if lbl not in labels:
            mask[mask == lbl] = 0

    return mask.astype(np.int32)
