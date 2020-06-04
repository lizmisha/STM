import os

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from src.data.data_utils import get_mask_border_from_coco, make_transforms, make_df_coco


class STMBorder(Dataset):

    def __init__(self, df: pd.DataFrame, data_coco: COCO, data_folder: str, transform=None):
        self.df = df
        self.data_coco = data_coco
        self.data_folder = data_folder
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(os.path.join(self.data_folder, row['image_name']))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = get_mask_border_from_coco(self.data_coco, row['id'])
        masks = np.stack(masks, axis=-1).astype('float')

        if self.transform is not None:
            augmented = self.transform(image=img, mask=masks)
            img = augmented['image']
            masks = augmented['mask']

        masks = masks.permute(2, 0, 1)
        return img, masks

    def __len__(self):
        return len(self.df)


def make_data(
        data_folder: str,
        dataset_name: str,
        mode: str,
        transform: dict,
        num_workers: int,
        batch_size: int,
):
    _transform = make_transforms(transform, mode)

    df, coco_data = make_df_coco(data_folder, dataset_name, mode)
    dataset = STMBorder(df, coco_data, data_folder, _transform)

    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return loader
