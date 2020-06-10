import os

import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from scipy import ndimage as ndi
from torch.utils.data import Dataset, DataLoader

from src.data.data_utils import make_transforms, make_df_coco, get_mask_border_background_from_coco
from src.modelling.constants import MASK_VALUES


class STMBorder(Dataset):

    def __init__(self, df: pd.DataFrame, data_coco: COCO, data_folder: str, mode: str, transform=None):
        self.df = df
        self.data_coco = data_coco
        self.data_folder = data_folder
        self.mode = mode
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = cv2.imread(os.path.join(self.data_folder, row['image']))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = get_mask_border_background_from_coco(self.data_coco, row['id'])
        masks = np.stack(masks, axis=-1).astype('float')

        if self.transform is not None:
            augmented = self.transform(image=img, mask=masks)
            img = augmented['image']
            masks = augmented['mask']

        labeled_mask = masks.numpy()[:, :, MASK_VALUES['main_mask']]
        labeled_mask = ndi.label(labeled_mask)[0]
        masks = masks.permute(2, 0, 1)
        # labeled_mask = get_labeled_mask_from_coco(self.data_coco, row['id'])

        if self.mode == 'train':
            return img, masks

        return img, masks, labeled_mask

    def __len__(self):
        return len(self.df)


def make_data(
        data_folder: str,
        dataset_folder: str,
        dataset_name: str,
        mode: str,
        transform: dict,
        num_workers: int,
        batch_size: int,
        test_dataset_name: str = None
):
    _transform = make_transforms(transform, mode)

    if test_dataset_name is None:
        df, coco_data = make_df_coco(dataset_folder, dataset_name, mode)
    else:
        if mode == 'train':
            df, coco_data = make_df_coco(dataset_folder, dataset_name, mode, split_data=False)
        else:
            df, coco_data = make_df_coco(dataset_folder, test_dataset_name, mode, split_data=False)

    dataset = STMBorder(df, coco_data, data_folder, mode, _transform)

    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return loader
