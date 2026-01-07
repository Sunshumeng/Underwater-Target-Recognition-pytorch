import os

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pandas import read_csv
from numpy import load
import numpy as np
from sklearn.model_selection import train_test_split


class SoundDataSet(Dataset):
    """
    初始化函数:
    输入: info_data 文件的数据框格式读取信息 + 数据集路径
    并且写入相关属性。
    """

    def __init__(self, data_dir, csv_path):
        """
        data_dir : 存放 .npy 特征的目录
        csv_path : referX.csv，
        """
        self.df = read_csv(csv_path, encoding='utf-8')
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        label = int(self.df['label'].iloc[index])

        rel_path = self.df['filepath'].iloc[index]
        file_path = os.path.join(self.data_dir, rel_path)

        data = load(file_path, allow_pickle=True)

        return data, label


def get_dataloader(data_dir, csv_path, batch_size, train_percent=0.9,
                   num_workers=4):

    dataset = SoundDataSet(data_dir, csv_path)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train

    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)


def get_dataloader_with_test(data_dir,
                             csv_path,
                             batch_size,
                             train_percent=0.6,
                             valid_percent=0.2,
                             num_workers=4,
                             random_state=42,
                             use_stratify=True):
    """
    参数:
        data_dir      : .npy 数据目录
        csv_path      : csv 标注文件
        batch_size    : batch 大小
        train_percent : 训练集占比 ( 0.6)
        valid_percent : 验证集占比 ( 0.2)，剩下的自动给 test
        num_workers   : DataLoader 的 num_workers
        use_stratify  : 是否按 label 分层划分

    返回:
        train_dl, valid_dl, test_dl, (train_idx, valid_idx, test_idx)
    """
    dataset = SoundDataSet(data_dir, csv_path)
    num_samples = len(dataset)

    indices = np.arange(num_samples)
    labels = dataset.df['label'].values

    train_ratio = float(train_percent)
    valid_ratio = float(valid_percent)
    test_ratio = 1.0 - train_ratio - valid_ratio
    if test_ratio < 0:
        raise ValueError("train_percent + valid_percent 不能大于 1.0")

    test_size_1 = 1.0 - train_ratio
    stratify_labels = labels if (use_stratify and len(np.unique(labels)) > 1) else None

    train_idx, temp_idx, train_y, temp_y = train_test_split(
        indices,
        labels,
        test_size=test_size_1,
        random_state=random_state,
        stratify=stratify_labels
    )

    if valid_ratio + test_ratio > 0:
        valid_ratio_in_temp = valid_ratio / (valid_ratio + test_ratio)
    else:
        valid_ratio_in_temp = 0.5

    stratify_temp = temp_y if (use_stratify and len(np.unique(temp_y)) > 1) else None

    valid_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_y,
        test_size=(1.0 - valid_ratio_in_temp),
        random_state=random_state,
        stratify=stratify_temp
    )

    train_ds = Subset(dataset, train_idx)
    valid_ds = Subset(dataset, valid_idx)
    test_ds = Subset(dataset, test_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_dl, valid_dl, test_dl, (train_idx, valid_idx, test_idx)
