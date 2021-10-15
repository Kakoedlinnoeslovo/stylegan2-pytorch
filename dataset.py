from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, load_resolution=384, train_resolution=64, num_letters=33):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.load_resolution = load_resolution
        self.train_resolution = train_resolution
        self.transform = transform
        self.num_letters = num_letters
        self.n_rows = 0
        while self.n_rows**2 < self.num_letters:
            self.n_rows+=1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.load_resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)
            
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = np.array(img)
        cell_width = self.train_resolution
        img_res = torch.zeros((self.num_letters, cell_width, cell_width), dtype=torch.float32)

        for k in range(0, self.num_letters):
            y = k // self.n_rows
            x = k % self.n_rows
            cell_img = img[y*cell_width:(y+1)*cell_width, x*cell_width:(x+1)*cell_width]
            cell_img = self.transform(cell_img)
            img_res[k, :, :] = cell_img[0, :, :]

        return img_res
