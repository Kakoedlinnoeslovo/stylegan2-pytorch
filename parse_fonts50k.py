import h5py
import cv2
import os
import numpy as np
from tqdm import tqdm

root_dir = "/home/roman/stylegan2-pytorch/fonts50k/"
filename = os.path.join(root_dir, "fonts.hdf5")
out_dir = "/home/roman/stylegan2-pytorch/fonts50k/imgs"
os.makedirs(out_dir, exist_ok=True)

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = f[a_group_key]
    for i, value in tqdm(enumerate(data)): #value has shape (62, 64, 64)
        cell_width = 64
        m = 8
        w = h = m*cell_width
        img = np.zeros((w, h, 3))
        for k in range(0, value.shape[0]):
            y = k // m #62//8 = 7
            x = k % m
            cell_img = np.repeat(value[k][:, :, None], 3, 2) #(64, 64, 3)
            img[y*cell_width:(y+1)*cell_width, x*cell_width:(x+1)*cell_width, :] = cell_img
        cv2.imwrite(os.path.join(out_dir, f"{i}.png"), img)
    
