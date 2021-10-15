import torch


def sample2img(sample, load_resolution=384, train_resolution=64, n_rows=6):
    #sample = (B, num_letters, H, W)
    rgb_mult = 3
    sample_img = torch.zeros(
        (sample.shape[0], rgb_mult, load_resolution, load_resolution))
    cell_width = train_resolution

    for k in range(0, sample.shape[1]):
        y = k // n_rows
        x = k % n_rows
        cell_img = sample[:, k:k+1, :, :]  # (B, 1, 64, 64)
        cell_img = cell_img.repeat(1, rgb_mult, 1, 1)  # (B, 3, 64, 64)
        sample_img[:, :, y*cell_width:(y+1)*cell_width,
                   x*cell_width:(x+1)*cell_width] = cell_img
    return sample_img
