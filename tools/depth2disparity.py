import numpy as np
import timeit
import matplotlib.pyplot as plt


def depth_to_color_realsense(
        d_img: np.array, d_max, d_min, map_zero_to_dmax=False,
        inverse_colorization=False):
    """Deprecated.
    This function was based on this
    https://web.archive.org/web/20230125235359/https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
    """

    if map_zero_to_dmax: d_img[d_img == 0] = d_max

    if inverse_colorization:
        disp = 1 / d_img
        disp_max = 1/ d_min
        disp_min = 1 / d_max
        d_normal = (disp - disp_min) / (disp_max - disp_min)
        # d_normal *= 1529
    else:
        # Normalization
        d_normal = np.clip((d_img - d_min), a_min=0, a_max=d_max)

        d_normal = d_normal / (d_max - d_min) * 1529

    # d_normal = np.clip(d_normal, a_min=0, a_max=1529)

    col = np.zeros((*d_img.shape[:2], 3), dtype=np.uint8)

    # RED
    pr_mask_255 = ((0 <= d_normal) & (d_normal <= 255)) | (
                (1275 < d_normal) & (d_normal <= 1529))
    col[:, :, 0][pr_mask_255] = 255

    pr_mask_255_d_normal = (255 < d_normal) & (d_normal <= 510)
    col[:, :, 0][pr_mask_255_d_normal] = 255 - d_normal[pr_mask_255_d_normal]

    pr_mask_zero = (510 < d_normal) & (d_normal <= 1020)
    col[:, :, 0][pr_mask_zero] = 0

    pr_mask_d_normal_1020 = (1020 < d_normal) & (d_normal <= 1275)
    col[:, :, 0][pr_mask_d_normal_1020] = d_normal[pr_mask_d_normal_1020] - 1020

    # GREEN
    pg_mask_dnormal = (0 < d_normal) & (d_normal <= 255)
    col[:, :, 1][pg_mask_dnormal] = d_normal[pg_mask_dnormal]

    pg_mask_255 = (255 < d_normal) & (d_normal <= 510)
    col[:, :, 1][pg_mask_255] = 255

    pg_mask_1020_dnormal = (510 < d_normal) & (d_normal <= 765)
    col[:, :, 1][pg_mask_1020_dnormal] = 765 - d_normal[pg_mask_1020_dnormal]

    pg_mask_zero = ((765 < d_normal) & (d_normal <= 1529))
    col[:, :, 1][pg_mask_zero] = 0

    # BLUE
    pb_mask_dnormal = (0 < d_normal) & (d_normal <= 765)
    col[:, :, 2][pb_mask_dnormal] = d_normal[pb_mask_dnormal] # 0  #

    pb_mask_dnormal_510 = (765 < d_normal) & (d_normal <= 1020)
    col[:, :, 2][pb_mask_dnormal_510] = d_normal[pb_mask_dnormal_510] - 765

    pb_mask_255 = (1020 < d_normal) & (d_normal <= 1275)
    col[:, :, 2][pb_mask_255] = 255

    pb_mask_1529 = (1275 < d_normal) & (d_normal <= 1529)
    col[:, :, 2][pb_mask_1529] = 1529 - d_normal[pb_mask_1529]

    # assert np.all(np.sum(col, axis=2) >= 255)

    return col


def depth_to_color_corrected(
        d_img: np.array, d_max, d_min, map_zero_to_dmax=False,
        inverse_colorization=True):
    """Depth to RGB compression method, implemented by David Schneider,
    inverse_colorization was set to True.

    This function was based on this
    https://web.archive.org/web/20230125235359/https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
    but since the described algorithm was faulty (wrong implementation of hue range),
    it was implemented using the information here:
    https://web.archive.org/web/20230125235132/https://photonlexicon.com/forums/showthread.php/23417-Color-Code-Mastering-RGB
    """

    if map_zero_to_dmax: d_img[d_img == 0] = d_max

    if inverse_colorization:
        disp = 1 / d_img
        disp_max = 1 / d_min
        disp_min = 1 / d_max
        d_normal = (disp - disp_min) / (disp_max - disp_min)
        d_normal *= 1529
    else:
        # Normalization
        d_normal = np.clip((d_img - d_min), a_min=0, a_max=d_max)

        d_normal = d_normal / (d_max - d_min) * 1529

    d_normal = np.clip(d_normal, a_min=0, a_max=1529)

    col = np.zeros((*d_img.shape[:2], 3), dtype=np.uint8)

    # RED
    pr_mask_255 = ((0 <= d_normal) & (d_normal <= 255)) | (
                (1275 < d_normal) & (d_normal <= 1529))
    col[:, :, 0][pr_mask_255] = 255

    pr_mask_255_d_normal = (255 < d_normal) & (d_normal <= 510)
    col[:, :, 0][pr_mask_255_d_normal] = 510 - d_normal[pr_mask_255_d_normal]

    pr_mask_zero = (510 < d_normal) & (d_normal <= 1020)
    col[:, :, 0][pr_mask_zero] = 0

    pr_mask_d_normal_1020 = (1020 < d_normal) & (d_normal <= 1275)
    col[:, :, 0][pr_mask_d_normal_1020] = d_normal[pr_mask_d_normal_1020] - 1020

    # GREEN
    pg_mask_dnormal = (0 < d_normal) & (d_normal <= 255)
    col[:, :, 1][pg_mask_dnormal] = d_normal[pg_mask_dnormal]

    pg_mask_255 = (255 < d_normal) & (d_normal <= 765)
    col[:, :, 1][pg_mask_255] = 255

    pg_mask_1020_dnormal = (765 < d_normal) & (d_normal <= 1020)
    col[:, :, 1][pg_mask_1020_dnormal] = 1020 - d_normal[pg_mask_1020_dnormal]

    pg_mask_zero = ((1020 < d_normal) & (d_normal <= 1529))
    col[:, :, 1][pg_mask_zero] = 0

    # BLUE
    pb_mask_dnormal = (0 < d_normal) & (d_normal <= 510)
    col[:, :, 2][pb_mask_dnormal] = 0  # d_normal[pb_mask_dnormal]

    pb_mask_dnormal_510 = (510 < d_normal) & (d_normal <= 765)
    col[:, :, 2][pb_mask_dnormal_510] = d_normal[pb_mask_dnormal_510] - 510

    pb_mask_255 = (765 < d_normal) & (d_normal <= 1275)
    col[:, :, 2][pb_mask_255] = 255

    pb_mask_1529 = (1275 < d_normal) & (d_normal <= 1529)
    col[:, :, 2][pb_mask_1529] = 1529 - d_normal[pb_mask_1529]

    assert np.all(np.sum(col, axis=2) >= 255)

    return col


def color_to_depth_realsense(color_img: np.array, d_max, d_min, inverse_colorization=False):
    """Convert rgb value to quantization 0-1529 value
    Reference:
        1) https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
    """
    if color_img.dtype == np.uint8:
        color_img = color_img.astype(np.int16)  # int32 too slow

    dnormal = np.zeros(color_img.shape[:-1])
    # color_img should be in RGB
    prr, prg, prb = np.squeeze(np.dsplit(color_img, 3))  # quicker than slicing

    # get all needed masks
    prr_greater_equal_prg = prr >= prg
    prr_greater_equal_prb = prr >= prb
    prg_greater_equal_prb = prg >= prb
    prg_smaller_prb = ~prg_greater_equal_prb
    prg_greater_equal_prr = prg >= prr
    prb_greater_equal_prg = prb >= prg
    prb_greater_equal_prr = prb >= prr
    prr_greater_equal_prg_and_prb = prr_greater_equal_prg & prr_greater_equal_prb

    # prg - prb (first row of the formulation)
    mask_prg_minus_prb = (prr_greater_equal_prg_and_prb
                          & prg_greater_equal_prb)
    dnormal[mask_prg_minus_prb] = (prg - prb)[mask_prg_minus_prb]

    # prg - prb + 1529 (second row of the formulation)
    mask_prg_minus_prb_plus_1529 = (prr_greater_equal_prg_and_prb & prg_smaller_prb)
    dnormal[mask_prg_minus_prb_plus_1529] = (prg - prb + 1529)[mask_prg_minus_prb_plus_1529]

    # prb - prr + 510
    mask_prb_minus_prr_plus_510 = prg_greater_equal_prr & prg_greater_equal_prb
    dnormal[mask_prb_minus_prr_plus_510] = (prb - prr + 510)[mask_prb_minus_prr_plus_510]

    # prr - prg + 1020
    mask_prr_minus_prg_plus_1020 = prb_greater_equal_prg & prb_greater_equal_prr
    dnormal[mask_prr_minus_prg_plus_1020] = (prr - prg + 1020)[mask_prr_minus_prg_plus_1020]

    if inverse_colorization:
        disp_min = 1 / d_max
        disp_max = 1 / d_min
        d_recovery = disp_min + (disp_max - disp_min) * dnormal / 1529
        d_recovery = 1 / d_recovery
    else:
        d_recovery = d_min + (d_max - d_min) * dnormal / 1529

    return d_recovery


class Color2Depth:
    """Color to depth based on lookup table"""
    def __init__(self, inverse_colorization=True, map_max=True):
        self.inverse_colorization = inverse_colorization
        self.map_max = map_max
        self.lookup_min, self.lookup_max = 1, 1530
        self.lookup_table = self._prepare_lookup_table()

    def _prepare_lookup_table(self):
        depth_range = (np.arange(self.lookup_min - 1, self.lookup_max) + 1)[np.newaxis, :]
        lookup = depth_to_color_corrected(
            depth_range, d_min=self.lookup_min, d_max=self.lookup_max,
            inverse_colorization=self.inverse_colorization,
        )[0]

        lookup_table = {}
        range_list = range(self.lookup_max)
        if self.map_max:
            range_list = range_list[::-1]

        for i in range_list:  # [0, 1529]
            color_cur = lookup[i].tobytes()
            if color_cur not in lookup_table:
                lookup_table[color_cur] = i + 1  # [1, 1530]

        return lookup_table

    def back_normalize(self, d, d_max, d_min):
        # convert to range [0, 1]
        d = (d - self.lookup_min) / (self.lookup_max - self.lookup_min)

        # convert to range [d_min, d_max]
        d = d * (d_max - d_min) + d_min
        return d

    def __call__(self, color_img: np.array, d_max, d_min):
        dnormal = np.ones(color_img.shape[:-1]) * self.lookup_min
        r, g, b = color_img[..., 0], color_img[..., 1], color_img[..., 2]
        lookup_table = self.lookup_table
        for key in lookup_table:
            key_arr = np.frombuffer(key, dtype=np.uint8)
            mask_cur = (r == key_arr[0]) & (g == key_arr[1]) & (b == key_arr[2])
            dnormal[mask_cur] = lookup_table[key]

        d_recovery = self.back_normalize(dnormal, d_max, d_min)
        return d_recovery


def tmp1():
    depth_rec_lookup = rec_fn(color_img, d_min=d_min, d_max=d_max)
    return depth_rec_lookup


def tmp2():
    depth_rec_realsense = color_to_depth_realsense(
        color_img, d_min=d_min, d_max=d_max, inverse_colorization=True)
    return depth_rec_realsense


def plot_depth(depth, color_img, d_max, d_min, show_max_depth=500):
    depth_rec_lookup = tmp1()
    depth_rec_realsense = tmp2()

    def normalize(*d):
        return [(d_ - d_min) / (d_max - d_min) * 255 for d_ in d]

    depth, depth_lookup, depth_realsense = normalize(depth, depth_rec_lookup, depth_rec_realsense)

    fig, ax = plt.subplots(nrows=4, ncols=1)
    ax[0].imshow(depth[:, :show_max_depth])
    ax[1].imshow(color_img[:, :show_max_depth])
    ax[2].imshow(depth_lookup[:, :show_max_depth])
    ax[3].imshow(depth_realsense[:, :show_max_depth])

    plt.show()


if __name__ == '__main__':
    d_min, d_max = 1, 1530
    depth = np.arange(d_min, d_max + 1)[np.newaxis, :].repeat(100, axis=0)
    color_img = depth_to_color_corrected(depth, d_min=d_min, d_max=d_max, inverse_colorization=True)

    # Reconstruction with lookup table
    rec_fn = Color2Depth(inverse_colorization=True, map_max=True)

    # Plot
    plot_depth(depth, color_img, d_max, d_min, show_max_depth=1530)

    # print(timeit.repeat("tmp2()", "from __main__ import tmp2", number=1000))


