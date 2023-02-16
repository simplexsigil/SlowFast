import numpy as np
import torch


def depth_to_color(d_img: np.array, d_max, d_min, map_zero_to_dmax=False, inverse_colorization=False):

    # This function was based on this
    # https://web.archive.org/web/20230125235359/https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
    # but since the described algorithm was faulty (wrong implementation of hue range),
    # it was implemented using the information here:
    # https://web.archive.org/web/20230125235132/https://photonlexicon.com/forums/showthread.php/23417-Color-Code-Mastering-RGB

    if map_zero_to_dmax: d_img[d_img == 0] = d_max

    if inverse_colorization:
        disp = 1 / d_img
        disp_max = 1/ d_min
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

def color_to_depth(color_img: np.array, d_max, d_min, inverse_colorization=False):
    """Convert rgb value to quantization 0-1535 value
    Reference:
        1) https://github.com/TetsuriSonoda/rs-colorize/blob/master/rs-colorize/rs-colorize.cpp#L36
        2) https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
    """
    dnormal = np.zeros(color_img.shape[:-1])
    # color_img should in RGB
    prr = color_img[..., 0]
    prg = color_img[..., 1]
    prb = color_img[..., 2]

    # get all needed masks
    prr_greater_equal_prg = prr >= prg
    prr_greater_equal_prb = prr >= prb
    prg_greater_equal_prb = prg >= prb
    prg_smaller_prb = ~prg_greater_equal_prb
    prg_greater_equal_prr = prg >= prr
    prb_greater_equal_prg = prb >= prg
    prb_greater_equal_prr = prb >= prr

    # prg - prb (first row of the formulation)
    mask_prg_minus_prb = (prr_greater_equal_prg & prr_greater_equal_prb
                          & prg_greater_equal_prb)
    dnormal[mask_prg_minus_prb] = (prg - prb)[mask_prg_minus_prb]

    # prg - prb + 1529 (second row of the formulation)
    mask_prg_minus_prb_plus_1529 = (prr_greater_equal_prg &
                                    prr_greater_equal_prb & prg_smaller_prb)
    dnormal[mask_prg_minus_prb_plus_1529] = (prg - prb + 1529)[mask_prg_minus_prb_plus_1529]

    # prb - prr + 510
    mask_prb_minus_prr_plus_510 = prg_greater_equal_prr & prg_greater_equal_prb
    dnormal[mask_prb_minus_prr_plus_510] = (prb - prr + 510)[mask_prb_minus_prr_plus_510]

    # prr - prg + 1020
    mask_prr_minus_prg_plus_1020 = prb_greater_equal_prg & prb_greater_equal_prr
    dnormal[mask_prr_minus_prg_plus_1020] = (prr - prg + 1020)[mask_prr_minus_prg_plus_1020]

    if inverse_colorization:
        disp_min = 1 / d_min
        disp_max = 1 / d_max
        d_recovery = 1529 * disp_min + (disp_max - disp_min) * dnormal
        d_recovery = 1529 / d_recovery
    else:
        d_recovery = d_min + (d_max - d_min) * dnormal / 1529

    return d_recovery


if __name__ == '__main__':
    color_img = torch.randn(224, 224, 3).numpy()
    color_to_depth(color_img, d_min=1, d_max=2000, inverse_colorization=True)

