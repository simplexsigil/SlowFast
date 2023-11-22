import torch


def calculate_receptive_fields(kernel, padding, stride, input_shape):
    receptive_fields = []

    for i in range(len(kernel)):
        field_indices = []
        for j in range((input_shape[i] + 2 * padding[i] - kernel[i]) // stride[i] + 1):
            start_idx = j * stride[i] - padding[i]
            end_idx = start_idx + kernel[i]
            field_indices.append((max(start_idx, 0), min(end_idx, input_shape[i])))
        receptive_fields.append(field_indices)

    return receptive_fields


def blackout_blocks(
    input_tensor: torch.Tensor, mask: torch.Tensor, receptive_fields: torch.Tensor
):
    """
    Blackout blocks in the input tensor based on the mask and receptive fields.

    :param input_tensor: The input tensor (shape: [batch, channels, depth, height, width])
    :param mask: The mask tensor (same spatial dimensions as input_tensor)
    :param receptive_fields: Receptive field ranges for each dimension
    :return: Modified input tensor with blocks blacked out
    """
    assert (
        input_tensor.shape[2:] == mask.shape
    ), "Input tensor and mask must have the same spatial dimensions"

    # Iterate over each pixel in the mask
    for d in range(mask.shape[0]):
        for h in range(mask.shape[1]):
            for w in range(mask.shape[2]):
                if mask[d, h, w] == 0:
                    # Determine the receptive field for this pixel
                    d_start, d_end = receptive_fields[0][d]
                    h_start, h_end = receptive_fields[1][h]
                    w_start, w_end = receptive_fields[2][w]

                    # Blackout the corresponding block in the input tensor
                    input_tensor[:, :, d_start:d_end, h_start:h_end, w_start:w_end] = 0

    return input_tensor
