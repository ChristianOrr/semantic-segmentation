import numpy as np
from colormaps import create_ade20k_label_colormap
import matplotlib.pyplot as plt



def overlay_seg_mask(image: np.ndarray, binary_mask: np.ndarray):
    """
    Converts the segmentation mask to RGB using the colormap, 
    then overlays the RGB segmentation mask ontop of the image.
    Args:
        image: An RGB image.
        binary_mask: Binarized semantic segmentation mask.
    Returns:
        An image with the segmentation mask overlayed on it.
    """

    image = np.squeeze(image)
    binary_mask = np.squeeze(binary_mask)
    color_seg = np.zeros((binary_mask.shape[-3], binary_mask.shape[-2], 3), dtype=np.uint8)
    palette = np.array(create_ade20k_label_colormap())
    for label, color in enumerate(palette):
        color_seg[binary_mask[:, :, label], :] = color
    color_seg = color_seg[..., ::-1]  # convert to BGR

    img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
    img = img.astype(np.uint8)
    return img


def plot_pred_and_target(image, pred, target):
    """
    Plots the prediction and target segmentations side-by-side,
    both overlayed on the original image.
    Args:
        image: An RGB image.
        pred: Binarized semantic segmentation model predictions.
        target: Binarized semantic segmentation labels.
    Returns:
        Nothing. Simply plots the image.
    """
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    axs[0].set_title("Prediction")
    pred_overlay = overlay_seg_mask(
        np.array(image),
        np.array(pred, dtype=bool)
    )
    axs[0].imshow(pred_overlay)
    axs[0].axis("off")
    axs[1].set_title("Target")
    target_overlay = overlay_seg_mask(
        np.array(image),
        np.array(target, dtype=bool)
    )
    axs[1].imshow(target_overlay)
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()