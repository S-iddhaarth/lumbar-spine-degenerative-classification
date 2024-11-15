import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def extract_regions_from_image(model, img, resize_size=(64, 64), num_regions=5):
    """
    Extracts, crops, and resizes regions from the image using the model's predicted mask.

    Parameters:
    - model: the model to generate the mask
    - img: input image tensor of shape (B, C, H, W)
    - resize_size: desired size to which each region will be resized (default is (32, 32))
    - num_regions: number of regions to extract per image (default is 5)

    Returns:
    - A tuple containing the resized regions for each unique mask value (excluding the background)
    """
    model.eval()
    with torch.no_grad():
        mask = model(img)
        _, mask = torch.max(mask, dim=1)  # Assuming mask output shape is (B, num_classes, H, W)

    batch_size = img.shape[0]
    regions = [torch.zeros((batch_size, 1, resize_size[0], resize_size[1])) for _ in range(num_regions)]

    for i in range(batch_size):  # Loop over batch dimension
        img_tensor = img[i]  # Get the image for the current batch index
        mask_tensor = mask[i]  # Get the mask for the current batch index
        unique_values = mask_tensor.unique().tolist()
        if 0 in unique_values:
            unique_values.remove(0)

        # Extract regions for the current image
        image_regions = []
        for value in unique_values:
            region = crop_and_resize_region(img_tensor, mask_tensor, value, resize_size)
            if region is not None:
                image_regions.append(region)

        # Fill the regions tensor
        for idx in range(min(num_regions, len(image_regions))):
            regions[idx][i] = image_regions[idx]

    return tuple(regions)

def crop_and_resize_region(image, mask, value, size=(32, 32)):
    """
    Crops and resizes the region of the image corresponding to a specific mask value.

    Parameters:
    - image: the input image tensor (C, H, W)
    - mask: the mask tensor (H, W)
    - value: the specific mask value to crop and resize
    - size: desired size to which the region will be resized (default is (32, 32))

    Returns:
    - The resized region tensor (C, new_H, new_W) or None if the region is empty
    """
    coords = torch.nonzero(mask == value)
    if len(coords) == 0:
        return None

    min_y, min_x = coords.min(dim=0).values
    max_y, max_x = coords.max(dim=0).values

    cropped_image = image[:, min_y-10:max_y + 10, min_x-10:max_x + 10]
    resized_image = TF.resize(cropped_image.unsqueeze(0), size)

    return resized_image.squeeze(0)

def visualize_regions(regions):
    """
    Visualizes the output regions.

    Parameters:
    - regions: a tuple of regions to visualize
    """
    for idx, region_batch in enumerate(regions):
        for i in range(region_batch.shape[0]):
            plt.figure()
            region = region_batch[i]
            if region.shape[0] == 1:  # Grayscale image
                plt.imshow(region.squeeze(), cmap='gray')
            else:  # RGB image
                plt.imshow(region.permute(1, 2, 0))  # (C, H, W) to (H, W, C)
            plt.title(f'Region {idx + 1} - Image {i + 1}')
            plt.show()