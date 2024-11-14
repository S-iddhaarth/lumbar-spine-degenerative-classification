
import os
import pandas as pd
import pydicom
from natsort import natsorted
import numpy as np
import sys
sys.path.append('../')
import utils.utility as utils

def get_sagi2axi_data(study_id, series_id):
    """
    Retrieves the middle sagittal slice and corresponding axial slices for a given study and series.

    This function locates the middle sagittal slice and all the axial slices for a given study ID and series ID.
    It reads the DICOM files from the specified directories and returns the DICOM dataset for the middle
    sagittal slice and a list of DICOM datasets for the axial slices.

    Args:
        study_id (str): The ID of the study to retrieve data for.
        series_id (str or None): The ID of the sagittal series. If None, the function returns None.

    Returns:
        tuple:
            - pydicom.dataset.FileDataset or None: The DICOM dataset for the middle sagittal slice, or None if not found.
            - list of pydicom.dataset.FileDataset or None: A list of DICOM datasets for the axial slices, or None if not found.
    """
    if pd.isnull(series_id):
        return None, None

    sag_dir = os.path.join(img_dir, str(study_id), str(series_id))
    axi_series = utils.get_series(df_series, study_id, 'Axial T2')
    if axi_series is None:
        return None, None
    axi_dir_list = [
        os.path.join(img_dir, str(study_id), str(axi_series_id))
        for axi_series_id in axi_series
    ]
    if os.path.isdir(sag_dir) and len(axi_dir_list) != 0:
        sag_file_list = natsorted(os.listdir(sag_dir))
        mid_sag_ds = pydicom.dcmread(os.path.join(sag_dir, sag_file_list[len(sag_file_list) // 2]))

        axi_ds_list = []
        for axi_dir in axi_dir_list:
            axi_file_list = natsorted(os.listdir(axi_dir))
            axi_ds_list += [pydicom.dcmread(os.path.join(axi_dir, file)) for file in axi_file_list]
        return mid_sag_ds, axi_ds_list
    else:
        return None, None

def get_coord_from_heatmap_pred(pred, idx):
    """
    Extracts normalized coordinates from a heatmap prediction.

    This function takes a heatmap prediction and the index of the specific heatmap to process.
    It finds the coordinates of the maximum value in the heatmap and normalizes these coordinates
    to the range [0, 1].

    Args:
        pred (np.ndarray): The prediction heatmap array of shape (batch_size, num_heatmaps, height, width).
        idx (int): The index of the heatmap to process.

    Returns:
        tuple:
            - float: Normalized x-coordinate of the maximum value in the heatmap.
            - float: Normalized y-coordinate of the maximum value in the heatmap.
    """
    predi = pred[0, idx, ...].squeeze()
    y_coord, x_coord = np.unravel_index(predi.argmax(), predi.shape)
    x_norm = x_coord / predi.shape[1]
    y_norm = y_coord / predi.shape[0]
    return x_norm, y_norm

def dcm_affine(ds):
    """
    Compute the affine transformation matrix for a given DICOM dataset.
    
    The affine transformation matrix maps pixel coordinates to world coordinates based on
    the DICOM attributes: ImageOrientationPatient, PixelSpacing, and ImagePositionPatient.
    
    Args:
        ds (pydicom.dataset.FileDataset): The DICOM dataset containing the image attributes.
    
    Returns:
        np.ndarray: The 4x4 affine transformation matrix.
    """
    F11, F21, F31 = ds.ImageOrientationPatient[3:]
    F12, F22, F32 = ds.ImageOrientationPatient[:3]
    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, 0, Sz],
            [0, 0, 0, 1],
        ]
    )

def sagi_coord_to_axi_instance_number(sag_x_norm, sag_y_norm, mid_sag_ds, axi_ds_list):
    """
    Converts normalized sagittal coordinates to the corresponding axial slice instance number.
    
    This function maps a point from the sagittal view to the closest corresponding point
    in the axial view based on DICOM metadata. It calculates the world coordinates from
    the normalized sagittal coordinates and finds the closest axial slice using the
    distance to the slice's position.

    Args:
        sag_x_norm (float): Normalized x-coordinate in the sagittal slice (range 0 to 1).
        sag_y_norm (float): Normalized y-coordinate in the sagittal slice (range 0 to 1).
        mid_sag_ds (pydicom.dataset.FileDataset): DICOM dataset for the middle sagittal slice.
        axi_ds_list (list of pydicom.dataset.FileDataset): List of DICOM datasets for the axial slices.

    Returns:
        tuple:
            - str or None: Series ID of the closest axial slice. None if no close slice is found.
            - int or float: Instance number of the closest axial slice. np.nan if no close slice is found.
    """
    # Calculate sagittal world coordinates based on middle slice
    sag_affine = dcm_affine(mid_sag_ds)
    sag_coord = np.array([sag_y_norm * mid_sag_ds.Rows, sag_x_norm * mid_sag_ds.Columns, 0, 1])
    sag_world_coord = (sag_affine @ sag_coord)[:-1]

    # Get closest axial slice
    dist_list = []
    for ds in axi_ds_list:
        normal = np.cross(ds.ImageOrientationPatient[:3], ds.ImageOrientationPatient[3:])
        normal /= np.linalg.norm(normal)
        dist = np.abs(np.dot(sag_world_coord - ds.ImagePositionPatient, normal))
        dist_list.append(dist)
    axi_slice_idx = np.argmin(dist_list)
    min_dist = dist_list[axi_slice_idx]
    if min_dist > 5:
        return None, np.nan
    axi_series_id, axi_instance_number = axi_ds_list[axi_slice_idx].filename.split('/')[-2:]

    return axi_series_id, int(axi_instance_number.replace('.dcm', ''))
