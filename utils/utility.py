import os
import torch
import random 
import numpy as np
import pandas as pd
import pandas.api.types
import sklearn.metrics

def grad_flow_dict(named_parameters: dict) -> dict:
    """
    Computes the average gradient of the parameters that require gradients and 
    are not biases from the given named parameters of a model.

    Args:
        named_parameters (dict): A dictionary of named parameters from a model.

    Returns:
        dict: A dictionary where keys are the layer names and values are the 
            average gradients of the respective layers.
    """
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            
    return {layers[i]: ave_grads[i] for i in range(len(ave_grads))}

def seed_everything(seed: int) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility across various libraries.
    
    This function sets the seed for the Python `random` module, the environment variable
    `PYTHONHASHSEED`, the NumPy random number generator, and the PyTorch random number
    generator (for both CPU and CUDA operations). It also configures PyTorch to use deterministic
    algorithms for operations to ensure reproducibility. 

    Args:
        seed (int): The seed value to be set for random number generators.

    Returns:
        None
    """
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Note: Setting this to True can improve performance but may affect reproducibility.


def get_series(df_series, study_id, series_description):
    """
    Retrieves a list of series IDs from a DataFrame based on the given study ID and series description.

    Args:
        df_series (pd.DataFrame): DataFrame containing series data with columns 'study_id', 'series_description', and 'series_id'.
        study_id (str): The study ID to filter the DataFrame.
        series_description (str): The series description to filter the DataFrame. It can be "Axial T2", "Sagittal T1", or "Sagittal T2/STIR".

    Returns:
        list or None: A list of series IDs matching the given study ID and series description, or None if no match is found.
    """
    # Filter the DataFrame based on study_id and series_description
    series_list = df_series[
        (df_series['study_id'] == study_id) & 
        (df_series['series_description'] == series_description)
    ]['series_id'].tolist()

    # Return None if no series IDs are found, otherwise return the list of series IDs
    if len(series_list) == 0:
        return None
    return series_list


def get_elements(length, size):
    if size <= length:
        start = (length - size) // 2
        return list(range(start, start + size))
    else:
        result = list(range(length))
        extra_elements = size - length
        result += list(range(extra_elements))
        return result

class ParticipantVisibleError(Exception):
    pass



def get_condition(full_location: str) -> str:
    # Given an input like spinal_canal_stenosis_l1_l2 extracts 'spinal'
    for injury_condition in ['spinal', 'foraminal', 'subarticular']:
        if injury_condition in full_location:
            return injury_condition
    raise ValueError(f'condition not found in {full_location}')


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        any_severe_scalar: float
    ) -> float:
    '''
    Pseudocode:
    1. Calculate the sample weighted log loss for each medical condition:
    2. Derive a new any_severe label.
    3. Calculate the sample weighted log loss for the new any_severe label.
    4. Return the average of all of the label group log losses as the final score, normalized for the number of columns in each group.
       This mitigates the impact of spinal stenosis having only half as many columns as the other two conditions.
    '''

    target_levels = ['normal_mild', 'moderate', 'severe']

    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission[target_levels].values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission[target_levels].values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission[target_levels].min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    solution['study_id'] = solution['row_id'].apply(lambda x: x.split('_')[0])
    solution['location'] = solution['row_id'].apply(lambda x: '_'.join(x.split('_')[1:]))
    solution['condition'] = solution['row_id'].apply(get_condition)

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    assert sorted(submission.columns) == sorted(target_levels)

    submission['study_id'] = solution['study_id']
    submission['location'] = solution['location']
    submission['condition'] = solution['condition']

    condition_losses = []
    condition_weights = []
    for condition in ['spinal', 'foraminal', 'subarticular']:
        condition_indices = solution.loc[solution['condition'] == condition].index.values
        condition_loss = sklearn.metrics.log_loss(
            y_true=solution.loc[condition_indices, target_levels].values,
            y_pred=submission.loc[condition_indices, target_levels].values,
            sample_weight=solution.loc[condition_indices, 'sample_weight'].values
        )
        condition_losses.append(condition_loss)
        condition_weights.append(1)

    any_severe_spinal_labels = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['severe'].max())
    any_severe_spinal_weights = pd.Series(solution.loc[solution['condition'] == 'spinal'].groupby('study_id')['sample_weight'].max())
    any_severe_spinal_predictions = pd.Series(submission.loc[submission['condition'] == 'spinal'].groupby('study_id')['severe'].max())
    any_severe_spinal_loss = sklearn.metrics.log_loss(
        y_true=any_severe_spinal_labels,
        y_pred=any_severe_spinal_predictions,
        sample_weight=any_severe_spinal_weights
    )
    condition_losses.append(any_severe_spinal_loss)
    condition_weights.append(any_severe_scalar)
    return np.average(condition_losses, weights=condition_weights)

def generate_ground_truth(root_dir, keys):
    keys = [int(i) for i in keys]
    train_main = pd.read_csv(os.path.join(root_dir, "train.csv"))
    train_main = train_main[train_main['study_id'].isin(keys)]
    solution = train_main.melt(id_vars=["study_id"], var_name="full_label", value_name="severity")
    solution["row_id"] = solution.apply(lambda row: str(row.study_id) + "_" + row.full_label, axis=1)
    
    # Fill severity with "Normal/Mild" where NaN
    solution.severity = solution.severity.fillna("Normal/Mild")
    
    # Set the normal_mild, moderate, and severe columns
    solution.loc[solution.severity == "Normal/Mild", "normal_mild"] = 1
    solution.loc[solution.severity == "Moderate", "moderate"] = 1
    solution.loc[solution.severity == "Severe", "severe"] = 1

    # Set sample_weight column
    solution.loc[solution.severity == "Normal/Mild", "sample_weight"] = 1
    solution.loc[solution.severity == "Moderate", "sample_weight"] = 2
    solution.loc[solution.severity == "Severe", "sample_weight"] = 3

    # Select and arrange columns
    solution = solution[["study_id", "row_id", "normal_mild", "moderate", "severe", "sample_weight"]]
    
    # Fill NaN values with 0
    solution = solution.fillna(0)
    
    # Sort the DataFrame by row_id
    solution = solution.sort_values(by=["row_id"])
    
    # Save the resulting DataFrame to a CSV file
    solution.to_csv("temp_train_solution.csv", index=False)
    return solution

def substitute_patterns(array):
    # Define the mapping
    pattern_to_value = {
        (1, 0, 0): 0,
        (0, 1, 0): 1,
        (0, 0, 1): 2
    }
    
    # Initialize an empty list to store the results
    result = []
    
    # Iterate over each row in the array
    for row in array:
        # Convert the row to a tuple so it can be used as a dictionary key
        row_tuple = tuple(row)
        
        # Append the corresponding value to the result list
        result.append(pattern_to_value.get(row_tuple, -1))  # Use -1 for any unmatched patterns
    
    return result