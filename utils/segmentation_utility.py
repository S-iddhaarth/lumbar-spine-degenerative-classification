import pandas as pd

def reconstruct_df(df)->tuple:
    '''
    takes dataframe with all data and splits it into three different dataframes 
    '''
    
    
    series = ['Sagittal T2/STIR', 'Sagittal T1', 'Axial T2']
    conditions = ['spinal_canal_stenosis', 'neural_foraminal_narrowing', 'subarticular_stenosis']
    df_name=[pd.DataFrame(),pd.DataFrame(),pd.DataFrame()]
    
    for i,(se, cond) in enumerate(zip(series, conditions)):
        filtered_df = df[df['series_description'] == se] # Filter the dataframe based on 'series_description'

        # Select columns to keep
        columns_to_keep = filtered_df.columns[:8].tolist()
        columns_to_keep += [col for col in filtered_df.columns[8:] if cond in col]

        #filter and update
        filtered_df = filtered_df[columns_to_keep] 
        df_name[i] = filtered_df
    
    return df_name[0],df_name[1],df_name[2]



def xy_spinal_neural(df):
    # Required levels and associated columns
    required_levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']

    ## filterout data which having both level
    df_filtered = df.groupby(['study_id','series_id','instance_number']).filter(
    lambda group: set(required_levels).issubset(group['level'].unique()))

    # make localize 
    grouped = df_filtered.groupby(['study_id', 'series_id', 'instance_number'])
    result_rows = []
    for (study_id, series_id, instance_number), group in grouped:
        # Create a dictionary to store the data for this group
        row_data = {'study_id': study_id, 'series_id': series_id, 'instance_number': instance_number}
    
        # Check if all other columns have the same value in the group, if so, keep them
        for col in group.columns:
            if col not in ['level', 'x', 'y']:  # Exclude 'level', 'x', and 'y' columns
                if group[col].nunique() == 1:  # Check if all values in this column are the same
                    row_data[col] = group[col].iloc[0]  # Take the unique value


        # Extract x and y coordinates for each required level
        for level in required_levels:
            level_data = group[group['level'] == level]
            if not level_data.empty:
                row_data[f'x_level_{level.lower().replace("/", "_")}'] = level_data['x'].values[0]
                row_data[f'y_level_{level.lower().replace("/", "_")}'] = level_data['y'].values[0]

        # Add the row data to the result_rows list
        result_rows.append(row_data)

    # Convert the result_rows list to a DataFrame
    result = pd.DataFrame(result_rows)
    return result


def xy_subarticular(df):
    # Required levels and associated columns
    require_condition=['Right Subarticular Stenosis','Left Subarticular Stenosis']

    ## filterout data which having both level
    df_filtered = df.groupby(['study_id','series_id','instance_number','level']).filter(
    lambda group: set(require_condition).issubset(group['condition'].unique()))

    # make localize 
    grouped = df_filtered.groupby(['study_id', 'series_id', 'instance_number','level'])
    result_rows = []
    for (study_id, series_id, instance_number,level), group in grouped:
        # Create a dictionary to store the data for this group
        row_data = {'study_id': study_id, 'series_id': series_id, 'instance_number': instance_number,'level':level}
    
        # Check if all other columns have the same value in the group, if so, keep them
        for col in group.columns:
            if col not in ['series_description','condition','x', 'y']:  # Exclude 'level', 'x', and 'y' columns
                if group[col].nunique() == 1:  # Check if all values in this column are the same
                    row_data[col] = group[col].iloc[0]  # Take the unique value

        # Extract x and y coordinates for each required level
        for condition in require_condition:
            condition_data = group[group['condition'] == condition]
            if not condition_data.empty:
                row_data[f'x_level_{condition.split(" ")[0]}'] = condition_data['x'].values[0]
                row_data[f'y_level_{condition.split(" ")[0]}'] = condition_data['y'].values[0]

        # Add the row data to the result_rows list
        result_rows.append(row_data)

    # Convert the result_rows list to a DataFrame
    df_result = pd.DataFrame(result_rows)
    
    #dummy varables
    columns_to_encode = 'level'
    df_encoded = pd.get_dummies(df_result[columns_to_encode], columns=columns_to_encode, prefix=columns_to_encode)
    df_encoded = df_encoded.astype(float)# Convert boolean True/False to 1.0/0.0
    df_result['level_array'] = df_encoded.values.tolist()
    #result=pd.concat([df_result,df_encoded],axis=1)
    df_result = df_result.drop(columns=columns_to_encode)
    
    return df_result