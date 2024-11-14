import os
import polars as pl
from tqdm import tqdm
import json
from collections import defaultdict
import random 

def naive_annotate(root,train_paths:dict,out:str)->None:
    series = pl.read_csv(os.path.join(root,train_paths["series"]))
    label = pl.read_csv(os.path.join(root,train_paths["labels"]))
    v1 = len(label)
    label = label.drop_nulls()
<<<<<<< HEAD
    print(v1 - len(label))
    category_order = ['Normal/Mild', 'Moderate', 'Severe']
    label = label.with_columns(
        [pl.col(column).cast(pl.Categorical).set_sorted(category_order) for column in label.columns[1:]]
    )
    label = label.to_dummies(columns=label.columns[1:])
    print(label.head())
=======
    
    label = label.to_dummies(columns=label.columns[1:])
    clms = label.columns
    for i in range(len(clms)):
        if i%3 == 0:
            continue
        if i%3 == 1:
            buffer = clms[i]
        if i%3 == 2:
            clms[i-1] = clms[i]
            clms[i] = buffer
    label = label[clms]
>>>>>>> aeb0f9e5afe9743dbb4f8bf85f54b3f051d213ec
    series_intermediate = defaultdict(list)
    os.makedirs(".cache",exist_ok=True)
    for series_data in tqdm(series.iter_rows(named=True),
                        total=series.shape[0], desc="Processing items"):
        series_intermediate[series_data["study_id"]].append(
            (os.path.join(
                str(train_paths["images"]),
                str(series_data["study_id"]),
                str(series_data["series_id"])
                ),
            series_data["series_description"])
        )
        
    series_intermediate = dict(series_intermediate)
    with open(os.path.join(".cache","series.json"),"w") as fl:
        json.dump(series_intermediate,fl)
    
    output = {}
    header = label.columns[1:]
    for rows in label.iter_rows():
        series_id,ground_truth = rows[0],dict(zip(header,rows[1:]))
        output[series_id] = (series_intermediate[series_id],ground_truth)
    with open(out,"w") as fl:
        json.dump(output,fl)

def remove_more_than_3(input:str,output:str)->None:
    with open(input,"r") as fl:
        annotation = json.load(fl)
    new = {}
    for key,value in annotation.items():
        count = len(value[0])
        if count == 3:
            new[key] = value
    with open(output,"w") as fl:
        json.dump(new,fl)

def split_annotations(annotations, split_ratio):
    # Convert the annotations to a list of items (key-value pairs)
    with open(annotations,"r") as fl:
        annotation = json.load(fl)
    items = list(annotation.items())
    
    # Shuffle the items to ensure randomness
    random.shuffle(items)
    
    # Calculate the split index
    split_index = int(len(items) * split_ratio)
    
    # Split the items into train and test sets
    train_items = items[:split_index]
    test_items = items[split_index:]
    
    # Convert the lists of items back to dictionaries
    train_annotations = dict(train_items)
    test_annotations = dict(test_items)
    
    out = {
        "train": train_annotations,
        "test": test_annotations
    }
    with open(annotations,"w") as fl:
        json.dump(out,fl)