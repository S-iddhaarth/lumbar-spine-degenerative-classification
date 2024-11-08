import os
import polars as pl
from tqdm import tqdm
import json
from collections import defaultdict

def naive_annotate(root,train_paths:dict,out:str)->None:
    series = pl.read_csv(os.path.join(root,train_paths["series"]))
    label = pl.read_csv(os.path.join(root,train_paths["labels"]))
    label = label.to_dummies(columns=label.columns[1:])
    
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
    for rows in label.iter_rows():
        series_id,ground_truth = rows[0],rows[1:]
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