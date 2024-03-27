import os
import yaml
import shutil
import pandas as pd

from tqdm import tqdm

from utils.data import convert_data 


def load_datasets(params):

    path = params["paths"]["data"]
    
    
    all_files = [ele for ele in os.listdir(path)
                 if ".png" in ele][:20]

    all_samples = []
    for current_file in tqdm(all_files, desc="Loading"):
        path_file = os.path.join(path, current_file)

        all_samples.append(path_file)

    
    data = convert_data(all_samples, params)
    
    return data

def load_yaml(argument):

    return yaml.load(open(argument), Loader = yaml.FullLoader)


def parse_args(all_args):

    tags = ["--", "-"]
    all_args = all_args[1:]

    results = {}

    i=0
    while i< len(all_args) -1:
        arg = all_args[i].lower()
        for current_tag in tags:
            if current_tag in arg:
                arg = arg.replace(current_tag, "")
        results[arg] = all_args[i+1]
        i+=2

        return results


def load_config(sys_args):

    args = parse_args(sys_args)
    params = load_yaml(args["config"])
    params["cl"] = args

    return params


