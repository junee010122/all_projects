import os
import yaml
import shutil
import pandas as pd



def clear_logfile(path, filename="metrics.csv"):
    
    #path_file = os.path.join(path, "lightning_logs", "regression", "training", filename)
    path_file = os.path.join(path, "lightning_logs", "training", filename)
    if os.path.exists(path_file):
        os.remove(path_file)


def log_params(params):

    print("\n---------------------------\n")

    print("Experiment Parameters")

    for current_key in params:
        print("\n%s Parameters:" % current_key.capitalize())
        for sub_key in params[current_key]:
            print("- %s: %s" % (sub_key, params[current_key][sub_key]))

    print("\n---------------------------\n")


def create_folder(path, overwrite=False):

    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def load_yaml(argument):

    return yaml.load(open(argument), Loader=yaml.FullLoader)


def parse_args(all_args):

    tags = ["--", "-"]

    all_args = all_args[1:]
    
    # 이 줄 왜 필요한거임...?
    if len(all_args) % 2 != 0:
        print("Argument '%s' not defined" % all_args[-1])

    results = {}
    
    # 왜 iterate하도록 한 거지? 그냥 yaml만추출하면 되는 거 아닌가?
    i = 0
    while i < len(all_args) - 1:
        arg = all_args[i].lower()
        for current_tag in tags:
            if current_tag in arg:
                arg = arg.replace(current_tag, "")
        results[arg] = all_args[i + 1]
        i += 2

    return results


def load_config(sys_args):

    args = parse_args(sys_args)
    params = load_yaml(args["config"])
    params["cl"] = args

    return params


def load_data(path):

    return pd.read_csv(path)


def collect_csv(path_root):
    path_list = []
    for i,ele in enumerate(os.listdir(path_root)):
        if "version" in ele:
            path_list.append(ele)
            tag = "version_%s" % i
            path_list[i] = os.path.join(path_root, tag, "metrics.csv")

    return path_list
