import os

from utils.data import load_dataset
from utils.general import create_folder
from utils.models import select_models, optimize_and_save


def run(params):

    path_data = params["paths"]["data"]
    path_results = params["paths"]["results"]

    all_files = [ele for ele in os.listdir(path_data) if ".joblib" in ele]

    for i, current_file in enumerate(all_files):

        current_file = os.path.join(path_data, current_file)

        folder = current_file.split("/")[-3]
        tag = current_file.split("/")[-1].strip(".joblib")
        path_save = os.path.join(path_results, folder, tag, "models")
        create_folder(path_save)

        data = load_dataset(current_file)

        all_models = select_models(params)

        print("\nTraining Models:")
        print("- Path Data = %s" % current_file)
        print("- Path Results = %s\n" % path_save)

        optimize_and_save(path_save, all_models, data, i)
