import os
import numpy as np

from tqdm import tqdm

from utils.general import create_folder, load_data
from utils.plots import plot_exploration, plot_bars


def save_figures(path, all_results):

    target_names, other_names, unique_names = [], [], []
    for name in list(all_results.keys()):
        name_split = name.split("_")
        if len(name_split) > 1:
            target_names.append(name)
            if name_split[0] not in unique_names:
                unique_names.append(name_split[0])
        else:
            other_names.append(name)

    best_results = {}
    explore_results = {}
    for u_name in unique_names:

        result_subset, x_vals, y_vals = [], [], []

        print("\nSearching Best %s Model\n" % u_name)

        for m_name in tqdm(target_names, desc="Processing"):

            if u_name in m_name:

                value = m_name.split("_")[-1]

                s_measures, u_measures = all_results[m_name]
                measure = u_measures["DBI"]

                x_vals.append(round(float(value), 2))
                y_vals.append(measure)
                result_subset.append(all_results[m_name])

        indices = np.argsort(x_vals)
        x_vals, y_vals = np.asarray(x_vals), np.asarray(y_vals)
        x_vals, y_vals = x_vals[indices], y_vals[indices]
        result_subset = np.asarray(result_subset)[indices]

        explore_results[u_name] = {"x_vals": x_vals, "y_vals": y_vals}
        best_index = np.argmin(y_vals)
        best_results[u_name] = result_subset[best_index]

    plot_exploration(path, explore_results, bounded=False)

    new_models = {}
    for name in other_names:
        new_models[name] = all_results[name]

    for name in best_results:
        new_models[name] = best_results[name]

    print("\nCalculating Final Model Analytics\n")

    all_results = {}
    for m_name in tqdm(list(new_models.keys()), desc="Processing"):

        s_measures, u_measures = new_models[m_name]

        path_save = os.path.join(path, "%s_supervised.png" % m_name)
        plot_bars(path_save, s_measures)

        path_save = os.path.join(path, "%s_ss.png" % m_name)
        subset = {"SS": u_measures["SS"]}
        plot_bars(path_save, subset, "SS")

        path_save = os.path.join(path, "%s_dbi.png" % m_name)
        subset = {"DBI": u_measures["DBI"]}
        plot_bars(path_save, subset, "DBI")


def load_files(path):

    data = {}

    all_files = [ele for ele in os.listdir(path) if ".joblib" in ele]
    for current_file in all_files:
        current_file = os.path.join(path, current_file)
        name = current_file.split("/")[-1].strip(".joblib")
        data[name] = load_data(current_file)

    return data


def run(params):

    path_data = params["paths"]["data"]
    path_results = params["paths"]["results"]

    all_files = [ele for ele in os.listdir(path_data) if ".joblib" in ele]

    for current_file in all_files:

        current_file = os.path.join(path_data, current_file)

        folder = current_file.split("/")[-3]
        tag = current_file.split("/")[-1].strip(".joblib")
        path_root = os.path.join(path_results, folder, tag)

        path_models = os.path.join(path_root, "models")
        path_save = os.path.join(path_root, "analytics")

        create_folder(path_save)

        print("\nTesting Models:")
        print("- Path Results = %s\n" % path_save)

        all_results = load_files(path_models)

        save_figures(path_save, all_results)
