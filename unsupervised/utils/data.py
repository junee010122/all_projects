from utils.general import load_data


class Dataset:

    def __init__(self, samples, labels, refs):

        self.samples = samples
        self.labels = labels
        self.refs = refs


def load_dataset(path):

    data = load_data(path)

    return Dataset(data["samples"], data["labels"], data["originals"])
