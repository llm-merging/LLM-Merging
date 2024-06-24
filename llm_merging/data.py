import json 

class Dataset(object):
    def __init__(
        self,
        dataset_filepath: str,
    ):
        self.dataset = []
        with open(dataset_filepath, "r") as f:
            for line in f.readlines():
                datapoint = json.loads(line.strip("\n"))
                self.dataset.append(datapoint)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

