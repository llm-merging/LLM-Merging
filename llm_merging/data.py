import pandas as pd

class Dataset(object):
    def __init__(
        self,
        dataset_filepath: str,
    ):
        self.dataset = []
        self.dataset = pd.read_csv(dataset_filepath).to_dict('records')
        for dp in self.dataset:
            if not dp['answer_choices'] or dp['answer_choices'] != dp['answer_choices']:
                del dp['answer_choices']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]