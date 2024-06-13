import csv, json
from pathlib import Path


def seed_all(seed):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def read_csv(fname):
    with open(fname) as f:
        reader = csv.reader(f)
        all_data = []
        for row in reader:
            all_data.append(row)

    return all_data


def write_csv(fname, data, header=None, mode='w', make_dir=False):
    if make_dir:
        fpath = fname.rsplit('/', 1)[0]
        Path(fpath).mkdir(parents=True, exist_ok=True)

    with open(fname, mode) as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def read_csv_as_json(fname, header=None):
    with open(fname) as f:
        if header:
            reader = csv.DictReader(f, header)
        else:
            tmp_reader = csv.reader(f)
            reader = csv.DictReader(f, next(tmp_reader, None))
        all_data = []
        for row in reader:
            all_data.append(row)

    return all_data


def read_csv_as_dictionary(fname, dict_key, header=None):
    with open(fname) as f:
        if header:
            reader = csv.DictReader(f, header)
        else:
            tmp_reader = csv.reader(f)
            reader = csv.DictReader(f, next(tmp_reader, None))
        all_data = []
        for row in reader:
            all_data.append(row)

    dict_data = dict()
    for row in all_data:
        if type(dict_key) == str:
            dict_data[row[dict_key]] = row
        elif callable(dict_key):
            dict_data[dict_key(row)] = row

    return dict_data


def read_json(fname):
    with open(fname) as f:
        return json.load(f)


def write_json(fname, data, mode='w', make_dir=False):
    if make_dir:
        fpath = fname.rsplit('/', 1)[0]
        Path(fpath).mkdir(parents=True, exist_ok=True)

    with open(fname, mode) as f:
        json.dump(data, f, indent=4)