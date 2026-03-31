import numpy as np
import os
import torch
from collections import defaultdict


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    dataset_lower = dataset.lower()
    if "eicu" in dataset_lower:
        data_list = process_eicu(data)
    elif "news" in dataset:
        data_list = process_text(data)
    elif "shakespeare" in dataset_lower:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_eicu(data):
    # New TPC format: separate temporal and static features
    if 'x_ts' in data:
        return _process_eicu_ts_static(data)

    # Legacy fused format
    x_list = data['x']
    if isinstance(x_list, np.ndarray):
        samples_iter = x_list.tolist()
    else:
        samples_iter = list(x_list)

    samples = [torch.as_tensor(np.asarray(sample), dtype=torch.float32) for sample in samples_iter]

    # Pad variable-length sequences to the max length in this client so
    # the default DataLoader collator can stack them into a batch.
    if samples and samples[0].dim() == 2:
        shapes = [s.shape[0] for s in samples]
        if len(set(shapes)) > 1:
            max_len = max(shapes)
            n_feat = samples[0].shape[1]
            padded = []
            for s in samples:
                if s.shape[0] < max_len:
                    pad = torch.zeros(max_len - s.shape[0], n_feat, dtype=s.dtype)
                    s = torch.cat([s, pad], dim=0)
                padded.append(s)
            samples = padded

    labels = torch.as_tensor(data['y'], dtype=torch.int64)
    return list(zip(samples, labels))


def _process_eicu_ts_static(data):
    """Process TPC data with separate temporal and static features.

    Returns list of ``((ts_tensor, static_tensor), label)`` tuples.
    """
    ts_list = data['x_ts']
    static_list = data['x_static']

    if isinstance(ts_list, np.ndarray):
        ts_list = ts_list.tolist()
    else:
        ts_list = list(ts_list)
    if isinstance(static_list, np.ndarray):
        static_list = static_list.tolist()
    else:
        static_list = list(static_list)

    ts_samples = [torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in ts_list]
    static_samples = [torch.as_tensor(np.asarray(s), dtype=torch.float32) for s in static_list]

    # Pad variable-length temporal sequences
    if ts_samples and ts_samples[0].dim() == 2:
        shapes = [s.shape[0] for s in ts_samples]
        if len(set(shapes)) > 1:
            max_len = max(shapes)
            n_feat = ts_samples[0].shape[1]
            padded = []
            for s in ts_samples:
                if s.shape[0] < max_len:
                    pad = torch.zeros(max_len - s.shape[0], n_feat, dtype=s.dtype)
                    s = torch.cat([s, pad], dim=0)
                padded.append(s)
            ts_samples = padded

    labels = torch.as_tensor(data['y'], dtype=torch.int64)
    return [((ts, static), label) for ts, static, label in zip(ts_samples, static_samples, labels)]


# import numpy as np
# import os
# import torch
# from collections import defaultdict


# def read_data(dataset, idx, is_train=True):
#     if is_train:
#         data_dir = os.path.join('../dataset', dataset, 'train/')
#     else:
#         data_dir = os.path.join('../dataset', dataset, 'test/')

#     file = data_dir + str(idx) + '.npz'
#     with open(file, 'rb') as f:
#         data = np.load(f, allow_pickle=True)['data'].tolist()
#     return data


# def read_client_data(dataset, idx, is_train=True, few_shot=0):
#     data = read_data(dataset, idx, is_train)
#     if "News" in dataset:
#         data_list = process_text(data)
#     elif "Shakespeare" in dataset:
#         data_list = process_Shakespeare(data)
#     else:
#         data_list = process_image(data)

#     if is_train and few_shot > 0:
#         shot_cnt_dict = defaultdict(int)
#         data_list_new = []
#         for data_item in data_list:
#             label = data_item[1].item()
#             if shot_cnt_dict[label] < few_shot:
#                 data_list_new.append(data_item)
#                 shot_cnt_dict[label] += 1
#         data_list = data_list_new
#     return data_list

# def process_image(data):
#     X = torch.Tensor(data['x']).type(torch.float32)
#     y = torch.Tensor(data['y']).type(torch.int64)
#     return [(x, y) for x, y in zip(X, y)]


# def process_text(data):
#     X, X_lens = list(zip(*data['x']))
#     y = data['y']
#     X = torch.Tensor(X).type(torch.int64)
#     X_lens = torch.Tensor(X_lens).type(torch.int64)
#     y = torch.Tensor(data['y']).type(torch.int64)
#     return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


# def process_Shakespeare(data):
#     X = torch.Tensor(data['x']).type(torch.int64)
#     y = torch.Tensor(data['y']).type(torch.int64)
#     return [(x, y) for x, y in zip(X, y)]

