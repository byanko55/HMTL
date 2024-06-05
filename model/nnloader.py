import sys
import json

import torch


_module_mapping = {
    'Conv2d': torch.nn.Conv2d,
    'MaxPool2d': torch.nn.MaxPool2d,
    'ReLU': torch.nn.ReLU,
    'Dropout2d': torch.nn.Dropout2d,
    'Linear': torch.nn.Linear,
    'LogSoftmax': torch.nn.LogSoftmax,
    'Flatten': torch.nn.Flatten
}


def load_layer(name: str, params: dict) -> torch.nn.Module:
    """
    # In
      - name: layer name
      - params: <class: dict>, kwargs to build a single neural layer
    # Out
      - md: 
    """
    if name not in _module_mapping:
        raise TypeError("[nnloader: load_layer] layer type %s does not exist"%(name))

    try: 
        md = _module_mapping[name](**params)
    except TypeError:
        unknown_arg = str(sys.exc_info()[1]).split(' ')[-1]
        raise AttributeError("[nnloader: load_layer] layer type '%s' got an unexpected keyword argument %s"%(name, unknown_arg))

    return md


def load_nn(md_file: str) -> torch.nn.Sequential:
    with open(md_file) as f:
        md = json.load(f)

        ns = torch.nn.Sequential()

        for lname, largs in md['nnet_seq'].items():
            md = load_layer(largs['type'], largs['params'])
            ns.add_module(lname, md)

        return ns