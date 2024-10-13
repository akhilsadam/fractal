import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def consistent(d1,d2):
    for k in d2.keys():
        if 'train' in k or 'test' in k:
            continue
        assert d1.__dict__[k] == d2[k], f'{k} mismatch: {d1.__dict__[k]} != {d2[k]}; retraining model from scratch'