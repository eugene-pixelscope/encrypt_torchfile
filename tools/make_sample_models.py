import os
import copy
import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def save_model(model, optimizer):
    model_dir = os.path.join('..', 'model_file')
    make_folder(model_dir)
    out = os.path.join(model_dir, 'sample_model.tar')
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, out)


def load_model(net, checkpoint):
    # state_dict = OrderedDict()
    state_dict = copy.deepcopy(net.state_dict())
    for k, v in checkpoint['net'].items():
        if 'module' in k:
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        else:
            name = k
        state_dict[name] = v
    # model load
    net.load_state_dict(state_dict)
    return net


if __name__ == '__main__':
    model = SampleModel()
    model = DataParallel(model)
    # warm-up
    out = model(torch.randn(1, 2))
    print(out.shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_model(model, optimizer)
