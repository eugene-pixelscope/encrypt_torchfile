import os
import torch
import torch.nn as nn


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


if __name__ == '__main__':
    model = SampleModel().eval().cuda()

    model_file_name = 'sample_model.pth'
    # warm-up
    x = torch.randn(1, 1, 2).cuda()
    out = model(x)
    torch.save(model.state_dict(), os.path.join('..', 'model_file', model_file_name))

    # torch2trt
    from torch2trt import torch2trt, TRTModule
    trt_model_file_name = 'sample_model_trt.pt'
    model_trt = torch2trt(model, [x])
    out_trt = model_trt(x)
    torch.save(model_trt.state_dict(), os.path.join('..', 'model_file', trt_model_file_name))
    del model
    del model_trt

    # check trt_model
    ## load model
    state_dict = torch.load(os.path.join('..', 'model_file', model_file_name), weights_only=False)
    model = SampleModel().eval().cuda()
    model.load_state_dict(state_dict)

    ## load trt_model
    state_trt_dict = torch.load(os.path.join('..', 'model_file', trt_model_file_name), weights_only=False)
    model_trt = TRTModule()
    model_trt.load_state_dict(state_trt_dict)
    print(f'trt-converted error: {torch.max(model(x) - model_trt(x)).detach().cpu().numpy()}')
