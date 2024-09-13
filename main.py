import torch
import io
import argparse
from tools.make_sample_models import SampleModel
from encrypt_torchfile.core import *


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_bin_path", type=str, default="model_file/a.bin")
    parser.add_argument("--passwd", type=str, required=True)
    parser.add_argument("--trt", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = make_parse()
    input_file_path = args.input_file_path
    output_bin_path = args.output_bin_path
    secret_key = args.passwd
    trt_mode = args.trt
    p = argon2.Parameters(
        type=argon2.low_level.Type.ID,
        version=0,
        salt_len=16,
        hash_len=32,
        time_cost=16,
        memory_cost=2 ** 20,
        parallelism=8
    )

    input_tensor = torch.randn(1, 1, 2).cuda()
    # ******************************
    # DECRYPT
    # ******************************
    # Load encrypted model
    model_buf = decrypt(secret_key=secret_key, in_file=output_bin_path)
    model_buf = io.BytesIO(model_buf)
    model_buf.seek(0)
    loaded_state_dict = torch.load(model_buf)
    if trt_mode:
        from torch2trt import TRTModule
        model = TRTModule()
        encrypted_model = TRTModule()
    else:
        model = SampleModel().eval().cuda()
        encrypted_model = SampleModel().eval().cuda()

    state_dict = torch.load(input_file_path)
    model.load_state_dict(state_dict)
    encrypted_model.load_state_dict(loaded_state_dict)

    output_tensor = model(input_tensor)
    output_bin_tensor = encrypted_model(input_tensor)
    print(f'output_tensor: {output_tensor}')
    print(f'output_bin_tensor: {output_bin_tensor}')

    print(f'error: {torch.max(output_tensor-output_bin_tensor).detach().cpu().numpy()}')
