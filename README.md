# Encrypt Pytorch Model

argon2와 fernet을 이용한 모델파일 암호화 레포지토리

## Dependencies
* [PyTorch](https://pytorch.org)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [Anaconda](https://www.anaconda.com/download/)
* argon2
* cryptography
* (optional) [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt.git)

## Installation

```shell
pip install -r requirements.txt
```
```shell
# install encrypt_torchfile
cd encrypt_torchfile
pip install -e .
cd ..
```

## Usage
암호화할 모델파일이 없다면 아래 명령어를 통해 샘플모델파일을 생성합니다.
```shell
python tools/make_sample_models.py
```
### encrypting torchfile
```shell
# encrypt-torchfile --input_file_path model_file/sample_model.tar --passwd pxscope
encrypt-torchfile --input_file_path [INPUT_FILE_PATH] --passwd [PASSWORD] \
--version [VERSION] --salt_len [SALT_LENGTH] \
--hash_len [HASH_LENGTH] --time_cost [TIME_COST] \
--memory_cost [MEMORY_COST] --parallelism [PARALLELISM]
```
### example of decrypting and run model
```shell
python main.py --input_file_path model_file/sample_model.tar --output_bin_path model_file/a.bin --passwd pxscope
```

### encrypted model load
```python
from encrypt_torchfile.core import decrypt
import io
import torch

# model define
net = ...

# Load encrypted model
model_buf = decrypt(secret_key=secret_key, in_file=output_bin_path)
model_buf = io.BytesIO(model_buf)
model_buf.seek(0)
state_dict = torch.load(model_buf, weights_only=True)

net.load_state_dict(state_dict)
```
## reference
* [Blog-Worried about your Deep Learning model being stolen or tampered with? Worry no more…](https://stephaniemaluso.medium.com/worried-about-your-deep-learning-model-being-stolen-or-tampered-with-worry-no-more-2f3d442a49cf)
* [doc-Argon2](https://argon2-cffi.readthedocs.io/en/stable/argon2.html)