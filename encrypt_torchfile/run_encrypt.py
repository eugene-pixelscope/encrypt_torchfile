import argparse
from core import *


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_bin_path", type=str, default="model_file/a.bin")
    parser.add_argument("--passwd", type=str, required=True)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--salt_len", type=int, default=16)
    parser.add_argument("--hash_len", type=int, default=32)
    parser.add_argument("--time_cost", type=int, default=16)
    parser.add_argument("--memory_cost", type=int, default=2**20)
    parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args()
    return args

def main():
    args = make_parse()
    input_file_path = args.input_file_path
    output_bin_path = args.output_bin_path
    secret_key = args.passwd

    version = args.version
    salt_len = args.salt_len
    hash_len = args.hash_len
    time_cost = args.time_cost
    memory_cost = args.memory_cost
    parallelism = args.parallelism

    # ******************************
    # ENCRYPT
    # ******************************
    p = argon2.Parameters(
        type=argon2.low_level.Type.ID,
        version=version,
        salt_len=salt_len,
        hash_len=hash_len,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism
    )
    encrypt(secret_key=secret_key, argon2_parameters=p, in_file=input_file_path, out_file=output_bin_path)

if __name__ == '__main__':
    main()

