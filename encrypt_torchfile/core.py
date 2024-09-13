import os.path

import argon2
import base64
from cryptography.fernet import Fernet
from logger import Logger

logger = Logger()

def encrypt(secret_key, argon2_parameters, in_file, out_file):
    logger.info(f'Start encrypting file: [{in_file}]')
    hasher = argon2.PasswordHasher(time_cost=argon2_parameters.time_cost,
                                   memory_cost=argon2_parameters.memory_cost,
                                   parallelism=argon2_parameters.parallelism,
                                   hash_len=argon2_parameters.hash_len,
                                   salt_len=argon2_parameters.salt_len)

    # generate the hash for the given secret_key
    secret_hash = hasher.hash(secret_key)
    salt = secret_hash.split("$")[-2]
    raw_hash = argon2.low_level.hash_secret_raw(time_cost=argon2_parameters.time_cost,
                                                memory_cost=argon2_parameters.memory_cost,
                                                parallelism=argon2_parameters.parallelism,
                                                hash_len=argon2_parameters.hash_len,
                                                secret=bytes(secret_key, "utf_16_le"),
                                                salt=bytes(salt, "utf_16_le"),
                                                type=argon2_parameters.type)

    key = base64.urlsafe_b64encode(raw_hash)
    fernet = Fernet(key)

    # save the password hash first before writing the encrypted data to the out_file
    with open(in_file, 'rb') as f_in, open(out_file, 'wb') as f_out:
        data = f_in.read()
        enc_data = fernet.encrypt(data)
        pw_hash = secret_hash + '\n'
        f_out.write(bytes(pw_hash, "utf-8"))
        f_out.write(enc_data)

    logger.info(f'End encrypting file: [{in_file}] (Output: {out_file})\n'
                f'--------------------\n'
                f'Encryption Parameters:\n'
                f'* version:\t{argon2_parameters.version}\n'
                f'* salt_len:\t{argon2_parameters.salt_len}\n'
                f'* hash_len:\t{argon2_parameters.hash_len}\n'
                f'* time_cost:\t{argon2_parameters.time_cost}\n'
                f'* memory_cost:\t{argon2_parameters.memory_cost}\n'
                f'* parallelism:\t{argon2_parameters.parallelism}\n'
                f'--------------------')

def decrypt(secret_key, in_file):
    logger.info(f'Start decrypting file: [{os.path.basename(in_file)}]')
    # Read the password hash from the encrypted file
    with open(in_file, 'r') as f:
        pw_hash = f.readline()[:-1]
    # Extract the Argon2 parameters from the hash
    p = argon2.extract_parameters(pw_hash)
    hasher = argon2.PasswordHasher(time_cost=p.time_cost,
                               memory_cost=p.memory_cost,
                               parallelism=p.parallelism,
                               hash_len=p.hash_len,
                               salt_len=p.salt_len)
    # Verify that the password used during encryption matches
    # with the one provided during decryption, if not stop
    try:
        hasher.verify(pw_hash, secret_key)
        logger.debug("Argon2 verify: true")
    except:
        logger.error("Argon2 verify: false, check secret_key")
        exit()

    # Extract the salt from the hash that will be used for generating the fernet key
    salt = pw_hash.split("$")[-2]
    # Generate the raw hash to be used as fernet key as done during encryption above
    raw_hash = argon2.low_level.hash_secret_raw(time_cost=p.time_cost,
                                    memory_cost=p.memory_cost,
                                    parallelism=p.parallelism,
                                    hash_len=p.hash_len,
                                    secret=bytes(secret_key, "utf_16_le"),
                                    salt=bytes(salt, "utf_16_le"),
                                    type=p.type)
    # base64 encode the raw hash (key)
    key = base64.urlsafe_b64encode(raw_hash)
    # Create the Fernet key for decryptin the files
    fernet = Fernet(key)
    dec_data = b''
    with open(in_file, 'rb') as f_in:
        enc_data = f_in.readlines()[1]
        try:
            dec_data = fernet.decrypt(enc_data)
        except:
            logger.error("decryption failed")
    logger.info(f'End decrypting file: [{os.path.basename(in_file)}]')
    return dec_data
