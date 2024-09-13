from setuptools import setup, find_packages


setup(
    name="encrypt_torchfile",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["cryptography==43.0.1", "argon2-cffi==23.1.0"],
    entry_points={
        "console_scripts": [
            "encrypt-torchfile=run_encrypt:main"
    ]},
)