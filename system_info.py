#!/usr/bin/python
import importlib, platform

print(f'os: {platform.platform()}')
for m in ('stardist','csbdeep','tensorflow'):
    try:
        print(f'{m}: {importlib.import_module(m).__version__}')
    except ModuleNotFoundError:
        print(f'{m}: not installed')

import tensorflow as tf
try:
    print(f'tensorflow GPU: {tf.test.is_gpu_available()}')
except:
    print(f'tensorflow GPU: {tf.config.list_physical_devices("GPU")}')
