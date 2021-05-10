import argparse
import logging
import math
import sys
import os
import random
import yaml
import time

from models import VGGNet

# import neuralworkslib modules for ...... $ python m-512.py
sys.path.append(os.getenv("NW_HOME") + "/neulib")
import neuenv
from job import Job

NW_HOME = neuenv.NW_HOME
'''
self.history = torch_model.fit(x=train_data[0], y=train_data[1], 
                               validation_data=valid_data,
                               epochs=job.hyper_paras["epochs"], 
                               batch_size=job.hyper_paras["batch_size"],   # batch_size 추가함
                               callbacks=[mygetmetric, mysavebestweight])
ypred = torch_model.predict(test_data[0])
'''

        
def get_model(input_shape, job=None):
    nw_home = neuenv.NW_HOME
    num_categories = job.get_num_of_class()
    category_names = job.get_name_of_class()

    with open(f'{nw_home}/neusite/models/{job.model_id}/{job.model_id}.yaml') as f:
        setting = yaml.load(f, Loader=yaml.SafeLoader)

    setting['nc'] = num_categories
    setting['classes'] = category_names

    network = VGGNet(name='vgg19', ch=3, num_classes=num_categories, setting=setting, job=job)
    # print(network)

    return network


if __name__ == "__main__":
    with open(f'./m-550.yaml') as f:
        setting = yaml.load(f, Loader=yaml.SafeLoader)

    print("\n[Setting.yaml] - entire setting")
    pprint(setting, indent=2)
    
    print("\n[Setting.yaml] - model setting")
    pprint(setting['model'], indent=2)

    print("\n[Setting.yaml] - hyper-parameter setting")
    pprint(setting['hyp'], indent=2)

    print("\n[Setting.yaml] - train setting")
    pprint(setting['train'], indent=2)

    print("\n[Setting.yaml] - valid setting")
    pprint(setting['valid'], indent=2)

    print("\n[Setting.yaml] - test setting")
    pprint(setting['test'], indent=2)

    # print(setting['model']['base'][300])