import argparse
import logging
import math
import sys
import os
import random
import yaml
import time

from models import VGGNet

        
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