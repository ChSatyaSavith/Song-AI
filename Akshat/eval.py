'''
Functions to get trained models from the 'models' directory and run them
on test datasets. Basically all stuff for model inference goes here.
'''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim


def lstm_vae_predict(model):
    pass

def embedding_predict(model):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/whatever', help='Path to trained model')
    # more arguments like dataset, whatever else we'll need
    args = parser.parse_args()
    # Call the appropriate predict function and do other shit