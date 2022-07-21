import torch
import torch.nn as nn
import numpy as np
from complexity_metrics import get_gmacs_and_params, get_runtime
from hdr.archs.efficienthdr_v4_arch import EfficientHDR_V4
from hdr.archs.efficienthdr_v3_arch import EfficientHDR_V3
from hdr.archs.efficienthdr_v2_arch import EfficientHDR_V2
from hdr.archs.efficienthdr_v1_arch import EfficientHDR_V1
# from hdr.archs.efficienthdr_arch import EfficientHDRModel
# from hdr.archs.toy_arch import ToyHDRModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-wp", "--write_path", type=str, default="./complexity/", help="Path to write the readme.txt file")
args = parser.parse_args()
write_path=args.write_path

# Load a pytorch model
model_name = 'EfficientHDRModel'
# model = EfficientHDR_V1()
# model = EfficientHDR_V2()
# model = EfficientHDR_V3()
model = EfficientHDR_V4()
# model = ToyHDRModel()
model.eval()

# Calculate MACs and Parameters
# total_macs, total_params = get_gmacs_and_params(model, input_size=(1, 3, 3, 1060, 1900))
total_macs, total_params = get_gmacs_and_params(model, input_size=(1, 3, 3, 1088, 1920))
# mean_runtime = get_runtime(model, input_size=(1, 3, 3, 1088, 1920))

print('GMACs: ', total_macs)
print('Params: ', total_params)
# print('Runtime: ', mean_runtime)

# Print model statistics to txt file
# with open(write_path + f'{model_name}.txt', 'w') as f:
#     f.write("runtime per image [s] : " + str(mean_runtime))
#     f.write('\n')
#     f.write("number of operations [GMAcc] : " + str(total_macs))
#     f.write('\n')
#     f.write("number of parameters  : " + str(total_params))
#     f.write('\n')
#     f.write("Other description: Toy Model for demonstrating example code usage.")
# Expected output of the readme.txt for ToyHDRModel should be:
# runtime per image [s] : 0.013018618555068967
# number of operations [GMAcc] : 20.146042
# number of parameters  : 8243
# Other description: Toy Model for demonstrating example code usage.

# print("You reached the end of the calculate_ops_example.py demo script. Good luck participating in the NTIRE 2022 HDR Challenge!")

