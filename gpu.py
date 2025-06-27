import os
import sys
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Replace 1 with the GPU index you want

print(torch.cuda.get_device_name(0))  # This will now refer to GPU 1 as "GPU 0" in the notebook context
print(torch.cuda.is_available())

print("Device count:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(0))
