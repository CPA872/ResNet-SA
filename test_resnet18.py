import torch

from resnet18_fp32 import myresnet18
from torchvision.models import ResNet18_Weights, resnet18

from time import time

weights = ResNet18_Weights.IMAGENET1K_V1

FPGA_resnet18 = myresnet18(weights=weights, progress=False)

CPU_resnet18 = resnet18(weights=weights, progress=False)
BATCH_SIZE = 64
dummy_input = torch.rand(BATCH_SIZE, 3, 224, 224, dtype=torch.float32)

print("Input shpae:", dummy_input.shape)

start = time()
out = FPGA_resnet18(dummy_input)
elapsed1 = time() - start

start = time()
out2 = CPU_resnet18(dummy_input)
elapsed2 = time() - start

assert out.shape == out2.shape
verify = torch.all(out.isclose(out2, atol=1e-4))
print(f"Verify result: {verify}")
assert verify
# print(weights.meta['categories'][out.argmax(dim=1)], target)
print(f"FPGA inference time = {elapsed1}s, CPU inference time = {elapsed2}")

print("Done")