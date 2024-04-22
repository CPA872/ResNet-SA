# ResNet On Systolic Array

## Implementations
`test_resnet118.py`: Full model inference tests, run `python3 test_resnet18.py` to see matching results and compare between FPGA and CPU performing full ResNet18 inference. 

`im2col.py`: This file contains im2col transformation implementations

`resset18_fp32.py` This file contains the FPGA host controller and our custom implementation of convolution and linear layers. It also includes the ResNet18 implementation with these custom layers. Using the new ResNet18 class with its `forward()` method automatically offloads all convolutions and linear layers to the systolic array on the FPGA. 

`utils.py`: This contains some dependencies methods. 


## Dependencies
Download our built bitstream here: https://drive.google.com/file/d/11-WMjltd5ekjU-s6g1W9vp8ePAcMkXLA/view?usp=sharing

The bitstream location, after unzipping, is `<your_workspace>/gemm_hls/build3_fp32_m64_k8_512/MatrixMultiplication_hw.xclbin`
You would need to change the XCLBIN_LOCATION variable for `resnet18_fp32.py` to ensure PYNQ loads the correct kernel. 

We use a Xilinx U280 FPGA in this project. It is very painful to find old versioned Xilinx stuff. 
(If you would like to recreate the results, it would be easier to contact me at yup014@ucsd.edu since I have all the environment ready)

Host Environment
- PYNQ 3.0.1
- Pytorch 1.13.1

FPGA Tools
- Xilinx Vitis 2021.1
- XRT Version 2.12
- Firmware: xilinx_u280_xdma_201920_3
Note: it is imperative to ensure the firmware and XRT match. Differences in versions will cause the bitstream to fail. Unfortunately, we tested that Vitis > 2021.1 has issues with the design so only older versions work. This also means the XRT and U280 firmware need to be kept old. 

Below is the output of `xbmgmt examine`, please ensure various versions listed under XRT and Device sections match with what you have. 
```
System Configuration (Yours will differ)
  OS Name              : Linux
  Release              : 4.18.0-305.25.1.el8_4.x86_64
  Version              : #1 SMP Mon Oct 18 14:34:11 EDT 2021
  Machine              : x86_64
  CPU Cores            : 16
  Memory               : 80391 MB
  Distribution         : Red Hat Enterprise Linux 8.4 (Ootpa)
  GLIBC                : 2.28
  Model                : MS-7D09

XRT (2.11 and 2.12 tested, others unknown)
  Version              : 2.12.427
  Branch               : 2021.2
  Hash                 : 2719b6027e185000fc49783171631db03fc0ef79
  Hash Date            : 2021-10-09 05:06:53
  XOCL                 : 2.12.427, 2719b6027e185000fc49783171631db03fc0ef79
  XCLMGMT              : 2.12.427, 2719b6027e185000fc49783171631db03fc0ef79

Devices present (xilinx_u280_xdma_201920_3 must match)
  [0000:01:00.0] : xilinx_u280_xdma_201920_3 mgmt(inst=256)
```

## Running
For full model inference, please run `python3 test_resnet18.py`
