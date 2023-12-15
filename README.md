# CSE240D-Final-Project

## Dependecies
We use a Xilinx U280 FPGA in this project. The dependencies is quite complicated. 
(If you would like to recreate the results, it would be easier to contact me at yup014@ucsd.edu since I have all environment ready)

Host Environment
- PYNQ 3.0.1
- Pytorch 1.13.1

FPGA Tools
- Xilinx Vitis 2021.1
- XRT Version 2.12
- Firmware: xilinx_u280_xdma_201920_3
Note: it is imperative to ensure the the firmware and XRT matches. Difference in versions will cause the bitstream to fail. 

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

Devices present (xilinx_u280_xdma_201920_3 must bmatch)
  [0000:01:00.0] : xilinx_u280_xdma_201920_3 mgmt(inst=256)
```

## Running
For full model inference, please run `python3 test_resnet18.py`
