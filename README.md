## LeNet5-CUDA

LeNet5中卷积层(Conv)，池化层(Pooling)，全连接层(Dense)的CUDA实现，GPU版本中，Conv和Dense层所有参数都在设备端分配，前向传播与后向传播分别通过核函数实现，Input、Loss和Output使用cudaMemcpy传递。

文件结构如下：

```
LeNet5-CUDA
│   README.md       本文件
│   main_cpu.cpp    CPU版本程序入口
│   main_gpu.cu     GPU版本程序入口
│   common.h        通用头文件
│   util.h          实现randn或其他工具的头文件
│   Makefile
│
└───layers
│   │   conv.h          Conv层的定义
│   │   conv.cu         Conv层的实现
│   │   dense.h         Dense层的定义
│   │   dense.cu        Dense层的实现
│   │   pooling.h       Pooling层的定义
│   └───pooling.cu      Pooling层的实现
│
└───logs
    │   cpu_1_2.txt     CPU版本运行log，input全部为1，loss全部为2
    │   gpu_1_2.txt     GPU版本运行log，input全部为1，loss全部为2
    │   cpu_norm.txt    CPU版本运行log
    └───gpu_norm.txt    GPU版本运行log
```
