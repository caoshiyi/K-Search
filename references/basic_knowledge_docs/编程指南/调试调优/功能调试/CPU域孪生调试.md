# CPU域孪生调试

**页面ID:** atlas_ascendc_10_0073  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0073.html

---

本节介绍CPU域调试的方法：CPU侧验证核函数，gdb调试、使用printf命令打印。

> **注意:** 

- 当前您可以基于Kernel直调样例工程实现CPU域调试功能。
>        

异构编译场景，开发者使用命令行或者编写Cmake文件进行编译的情况，暂不支持CPU孪生调试并将在后续的版本中逐步支持。

- CPU调测过程中，配置日志相关环境变量，可以记录程序的运行过程及异常信息，有助于开发者进行功能调测。
>        

关于环境变量的使用约束以及详细说明，可参见日志。

#### CPU侧验证核函数

在非昇腾设备上，开发者可以利用CPU仿真环境先行进行算子开发和测试，并在准备就绪后，利用昇腾设备进行加速计算。在编译与运行章节，我们已经介绍了算子Kernel程序NPU域的编译运行。相比于NPU域的算子运行逻辑，CPU域调试，实际上是通过标准的GCC编译器编译算子Kernel程序。此时算子Kernel程序链接CPU调测库，执行编译生成的可执行文件，可以完成算子CPU域的运行验证。CPU侧的运行程序，通过GDB通用调试工具进行单步调试，可以精准验证程序执行流程是否符合预期。

**图1 **CPU域和NPU域的核函数运行逻辑对比
<!-- img2text -->
```
CPU域                                           NPU域
                                               
┌──────────────────────────┐         ┌──────────────────────────┐
│ Host                     │         │ Host                     │
│                          │         │                          │
│  ┌────────────────────┐  │         │  ┌────────────────────┐  │
│  │   Host APP程序     │  │         │  │   Host APP程序     │  │
│  └────────────────────┘  │         │  └────────────────────┘  │
│            ↓             │         │            ↓             │
│  ┌────────────────────┐  │         │  ┌────────────────────┐  │
│  │     CPU调测库      │  │         │  │   AscendCL API库   │  │
│  └────────────────────┘  │         │  └────────────────────┘  │
│            ↓             │         │                          │
│  ┌────────────────────┐  │         └──────────────────────────┘
│  │   算子Kernel程序   │  │
│  └────────────────────┘  │         ┌──────────────────────────┐
│            ↓             │         │ Device                   │
│  ┌────────────────────┐  │         │                          │
│  │    AscendC 类库    │  │         │  ┌────────────────────┐  │
│  └────────────────────┘  │         │  │   算子Kernel程序   │  │
└──────────────────────────┘         │  └────────────────────┘  │
                                     │            ↓             │
                                     │  ┌────────────────────┐  │
                                     │  │    AscendC 类库    │  │
                                     │  └────────────────────┘  │
                                     └──────────────────────────┘
```

基于Kernel直调样例工程，通过ACLRT_LAUNCH_KERNEL接口调用核函数时，可实现CPU与NPU域的代码的统一，且该方式仅支持以下型号：

- 
        Atlas A3 训练系列产品
       /
        Atlas A3 推理系列产品

- 
        Atlas A2 训练系列产品
       /
        Atlas A2 推理系列产品

- 
        Atlas 推理系列产品

如果通过<<<>>>调用核函数，则需要使用单独的调测接口进行内存分配等操作，并对CPU域和NPU域的代码进行宏隔离。

下面代码以add_custom算子为例，介绍算子核函数在CPU侧验证时，算子调用的应用程序如何编写（通过ACLRT_LAUNCH_KERNEL接口调用核函数的方式）。您在实现自己的应用程序时，需要关注由于算子核函数不同带来的修改，包括算子核函数名，入参出参的不同等，合理安排相应的内存分配、内存拷贝和文件读写等，相关API的调用方式直接复用即可。

1. 按需包含头文件。

```
#include "data_utils.h"
#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"
```

2. 应用程序框架编写。

```
int32_t main(int32_t argc, char* argv[])
{
    uint32_t blockDim = 8;
    size_t inputByteSize = 8 * 2048 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);
    // 运行算子的调用程序
    return 0;
}
```

3. 运行验证。

```
// 初始化
    CHECK_ACL(aclInit(nullptr));
    // 运行管理资源申请
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    // 分配Host内存
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputByteSize));
    // 分配Device内存
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // Host内存初始化
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);
    // 将数据从Host上拷贝到Device上
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // 用ACLRT_LAUNCH_KERNEL接口调用核函数完成指定的运算
    ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    // 将Device上的运算结果拷贝回Host
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);
    // 释放申请的资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    // 去初始化
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
```

> **注意:** 

为了实现CPU域与NPU域代码归一，仅对部分acl接口进行适配，开发者在使用CPU域调测功能时，仅支持使用如下acl接口：

- 有实际功能接口，支持CPU域调用
>         

  - aclDataTypeSize、aclFloat16ToFloat、aclFloatToFloat16。
  - aclrtMalloc、aclrtFree、aclrtMallocHost、aclrtFreeHost、aclrtMemset、aclrtMemsetAsync、aclrtMemcpy、aclrtMemcpyAsync、aclrtMemcpy2d、aclrtMemcpy2dAsync、aclrtCreateContext、aclrtDestroyContext。

- 无实际功能接口，打桩实现。
>         

  - Profiling数据采集
>           

aclprofInit、aclprofSetConfig、aclprofStart、aclprofStop、aclprofFinalize。

  - 系统配置
>           

aclInit、aclFinalize、aclrtGetVersion。

  - 运行时管理
>           

aclrtSetDevice、aclrtResetDevice、aclrtCreateStream、aclrtCreateStreamWithConfig、aclrtDestroyStream、aclrtDestroyStreamForce、aclrtSynchronizeStream、aclrtCreateContext、aclrtDestroyContext。

#### gdb调试

可使用**gdb**单步调试算子计算精度。由于cpu调测已转为多进程调试，每个核都会拉起独立的子进程，故gdb需要转换成子进程调试的方式。针对
       Atlas 推理系列产品
      、
        Atlas 训练系列产品
       ，每个核会拉起1个子进程。针对
       Atlas A2 训练系列产品
      /
       Atlas A2 推理系列产品
      ，每个核会拉起3个子进程，1个Cube，2个Vector。

- 调试单独一个子进程

启动gdb，示例中的add_custom_cpu为CPU域的算子可执行文件，参考修改并执行一键式编译运行脚本，将一键式编译运行脚本中的run-mode设置成cpu，即可编译生成CPU域的算子可执行文件。

       gdb启动后，首先设置跟踪子进程，之后再打断点，就会停留在子进程中，但是这种方式只会停留在遇到断点的第一个子进程中，其余子进程和主进程会继续执行直到退出。涉及到核间同步的算子无法使用这种方法进行调试。

```
gdb --args add_custom_cpu  // 启动gdb，add_custom_cpu为算子可执行文件
(gdb) set follow-fork-mode child
```

- 调试多个子进程

如果涉及到核间同步，那么需要能同时调试多个子进程。

在gdb启动后，首先设置调试模式为只调试一个进程，挂起其他进程。设置的命令如下：

```
(gdb) set detach-on-fork off
```

查看当前调试模式的命令为：

```
(gdb) show detach-on-fork
```

中断gdb程序要使用捕捉事件的方式，即gdb程序捕捉fork这一事件并中断。这样在每一次起子进程时就可以中断gdb程序。设置的命令为：

```
(gdb) catch fork
```

当执行r后，可以查看当前的进程信息：

```
(gdb) info inferiors
  Num  Description
* 1    process 19613
```

可以看到，当第一次执行fork的时候，程序断在了主进程fork的位置，子进程还未生成。

执行c后，再次查看info inferiors，可以看到此时第一个子进程已经启动。

```
(gdb) info inferiors
  Num  Description 
* 1    process 19613
  2    process 19626
```

这个时候可以使用切换到第二个进程，也就是第一个子进程，再打上断点进行调试，此时主进程是暂停状态：

```
(gdb) inferior 2
[Switching to inferior 2 [process 19626] ($HOME/demo)]
(gdb) info inferiors
  Num  Description
  1    process 19613
* 2    process 19626
```

请注意，inferior后跟的数字是进程的序号，而不是进程号。

如果遇到同步阻塞，可以切换回主进程继续生成子进程，然后再切换到新的子进程进行调试，等到同步条件完成后，再切回第一个子进程继续执行。

如下是调试一个单独子进程的命令样例：

```
gdb --args add_custom_cpu
set follow-fork-mode child
break add_custom.cpp:45
run
list
backtrace
print i
break add_custom.cpp:56
continue
display xLocal
quit
```

#### 使用printf打印命令打印

     在代码中直接编写printf(...)来观察数值的输出。样例代码如下：

```
printf("xLocal size: %d\n", xLocal.GetSize()); 
printf("tileLength: %d\n", tileLength);
```
