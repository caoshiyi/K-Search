# AI CPU编程<a name="ZH-CN_TOPIC_0000002523351572"></a>

AI CPU是位于Device侧ARM64架构的处理器，其具备与AI Core相同的内存访问能力，可直接访问Device侧内存资源；也可以与Host侧的CPU一样，进行类似的数据计算，通常作为AI Core的补充，主要承担非矩阵类、逻辑比较复杂的分支密集型计算。AI CPU的运行环境为基础的Linux环境，编程时可使用libc库，C++标准库，STL模板库等。其硬件架构图如下所示：

**图 1**  AI CPU硬件架构图<a name="fig1146075111341"></a>  
<!-- img2text -->
```text
┌───────────────────────┐               ┌──────────────────────────────────────────────────────────────┐
│                       │               │                                                              │
│         Host          │               │  ┌────────────┐      ┌────────────┐                         │
│        服务器         │──────────────→│  │   AI CPU   │      │  AI Core   │                         │
│                       │   PCIe接口    │  └────────────┘      └────────────┘                         │
│                       │               │                                                              │
└───────────────────────┘               │  ┌────────────────────────────────────────────────────────┐  │
                                        │  │                        L2 Cache                        │  │
                                        │  └────────────────────────────────────────────────────────┘  │
                                        │                                                              │
                                        │  ┌────────────────────────────────────────────────────────┐  │
                                        │  │                    Global Memory                      │  │
                                        │  └────────────────────────────────────────────────────────┘  │
                                        │                                                              │
                                        │                                   Device                     │
                                        │                                  AI 处理器                   │
                                        └──────────────────────────────────────────────────────────────┘
```

本节介绍的AI CPU编程仅支持如下产品型号：

-   Ascend 950PR/Ascend 950DT

## AI CPU核函数定义<a name="section4987175618443"></a>

在进行AI CPU编程时，与AI Core类似，同样需要定义设备侧函数入口（即核函数），该函数必须通过\_\_aicpu\_\_标识符进行声明，并且需与\_\_global\_\_标识符联合使用以表明其只能被Host侧调用。AI CPU的Device侧实现文件需要以.aicpu为后缀（或者在编译时增加-x aicpu选项）。该实现文件中包括上面介绍的核函数以及AI CPU普通函数定义，AI CPU普通函数无需添加执行空间标识符。

如下是一个AI CPU“Hello World”程序的示例，hello\_world.aicpu文件内容如下：

```
// 调用printf接口需要包含的头文件
#include "aicpu_api.h"

__global__ __aicpu__ uint32_t hello_world(void *args)
{
    AscendC::printf("Hello World!!!\n");
    return 0;
}
```

> **说明：** 
>编程时需要遵循如下规范：
>-   \_\_aicpu\_\_ \_\_global\_\_函数不能是void返回类型，并且入参只能是一个指针。
>-   \_\_aicpu\_\_ \_\_global\_\_函数不能是类的成员函数，也不能存在于匿名空间下。
>-   尽管AI CPU的Kernel函数有返回值，但该返回值仅用于Runtime组件报告运行状态，开发者无需编写返回逻辑，也无法使用该返回值。因此，对于用户而言，AI CPU Kernel函数等同于void类型，不能作为右值使用。

## AI CPU核函数调用<a name="section178512255013"></a>

AI CPU核函数的调用需要在.asc文件中进行，和AI Core的算子调用类似，同样使用<<<\>\>\>语法。

```
hello_world<<<numBlocks, nullptr, stream>>>(&args, sizeof(KernelArgs));
```

-   numBlocks：AI CPU Device侧暂不支持分核逻辑，因此Host侧调用多核无实际意义。建议设置为1。
-   l2ctrl：保留参数，当前固定为nullptr，开发者无需关注。
-   stream：类型为aclrtStream，stream用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序在Device上执行。stream创建等管理接口请参考《应用开发指南 \(C&C++\)》中的“AscendCL API参考 \> 运行时管理 \> Stream管理”章节。

> **说明：** 
>在编写调用代码时需要遵循如下规范：
>-   \_\_aicpu\_\_ \_\_global\_\_函数不能在.asc文件中进行定义，只能声明，且需要使用extern。
>-   Host侧调用\_\_global\_\_ \_\_aicpu\_\_函数时必须使用<<<\>\>\>异构调用语法，输入的函数入参在入参指针的基础上需要输入从指针中读取的数据大小。
>-   在Host侧使用内核调用符<<<...\>\>\>调用AI Core与AI CPU算子时不能使用同一条stream。

加载和运行算子时，需要使用Runtime API，完成运行时管理和配置，详细内容请参考[算子运行](算子运行.md)。AI CPU算子的编译请参考[AI CPU算子编译](AI-CPU算子编译.md)。

## AI CPU模板核函数<a name="section135075471718"></a>

若需要使用模板核函数，则需要在.aicpu文件中给出模板核函数的实例化声明，参考如下：

```
template<typename T, int BUFF_SIZE>
__global__ __aicpu__ uint32_t hello_world(void *args)
{
    AscendC::printf("Hello World!!!\n");
    AscendC::printf("buffer_size is %d\n", BUFF_SIZE);
    return 0;
}
template __global__ __aicpu__ uint32_t hello_world<KernelArgs, 4096>(void *args);
```

并在.asc文件中新增模板核函数实例化的extern声明：

```
template<typename T, int BUFF_SIZE>
extern __global__ __aicpu__ uint32_t hello_world(void *args);

template extern __global__ __aicpu__ uint32_t hello_world<KernelArgs, 4096>(void *args);
```

## 更多进阶用法<a name="section109268265171"></a>

> **说明：** 
>更多AI CPU API的使用方法请参考[AI CPU API](AI-CPU-API.md)。

