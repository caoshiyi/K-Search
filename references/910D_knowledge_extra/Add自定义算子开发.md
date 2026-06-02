# Add自定义算子开发<a name="ZH-CN_TOPIC_0000002554332357"></a>

本入门教程，将会引导你完成以下任务，体验Ascend C  SIMT算子开发基本流程。

1.  算子分析，明确数学表达式和计算逻辑等内容；
2.  Add算子核函数开发；
3.  算子核函数运行验证。

在正式的开发之前，需要先完成环境准备工作，开发Ascend C算子的基本流程如下图所示：

**图 1**  开发Ascend C算子的基本流程<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_fig72899531739"></a>  
<!-- img2text -->
```text
┌──────────────────────────────┐
│           环境准备           │
└──────────────────────────────┘
               │
               ↓
┌──────────────────────────────┐
│           算子分析           │
└──────────────────────────────┘
               │
               ↓
┌──────────────────────────────┐
│         核函数开发           │
└──────────────────────────────┘
               │
               ↓
┌──────────────────────────────┐
│       核函数运行验证         │
└──────────────────────────────┘
```

> **说明：** 
>-   请点击[LINK](https://gitcode.com/cann/asc-devkit/tree/master/examples/00_introduction/01_add/simt_add)获取样例代码。
>-   使用本教程只需要您具有一定的C/C++基础，在此基础上，如果您已经对Ascend C  SIMT编程模型有一定的了解，您可以在实际操作的过程中加深对理论的理解；如果您还没有开始了解Ascend C  SIMT编程模型，也无需担心，您可以先尝试跑通教程中的样例，参考教程最后的[指引](#section2099191619557)进行进一步的学习。

## 环境准备<a name="zh-cn_topic_0000001565030288_zh-cn_topic_0000001552186366_section412999115218"></a>

-   CANN软件安装

    开发算子前，需要先准备好开发环境和运行环境，开发环境和运行环境的介绍和具体的安装步骤可参见《CANN软件安装指南》。

-   环境变量配置

    安装CANN软件后，使用CANN运行用户进行编译、运行时，需要以CANN运行用户登录环境，执行**source $_\{INSTALL\_DIR\}_/set\_env.sh**命令设置环境变量。$\{INSTALL\_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

## 算子分析<a name="section8240237193518"></a>

主要分析算子的数学表达式、输入输出的数量、Shape范围以及计算逻辑的实现，明确需要调用的Ascend C SIMT接口或操作符。下文以Add算子为例，介绍具体的分析过程。

1.  明确算子的数学表达式及计算逻辑。

    Add算子的数学表达式为：

    <!-- img2text -->
$$z_i = x_i + y_i,\ i = 0,1,2,\ldots,n-1$$

    计算逻辑是：逐元素将外部存储Global Memory对应位置上的输入x与y相加，结果存储在Global Memory输出z上。

2.  明确输入和输出。
    -   Add算子有两个输入：x与y，输出为z。
    -   本样例中算子输入支持的数据类型为float，算子输出的数据类型与输入数据类型相同。
    -   算子的输入、输出shape为（48，256）。

3.  确定核函数名称和参数。
    -   本样例中核函数命名为add\_custom。
    -   根据对算子输入输出的分析，确定核函数有3个输入输出参数x，y，z， 数据类型均为float。
    -   增加一个核函数入参total\_length，用于记录算子实际的输入、输出数据长度，数据类型为uint64\_t。

4.  确定算子实现逻辑。
    -   将数据均分到48个Thread Block上，每个Thread Block起256个线程处理256个元素，每个线程处理一个元素。
    -   通过每个线程独有的线程索引，计算当前线程需要处理的数据的偏移量。

通过以上分析，得到Ascend C  SIMT实现的Add算子的设计规格如下：

**表 1**  Add算子输入输出规格

<a name="table4934296305"></a>
<table><thead align="left"><tr id="row59358913304"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.1"><p id="p43713117344"><a name="p43713117344"></a><a name="p43713117344"></a><strong id="b143710311340"><a name="b143710311340"></a><a name="b143710311340"></a>name</strong></p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.2"><p id="p163763103418"><a name="p163763103418"></a><a name="p163763103418"></a><strong id="b9371831123410"><a name="b9371831123410"></a><a name="b9371831123410"></a>shape</strong></p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.3"><p id="p203733163411"><a name="p203733163411"></a><a name="p203733163411"></a><strong id="b237143103416"><a name="b237143103416"></a><a name="b237143103416"></a>data type</strong></p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.4"><p id="p133723116344"><a name="p133723116344"></a><a name="p133723116344"></a><strong id="b1737231143419"><a name="b1737231143419"></a><a name="b1737231143419"></a>format</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row393589203016"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p18371131183417"><a name="p18371131183417"></a><a name="p18371131183417"></a>x（输入）</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p153773113419"><a name="p153773113419"></a><a name="p153773113419"></a>48 * 256</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p193773114347"><a name="p193773114347"></a><a name="p193773114347"></a>float*</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p73793143411"><a name="p73793143411"></a><a name="p73793143411"></a>ND</p>
</td>
</tr>
<tr id="row6935119173013"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p10371231163414"><a name="p10371231163414"></a><a name="p10371231163414"></a>y（输入）</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p970912419148"><a name="p970912419148"></a><a name="p970912419148"></a>48 * 256</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p555064991411"><a name="p555064991411"></a><a name="p555064991411"></a>float*</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p173711317346"><a name="p173711317346"></a><a name="p173711317346"></a>ND</p>
</td>
</tr>
<tr id="row59354943016"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p837131103419"><a name="p837131103419"></a><a name="p837131103419"></a>z（输出）</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p845744121412"><a name="p845744121412"></a><a name="p845744121412"></a>48 * 256</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p937173114348"><a name="p937173114348"></a><a name="p937173114348"></a>float*</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p143718310348"><a name="p143718310348"></a><a name="p143718310348"></a>ND</p>
</td>
</tr>
<tr id="row33165171711"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p14318513177"><a name="p14318513177"></a><a name="p14318513177"></a>total_length</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p5319531715"><a name="p5319531715"></a><a name="p5319531715"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p3367187151814"><a name="p3367187151814"></a><a name="p3367187151814"></a>uint64_t</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p153112510171"><a name="p153112510171"></a><a name="p153112510171"></a>-</p>
</td>
</tr>
</tbody>
</table>

-   核函数名称：add\_custom
-   算子实现文件名称：add.asc

## 核函数开发<a name="section724017377351"></a>

通过当前线程块索引blockIdx、单个线程块包含的线程数blockDim、当前线程索引threadIdx计算获得当前线程的索引，以当前线程索引作为当前计算数据行的偏移量。

```
int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
```

通过下标偏移和加法运算符，计算该偏移位置的数据相加的结果，并将结果写入到输出中。

```
z[idx] = x[idx] + y[idx];
```

完整的核函数代码实现如下所示：

```
__global__ void add_custom(float* x, float* y, float* z, uint64_t total_length)
{
    // Calculate global thread ID
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Maps to the row index of output tensor
    if (idx >= total_length) {
        return;
    }
    z[idx] = x[idx] + y[idx];
}
```

## 核函数运行验证<a name="section5992124285511"></a>

完成Kernel侧核函数开发后，即可编写Host侧的核函数调用程序。实现从Host侧的APP程序调用算子，执行计算过程。

1.  Host侧应用程序框架的编写。

    ```
    //Host调用需要的头文件
    #include <vector>
    #include "acl/acl.h"
    
    //核函数开发部分
    __global__ void add_custom(float* x, float* y, float* z, uint64_t total_length)
    {
        ...
    }
    
    // 通过<<<...>>>内核调用符调用算子
    std::vector<float> add(std::vector<float>& x, std::vector<float>& y)
    {
        ...
        // Calc splite params
        uint32_t block_num = 48;
        uint32_t thread_num_per_block = 256;
        uint32_t dyn_ubuf_size = 0;  // No need to alloc dynamic memory.
        // Call kernel funtion with <<<...>>>
        add_custom<<<block_num, thread_num_per_block, dyn_ubuf_size, stream>>>(x_device, y_device, z_device, x.size());
        ...
        return output;
    }
    
    // 计算结果比对
    uint32_t verify_result(std::vector<float>& output, std::vector<float>& golden)
    {
        if (std::equal(output.begin(), output.end(), golden.begin())) {
            std::cout << "[Success] Case accuracy is verification passed." << std::endl;
            return 0;
        } else {
            std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
            return 1;
        }
        return 0;
    }
    
    // 验证算子主程序
    int32_t main(int32_t argc, char* argv[])
    {
        constexpr uint32_t in_shape = 48 * 256;
        std::vector<float> x(in_shape);
        std::vector<float> y(in_shape);
        std::vector<float> golden(in_shape);
        ...
        std::vector<float> output = add(x, y);
        return verify_result(output, golden);
    }
    ```

2.  编写**通过<<<...\>\>\>内核调用符调用**算子的代码。

    **图 2**  调用步骤<a name="zh-cn_topic_0000001565030288_fig1034911325594"></a>  
    <!-- img2text -->
```text
┌──────────────────────────────┐
│            初始化            │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│       运行管理资源申请       │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│ 分配Host内存，并进行数据初始化 │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│ 分配Device内存，并将数据从    │
│ Host上拷贝到Device上         │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│ 用内核调用符<<<>>>调用核函   │
│ 数完成指定的运算，并同步等待 │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│   将Device上的运算结果       │
│   拷贝回Host                │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│        释放申请的资源        │
└──────────────────────────────┘
               ↓
┌──────────────────────────────┐
│           去初始化           │
└──────────────────────────────┘
```

    如下示例中的acl API使用方法请参考《应用开发指南 \(C&C++\)》中的“acl API参考”章节。

    ```
    std::vector<float> add(std::vector<float>& x, std::vector<float>& y)
    {
        size_t total_byte_size =x.size() * sizeof(float);
        int32_t device_id = 0;
        aclrtStream stream = nullptr;
        uint8_t* x_host = reinterpret_cast<uint8_t *>(x.data());
        uint8_t* y_host = reinterpret_cast<uint8_t *>(y.data());
        uint8_t* z_host = nullptr;
        float* x_device = nullptr;
        float* y_device = nullptr;
        float* z_device = nullptr;
        // Init
        aclInit(nullptr);
        aclrtSetDevice(device_id);
        aclrtCreateStream(&stream);
        // Malloc memory in host and device
        aclrtMallocHost((void **)(&z_host), total_byte_size);
        aclrtMalloc((void **)&x_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&y_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc((void **)&z_device, total_byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemcpy(x_device, total_byte_size, x_host, total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(y_device, total_byte_size, y_host, total_byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
        // Calc splite params
        uint32_t block_num = 48;
        uint32_t thread_num_per_block = 256;
        uint32_t dyn_ubuf_size = 0;  // No need to alloc dynamic memory.
        // Call kernel funtion with <<<...>>>
        add_custom<<<block_num, thread_num_per_block, dyn_ubuf_size, stream>>>(x_device, y_device, z_device, x.size());
        aclrtSynchronizeStream(stream);
        // Copy result from device to host
        aclrtMemcpy(z_host, total_byte_size, z_device, total_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
        std::vector<float> output((float *)z_host, (float *)(z_host + total_byte_size));
        // Free memory
        aclrtFree(x_device);
        aclrtFree(y_device);
        aclrtFree(z_device);
        aclrtFreeHost(z_host);
        // DeInt
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_id);
        aclFinalize();
        return output;
    }
    ```

3.  使用bisheng -x dpp编译代码，具体编译配置如下：

    ```
    CANN_PATH=$(printenv ASCEND_HOME_PATH)
    OUTPUT="demo"
    echo "ASCEND_HOME_PATH: $CANN_PATH"
    bisheng \
      -x dpp --cce-aicore-arch=dav-c310-vec \
      -std=c++17 \
       gather.asc \
      -I${CANN_PATH}/include \
      -I${CANN_PATH}/include/ascendc/host_api \
      -I${CANN_PATH}/compiler/ascendc/include/highlevel_api \
      -I${CANN_PATH}/compiler/tikcpp/tikcfw \
      -I${CANN_PATH}/compiler/tikcpp/tikcfw/lib \
      -I${CANN_PATH}/compiler/tikcpp/tikcfw/lib/matmul \
      -I${CANN_PATH}/compiler/tikcpp/tikcfw/impl \
      -I${CANN_PATH}/compiler/tikcpp/tikcfw/interface \
      -L${CANN_PATH}/lib64 \
      -lascendc_runtime \
      -lascendcl \
      -lruntime \
      -lregister \
      -lerror_manager \
      -lprofapi \
      -lascendalog \
      -lmmpa \
      -lascend_dump \
      -lc_sec \
      -lstdc++ \
      -o ${OUTPUT}
    ```

4.  运行步骤如下

    ```
    ./demo
    ```

    > **说明：** 
    >-   该样例仅支持如下型号：
    >    -   Ascend 950PR/Ascend 950DT
    >-   -x dpp为纯SIMT场景的编译方式。
    >-   --cce-aicore-arch用于指定NPU的架构版本，dav-后为架构版本号，各AI处理器型号对应的架构版本号请通过[AI处理器型号和\_\_NPU\_ARCH\_\_的对应关系](SIMD-BuiltIn关键字和API.md#table65291052154114)进行查询。

## 接下来的引导<a name="section2099191619557"></a>

如果您想了解更多SIMT编程相关概念，可以参考[AI Core纯SIMT编程](AI-Core纯SIMT编程.md)学习基本概念，再来回顾本教程；如果您已经了解相关概念，并跑通了该样例，您可以参考[纯SIMT算子实现](纯SIMT算子实现.md)了解Ascend C  SIMT编程中的更多细节。

