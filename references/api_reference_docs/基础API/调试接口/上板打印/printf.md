# printf

**页面ID:** atlasascendc_api_07_0193  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0193.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

该接口提供CPU域/NPU域调试场景下的格式化输出功能。

在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。样例如下：

```
#include "kernel_operator.h"
AscendC::printf("fmt string %d\n", 0x123);
AscendC::PRINTF("fmt string %d\n", 0x123);
```

> **注意:** 

printf（PRINTF）接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过设置ASCENDC_DUMP=0的方式关闭打印功能。

#### 函数原型

```
template <class... Args>
__aicore__ inline void printf(__gm__ const char* fmt, Args&&... args)
template <class... Args>
__aicore__ inline void PRINTF(__gm__ const char* fmt, Args&&... args)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| fmt | 输入 | 格式控制字符串，包含两种类型的对象：普通字符和转换说明。 - 普通字符将原样不动地打印输出。- 转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。支持的转换类型包括：  - %d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t  - %f：输出实数，支持打印的数据类型：float/half/bfloat16_t  - %x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t  - %s：输出字符串  - %u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t  - %p：输出指针地址   **注意**： - 上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。- 在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位，CPU域上的输出为32位。 |
| args | 输入 | 附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。 |

#### 约束说明

- 本接口不支持打印除换行符之外的其他转义字符。
- 如果开发者需要包含标准库头文件stdio.h和cstdio，请在kernel_operator.h头文件之前包含，避免printf符号冲突。
- 单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。在使用自定义算子工程进行工程化算子开发时，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1MB。请开发者自行控制待打印的内容数据量，超出则不会打印。
- 根据算子执行方式的不同，printf的打印结果输出方式不同。动态图或者单算子直调场景下，待输出内容会被解析并打印在屏幕上；静态图场景下，整图算子需要全下沉到NPU侧执行，无法直接调用接口打印出单个算子的信息，因此需要在模型执行完毕后，将待输出内容落盘在dump文件中，dump文件需要通过工具解析为可读内容。

  - dump文件落盘路径按照优先级排列如下：

    - 如果开启了Data Dump功能，dump文件落盘到开发者配置的dump_path路径下。如何开启Dump功能依赖于具体的网络运行方式。以TensorFlow在线推理为例，通过enable_dump、dump_path、dump_mode等参数进行配置。配置方式可参考《TensorFlow 2.6.5模型迁移指南》中的API参考 > TF Adapter 接口（2.x）> npu.global_options > 配置参数说明章节。
    - 如果未开启Data Dump功能，但配置了ASCEND_WORK_PATH环境变量，dump文件落盘到ASCEND_WORK_PATH下的printf目录下。
    - 如果未开启Data Dump功能也没有配置ASCEND_WORK_PATH环境变量，dump文件落盘到当前程序执行目录下的printf路径下。

  - 落盘dump文件需要使用工具解析为用户可读内容：使用show_kernel_debug_data工具将dump二进制文件解析为用户可读内容，命令格式如下。

```
**show_kernel_debug_data**  *bin_file*  *output_dir*
```

- 算子入图场景，若一个动态Shape模型中有可下沉的部分，框架内部会将模型拆分为动态调度和下沉调度（静态子图）两部分，静态子图中的算子不支持该printf特性。

#### 调用示例

```
#include "kernel_operator.h"

// 整型打印：
AscendC::printf("fmt string %d\n", 0x123);
AscendC::PRINTF("fmt string %d\n", 0x123);

// 浮点型打印：
float a = 3.14;
AscendC::printf("fmt string %f\n", a);
AscendC::PRINTF("fmt string %f\n", a);

// 指针打印：
int *b;
AscendC::printf("TEST %p\n", b);
AscendC::PRINTF("TEST %p\n", b);
```

NPU模式下，程序运行时打印效果如下（CANN Version和TimeStamp仅在使用自定义算子工程时才会打印）：

```
CANN Version: XXX.XX, TimeStamp: 202408
fmt string 291
fmt string 291
fmt string 3.140000
fmt string 3.140000
TEST 0x12c08001a000
TEST 0x12c08001a000
```
