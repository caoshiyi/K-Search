# ASC_CPU_LOG

**页面ID:** atlasascendc_api_07_00152  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00152.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

提供Host侧打印Log的功能。开发者可以在算子的TilingFunc代码中使用ASC_CPU_LOG_XXX接口来输出相关内容。一般情况下，开发者也可以选择使用printf等Host侧通用的打印方式进行调试。然而，在Tiling下沉场景中，由于Tiling函数运行在AI CPU上，必须使用本接口进行打印。

- 非Tiling下沉场景，日志输出到plog中。比如，debug级别的日志输出到/root/ascend/log/debug/plog中，日志级别通过环境变量ASCEND_GLOBAL_LOG_LEVEL控制。会打印日志级别、时间戳、日志所在代码行和日志所在函数名。
- Tiling下沉场景，日志不会输出到plog中，而是需要落盘并进行解析。算子运行之前需要开启Dump功能，使得日志Dump功能生效。如何开启Dump功能依赖于具体的网络运行方式。以TorchAir图模式为例，需要配置enable_dump、dump_path、dump_mode等Dump参数。详细说明可参考《PyTorch图模式使用指南(TorchAir)》中的“max-autotune模式功能 > 算子输入输出dump功能（图模式）”章节。示例如下：

```
import torch_npu, torchair
config = torchair.CompilerConfig()
# data dump开关：[必选]
config.dump_config.enable_dump = True
# dump类型：[可选]，all代表dump所有数据
config.dump_config.dump_mode = "all"
# dump路径：[可选]，缺省为当前目录
config.dump_config.dump_path = '/home/dump'
...
```

算子运行完成后，在Dump数据存放路径下会有日志Dump文件生成，文件名命名规则格式为*{op_type}.{op_name}.{taskid}.{stream_id}.{timestamp}*，其中*{op_type}*表示算子类型，*{op_name}*表示算子名称，*{taskid}*表示调用算子计算接口的taskId，*{stream_id}*表示算子具体执行的流Id，*{timestamp}*表示时间戳。

#### 需要包含的头文件

```
#include "utils/log/asc_cpu_log.h"
```

#### 函数原型

```
#define ASC_CPU_LOG_ERROR(format, ...)
#define ASC_CPU_LOG_INFO(format, ...)
#define ASC_CPU_LOG_WARNING(format, ...)
#define ASC_CPU_LOG_DEBUG(format, ...)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| format | 输入 | 格式控制字符串，包含两种类型：普通字符和转换说明。                     - 普通字符将直接输出。           - 转换说明用于控制参数的格式化输出。每个转换说明以百分号（%）开始，后跟类型说明符，用于指定输出数据的类型。支持的数据类型和C/C++规范保持一致。 |
| ... | 输入 | 附加参数，数量和类型可变的参数列表。其数量和类型需与格式控制字符串中的%标签数量和类型匹配。每个参数将替换格式字符串中的相应%标签，以实现预期的输出效果。 |

#### 约束说明

Tiling下沉场景下，若使用旧版本CANN包（不支持ASC_CPU_LOG接口）生成的自定义算子工程，需特别注意兼容性问题，此时调用该接口无法输出日志。您可以通过查看自定义算子工程下cmake/device_task.cmake中有无DEVICE_OP_LOG_BY_DUMP字段来确认当前工程是否支持日志Dump功能，如果未找到该字段，则需要重新生成自定义算子工程。

#### 调用示例

```
#include "utils/log/asc_cpu_log.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    ...
    ASC_CPU_LOG_ERROR("I am ERROR log: %d\n", 0x123);
    ASC_CPU_LOG_INFO("I am INFO log: %d\n", 0x123);
    ASC_CPU_LOG_WARNING("I am WARNING log: %d\n", 0x123);
    ASC_CPU_LOG_DEBUG("I am DEBUG log: %d\n", 0x123);
    ...
}
} // namespace optiling
```

     非Tiling下沉场景，打印会输出至xxxxxxx_2025xxxxxxxxxxxxx.log中，结果示例如下：

```
[ERROR] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am ERROR log: 291
[INFO] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am INFO log: 291
[WARNING] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am WARNING log: 291
[DEBUG] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am DEBUG log: 291
```
