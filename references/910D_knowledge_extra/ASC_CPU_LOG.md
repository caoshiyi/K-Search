# ASC\_CPU\_LOG<a name="ZH-CN_TOPIC_0000002554344753"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

提供Host侧打印Log的功能。开发者可以在算子的TilingFunc代码中使用ASC\_CPU\_LOG\_XXX接口来输出相关内容。一般情况下，开发者也可以选择使用printf等Host侧通用的打印方式进行调试。然而，在Tiling下沉场景中，由于Tiling函数运行在AI CPU上，必须使用本接口进行打印。

-   非Tiling下沉场景，日志输出到plog中。比如，debug级别的日志输出到/root/ascend/log/debug/plog中，日志级别通过环境变量ASCEND\_GLOBAL\_LOG\_LEVEL控制。会打印日志级别、时间戳、日志所在代码行和日志所在函数名。
-   Tiling下沉场景，日志不会输出到plog中，而是需要落盘并进行解析。算子运行之前需要开启Dump功能，使得日志Dump功能生效。如何开启Dump功能依赖于具体的网络运行方式。以TorchAir图模式为例，需要配置enable\_dump、dump\_path、dump\_mode等Dump参数。详细说明可参考《PyTorch图模式使用指南\(TorchAir\)》中的“max-autotune模式功能\>算子输入输出dump功能（图模式）”章节。示例如下：

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

    算子运行完成后，在Dump数据存放路径下会有日志Dump文件生成，文件名命名规则格式为_\{op\_type\}.\{op\_name\}.\{taskid\}.\{stream\_id\}.\{timestamp\}_，其中_\{op\_type\}_表示算子类型，_\{op\_name\}_表示算子名称，_\{taskid\}_表示调用算子计算接口的taskId，_\{stream\_id\}_表示算子具体执行的流Id，_\{timestamp\}_表示时间戳。

## 需要包含的头文件<a name="section12341115212912"></a>

```
#include "utils/log/asc_cpu_log.h"
```

## 函数原型<a name="section620mcpsimp"></a>

```
#define ASC_CPU_LOG_ERROR(format, ...)
#define ASC_CPU_LOG_INFO(format, ...)
#define ASC_CPU_LOG_WARNING(format, ...)
#define ASC_CPU_LOG_DEBUG(format, ...)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>format</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1965816506312"><a name="p1965816506312"></a><a name="p1965816506312"></a>格式控制字符串，包含两种类型：普通字符和转换说明。</p>
<a name="ul419411543310"></a><a name="ul419411543310"></a><ul id="ul419411543310"><li>普通字符将直接输出。</li><li>转换说明用于控制参数的格式化输出。每个转换说明以百分号（%）开始，<span>后跟类型说明符，用于指定输出数据的类型。</span>支持的数据类型和C/C++规范保持一致。</li></ul>
</td>
</tr>
<tr id="row107121531114710"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p107123312476"><a name="p107123312476"></a><a name="p107123312476"></a>...</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p17712123120472"><a name="p17712123120472"></a><a name="p17712123120472"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1471223110478"><a name="p1471223110478"></a><a name="p1471223110478"></a>附加参数，数量和类型可变的参数列表。<span>其数量和类型需与格式控制字符串中的%标签数量和类型匹配。</span><span>每个参数将替换格式字符串中的相应%标签，以实现预期的输出效果。</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

Tiling下沉场景下，若使用旧版本CANN包（不支持ASC\_CPU\_LOG接口）生成的自定义算子工程，需特别注意兼容性问题，此时调用该接口无法输出日志。您可以通过查看自定义算子工程下cmake/device\_task.cmake中有无DEVICE\_OP\_LOG\_BY\_DUMP字段来确认当前工程是否支持日志Dump功能，如果未找到该字段，则需要重新生成自定义算子工程。

## 调用示例<a name="section837496171220"></a>

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

非Tiling下沉场景，打印会输出至xxxxxxx\_2025xxxxxxxxxxxxx.log中，结果示例如下：

```
[ERROR] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am ERROR log: 291
[INFO] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am INFO log: 291
[WARNING] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am WARNING log: 291
[DEBUG] ASCENDCKERNEL(xxx,execute_add_op):2025-xx-xx-xx:xx:xx.xxx.xxx [/xxx/xxx.cpp:xx][TilingFunc] I am DEBUG log: 291
```

