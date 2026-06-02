# TRACE_STOP

**页面ID:** atlasascendc_api_07_1213  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1213.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。

用于表示终止位置打点，一般与TRACE_START配套使用。

> **注意:** 

该功能主要用于**调试和性能分析**，开启后会对算子性能产生一定影响，通常在调测阶段使用，**生产环境建议关闭**。

默认情况下，该功能关闭，开发者可以按需通过如下方式开启打点功能。

修改Kernel直调工程cmake目录下的npu_lib.cmake文件，在ascendc_compile_definitions命令中增加-DASCENDC_TRACE_ON编译选项，来开启打点功能。示例如下：

```
// 打开算子的打点功能
ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    -DASCENDC_TRACE_ON
)
```

#### 函数原型

```
#define TRACE_STOP(TraceId apid)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| apid | 输入 | 取值需与TRACE_START参数取值保持一致，否则影响打点结果。 |

#### 约束说明

- TRACE_START/TRACE_STOP需配套使用，若Trace图上未显示打点，则说明两者没有配对。
- 不支持跨核使用，例如TRACE_START在AI Cube打点，则TRACE_STOP打点也需要在AI Cube上，不能在AI Vector上。
- 宏支持所有的产品型号，但实际调用时需与调测工具支持的型号保持一致。
- 仅支持Kernel直调工程，不支持自定义算子工程下开启打点功能。

#### 调用示例

在Kernel代码中特定指令位置打上TRACE_START/TRACE_STOP：

```
TRACE_START(0x1);
DataCopy(zGm, zLocal, this->totalLength);
TRACE_STOP(0x1);
```
