# 如何在矢量编程时使能Vector Core

**页面ID:** atlas_ascendc_10_0100  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0100.html

---

针对
      Atlas 推理系列产品
     ，其硬件架构除了AI Core外，还额外设置了单独的Vector Core，作为AI Core中Vector计算单元的补充，从而缓解Vector计算瓶颈。Vector Core只包括了两种基础计算资源：向量计算单元（Vector Unit）和标量计算单元（Scalar Unit），分别用于完成向量与标量的数据计算。矢量算子开发时，使能Vector Core，算子执行时会同时启动AI Core和Vector Core，这些核并行执行相同的核函数代码。

本节将重点介绍如何使能
      Atlas 推理系列产品
     中的Vector Core。学习本节内容之前，建议您先熟悉算子实现、基于样例工程完成Kernel直调、工程化算子开发的相关内容，掌握基于AI Core的算子端到端开发流程。在此基础上本章将重点阐述使能Vector Core时的差异点。具体如下：

1. 完成算子kernel侧开发时，需要通过宏KERNEL_TASK_TYPE_DEFAULT使能Vector Core，算子执行时会同时启动AI Core和Vector Core， 此时AI Core会当成Vector Core使用。如下的代码样例展示了使能Vector Core的方法：

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usr = AscendC::GetUserWorkspace(workspace);
    KernelAdd op;
    op.Init(x, y, z, tilingData.blockDim, tilingData.totalLength, tilingData.tileNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE); // 使能VectorCore
    if (TILING_KEY_IS(1)) {
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        op.Process2();
    }
    // ...
}
```

2. 完成host侧tiling开发时，设置的blockDim代表的是AI Core和Vector Core的总数，比如用户在host侧设置blockDim为10，则会启动总数为10的AI Core和Vector Core；为保证启动Vector Core，设置数值应大于AI Core的核数。您可以通过GetCoreNumAic接口获取AI Core的核数，GetCoreNumVector接口获取Vector Core的核数。 如下代码片段，分别为使用kernel直调工程和自定义算子工程时的设置样例，此处设置为AI Core和Vector Core的总和，表示所有AI Core和Vector Core都启动。

  - kernel直调工程

```
auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
auto totalCoreNum = ascendcPlatform.GetCoreNumAic();
// ASCENDXXX请替换为实际的版本型号
if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
   totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
}
...
kernel_name<<<totalCoreNum , l2ctrl, stream>>>(argument list);
```

  - 自定义算子工程

```
// 配套的host侧tiling函数示例：
ge::graphStatus TilingFunc(gert::TilingContext* context)
{	
    // 使能VectorCore，将blockDim置为AI Core中vector核数 + Vector Core中的vector核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto totalCoreNum = ascendcPlatform.GetCoreNumAic();
    // ASCENDXXX请替换为实际的版本型号
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
       totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    context->SetBlockDim(totalCoreNum);
}
```

> **注意:** 

- 请参考Ascend C API中具体API支持的型号，来判断API接口是否支持
>          Atlas 推理系列产品
>         Vector Core。
- 支持Vector Core后，因为AI Core和Vector Core会分别执行，通过不同的任务进行调度，所以不支持核间同步指令，如IBSet、IBWait、SyncAll等。
- 算子计算溢出（输入inf/nan或计算结果超出范围）时，需注意AI Core和Vector Core结果表现不一致，AI Core仅支持饱和模式，Vector Core仅支持inf/nan模式。
