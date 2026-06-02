# 设置Kernel类型

**页面ID:** atlasascendc_api_07_0218  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0218.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

用于用户自定义设置kernel类型，控制算子执行时只启动该类型的核，避免启动不需要工作的核，缩短核启动开销。

#### 函数原型

- 设置全局默认的kernel type，对所有的tiling key生效。

当前支持在自定义算子工程和Kernel直调工程中使用。

```
KERNEL_TASK_TYPE_DEFAULT(value)
```

- 设置某一个具体的tiling key对应的kernel type。

当前仅支持在自定义算子工程中使用。

```
KERNEL_TASK_TYPE(key, value)
```

#### 参数说明

**表1 **参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| key | 输入 | tiling key的key值，此参数是正数，表示某个核函数的分支。 |
| 设置的kernel类型，可选值范围，kernel类型具体说明请参考表2。                                                                                                                           ``` enum KernelMetaType {     KERNEL_TYPE_AIV_ONLY,     KERNEL_TYPE_AIC_ONLY,     KERNEL_TYPE_MIX_AIV_1_0,     KERNEL_TYPE_MIX_AIC_1_0,     KERNEL_TYPE_MIX_AIC_1_1,     KERNEL_TYPE_MIX_AIC_1_2,     KERNEL_TYPE_AICORE,     KERNEL_TYPE_VECTORCORE,     KERNEL_TYPE_MIX_AICORE,     KERNEL_TYPE_MIX_VECTOR_CORE,     KERNEL_TYPE_MAX }; ``` |  |  |

**表2 **kernel type取值说明

| 参数 | 说明 |
| --- | --- |
| KERNEL_TYPE_AIV_ONLY | 算子执行时仅启动AI Core上的Vector核：比如用户在host侧设置blockDim为10，则会启动10个Vector核。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：支持该参数                       Atlas 推理系列产品            AI Core：不支持该参数 |
| KERNEL_TYPE_AIC_ONLY | 算子执行时仅启动AI Core上的Cube核：比如用户在host侧设置blockDim为10，则会启动10个Cube核。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：支持该参数                       Atlas 推理系列产品            AI Core：不支持该参数 |
| KERNEL_TYPE_MIX_AIV_1_0 | AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX AIV:AIC 1:0（带有硬同步），算子执行时仅会启动AI Core上的Vector核，比如用户在host侧设置blockDim为10，则会启动10个Vector核。          硬同步的概念解释如下：当不同核之间操作同一块全局内存且可能存在读后写、写后读以及写后写等数据依赖问题时，通过调用SyncAll()函数来插入同步语句来避免上述数据依赖时可能出现的数据读写错误问题。目前多核同步分为硬同步和软同步，硬同步是利用硬件自带的全核同步指令由硬件保证多核同步。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：支持该参数                       Atlas 推理系列产品            AI Core：不支持该参数 |
| KERNEL_TYPE_MIX_AIC_1_0 | AIC、AIV混合场景下，使用了多核控制相关指令时，设置核函数的类型为MIX AIC:AIV 1:0（带有硬同步），算子执行时仅会启动AI Core上的Cube核，比如用户在host侧设置blockDim为10，则会启动10个Cube核。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：支持该参数                       Atlas 推理系列产品            AI Core：不支持该参数 |
| KERNEL_TYPE_MIX_AIC_1_1 | AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:1，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置blockDim为10，则会启动10个Cube核和10个Vector核。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：支持该参数                       Atlas 推理系列产品            AI Core：不支持该参数 |
| KERNEL_TYPE_MIX_AIC_1_2 | AIC、AIV混合场景下，设置核函数的类型为MIX AIC:AIV 1:2，算子执行时会同时启动AI Core上的Cube核和Vector核，比如用户在host侧设置blockDim为10，则会启动10个Cube核和20个Vector核。                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：支持该参数                       Atlas 推理系列产品            AI Core：不支持该参数 |
| KERNEL_TYPE_AICORE | 算子执行时仅会启动AI Core，比如用户在host侧设置blockDim为5，则会启动5个AI Core。                       Atlas 推理系列产品            ：支持该参数                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：不支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：不支持该参数 |
| KERNEL_TYPE_VECTORCORE | **该参数为预留参数，当前版本暂不支持。** |
| KERNEL_TYPE_MIX_AICORE | **该参数为预留参数，当前版本暂不支持。** |
| KERNEL_TYPE_MIX_VECTOR_CORE | 基于Ascend C开发的矢量计算相关的算子可以运行在Vector Core上，调用本接口传入该参数用于使能Vector Core。          使能Vector Core后，算子执行时会同时启动AI Core和Vector Core，用于并行计算。比如用户在host侧设置block_dim为10，则会启动总数为10的AI Core和Vector Core。          需要注意的是，通过SetBlockDim设置核数时，需要大于AI Core的核数，否则不会启动VectorCore。                       Atlas 推理系列产品            ：支持该参数                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ：不支持该参数                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ：不支持该参数 |

#### 约束说明

- **KERNEL_TASK_TYPE**优先级高于**KERNEL_TASK_TYPE_DEFAULT**，同时设置了全局kernel type和某一个tiling key的kernel type，该tiling key的kernel type以**KERNEL_TASK_TYPE**设置的为准。
- 没有设置全局默认kernel type的情况下，如果开发者只为其中的某几个tiling key设置kernel type，即部分tiling key没有设置kernel type，会导致算子kernel编译报错。
- 当设置具体的kernel task type时，用户的算子实现需要与kernel type相匹配。比如用户设置kernel type为KERNEL_TYPE_MIX_AIC_1_2，则算子内部实现应与核配比AIC:AIV为1:2相对应；若用户设置kernel type为KERNEL_TYPE_AIC_ONLY， 则算子内部实现应该为纯cube逻辑，不应该存在vector部分的逻辑。其他的kernel type类似。
- 当纯cube或者纯vec算子强制设定kernel type为MIX类型时，workspace的大小不能设置为0，需要设置一个大于0的值（比如16、32等）。
- 使用Tiling模板编程时，需要通过ASCENDC_TPL_KERNEL_TYPE_SEL设置Kernel类型即可，无需再通过该接口进行设置，本接口不生效。

#### 调用示例

- 示例一：使能VectorCore样例

  1. 完成算子kernel侧开发时，需要通过本接口使能Vector Core，算子执行时会同时启动AI Core和Vector Core， 此时AI Core会当成Vector Core使用。示例如下：

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
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

  2. 完成算子host侧Tiling开发时，设置的block_dim代表的是AI Core和Vector Core的总数，比如用户在host侧设置blockDim为10，则会启动总数为10的AI Core和Vector Core；为保证启动Vector Core，设置数值应大于AI Core的核数。您可以通过GetCoreNumAic接口获取AI Core的核数，GetCoreNumVector接口获取Vector Core的核数。 如下代码片段，展示了block_dim的设置方法，此处设置为AI Core和Vector Core的总和，表示所有AI Core和Vector Core都启动。

```
// 配套的host侧tiling函数示例：
ge::graphStatus TilingFunc(gert::TilingContext* context)
{	
    // 使能VectorCore，将block_dim置为AI Core中vector核数 + Vector Core中的vector核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // ASCENDXXX请替换为实际的版本型号
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCENDXXX) {
       totalCoreNum = totalCoreNum + ascendcPlatform.GetCoreNumVector();
    }
    context->SetBlockDim(totalCoreNum);
}
```

- 示例二：设置某一个具体的tiling key对应的kernel type。如下代码为伪代码 ，不可直接运行。

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    KernelAdd op;
    op.Init(x, y, z, tilingData.blockDim, tilingData.totalLength, tilingData.tileNum);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 设置默认的kernel类型为纯AIV类型
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0); // 设置tiling key=1对应的kernel类型为MIX AIV 1:0
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_AIV_ONLY); // 设置tiling key=2对应的kernel类型为纯AIV类型
        op.Process2();
    }
    // ...
}
// 配套的host侧tiling函数示例：
ge::graphStatus TilingFunc(gert::TilingContext* context)
{	
    // ...
    if (context->GetInputShape(0) > 10) {
        context->SetTilingKey(1);
    } else if (some condition) {
        context->SetTilingKey(2);
    }
}
```
