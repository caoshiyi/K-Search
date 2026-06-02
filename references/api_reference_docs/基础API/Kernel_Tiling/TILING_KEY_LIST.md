# TILING_KEY_LIST

**页面ID:** atlasascendc_api_07_00186  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00186.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

TILING_KEY_LIST函数用于在核函数中判断当前执行的TilingKey是否与Host侧配置的指定TilingKey匹配，从而标识满足TilingKey == key1或TilingKey == key2条件的分支逻辑。

#### 函数原型

```
TILING_KEY_LIST(key1,key2)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| key | 输入 | key表示某个核函数的分支，必须是非负整数。 |

#### 约束说明

- TILING_KEY_LIST运用于if和else if分支，不支持else分支，即用TILING_KEY_LIST函数来表征N个分支，必须用N个TILING_KEY_LIST(key1,key2)来分别表示。
- 支持传入两个TilingKey，每个TilingKey具备唯一性。
- 使用该接口时，必须设置默认的Kernel类型，也可以为某个TilingKey单独配置Kernel类型，该配置会覆盖默认Kernel类型。Kernel类型仅支持配置为KERNEL_TYPE_MIX_AIC_1_1、KERNEL_TYPE_MIX_AIC_1_2。
- 暂不支持Kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }
    KernelAdd op;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    op.Init(x, y, z, tilingData.blockDim, tilingData.totalLength, tilingData.tileNum);
    // 当TilingKey为1或2时，执行Process1；为3或4时，执行Process2
    if (TILING_KEY_LIST(1,2)) {
        op.Process1();
    } else if (TILING_KEY_LIST(3,4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        op.Process2();
    } 
    // 其他代码逻辑
    ...
    // 此处示例当TilingKey为3或4时，会执行ProcessOther
    if (TILING_KEY_LIST(3,4)) {
        op.ProcessOther();
    }
}
```

配套的Host侧Tiling函数示例（伪代码）：

```
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 其他代码逻辑
    ...
    if (context->GetInputShape(0) > 10) {
        context->SetTilingKey(1);
    } else if (some condition) {
        context->SetTilingKey(2);
    } else if (some condition) {
        context->SetTilingKey(3);
    } else if (some condition) {
        context->SetTilingKey(4);
    }
}
```
