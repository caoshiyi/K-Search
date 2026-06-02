# TILING_KEY_IS

**页面ID:** atlasascendc_api_07_0217  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0217.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明

在核函数中判断本次执行时的tiling_key是否等于host侧运行时设置的某个key，从而标识tiling_key==key的一条kernel分支。

#### 函数原型

```
TILING_KEY_IS(key)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| key | 输入 | key表示某个核函数的分支，必须是非负整数。 |

#### 约束说明

- TILING_KEY_IS运用于if和else if分支，不支持else分支，即用TILING_KEY_IS函数来表征N个分支，必须用N个TILING_KEY_IS(key)来分别表示。
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
    op.Init(x, y, z, tilingData.blockDim, tilingData.totalLength, tilingData.tileNum);
    // 当TilingKey为1时，执行Process1；为2时，执行Process2；为3时，执行Process3
    if (TILING_KEY_IS(1)) {
        op.Process1();
    } else if (TILING_KEY_IS(2)) {
        op.Process2();
    } else if (TILING_KEY_IS(3)) {
        op.Process3();
    }
    // 其他代码逻辑
    ...
    // 此处示例当TilingKey为3时，会执行ProcessOther
    if (TILING_KEY_IS(3)) {
        op.ProcessOther();
    }
}
```

配套的host侧tiling函数示例（伪代码）：

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
    }
}
```
