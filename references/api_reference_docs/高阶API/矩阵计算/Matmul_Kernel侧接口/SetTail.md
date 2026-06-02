# SetTail

**页面ID:** atlasascendc_api_07_0647  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0647.html

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

在不改变Tiling的情况下，重新设置本次计算的singleCoreM/singleCoreN/singleCoreK，以元素为单位。

#### 函数原型

```
__aicore__ inline void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tailM | 输入 | 重新设置的singleCoreM值 |
| tailN | 输入 | 重新设置的singleCoreN值 |
| tailK | 输入 | 重新设置的singleCoreK值 |

#### 约束说明

无

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.SetTail(tailM,tailN,tailK);    // 如果是尾核，需要调整singleCoreM/singleCoreN/singleCoreK
mm.IterateAll(gm_c);
```
