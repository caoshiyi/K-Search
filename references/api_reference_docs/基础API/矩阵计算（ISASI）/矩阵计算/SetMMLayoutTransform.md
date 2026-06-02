# SetMMLayoutTransform

**页面ID:** atlasascendc_api_07_0260  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0260.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置Mmad计算时优先通过M/N中的哪个方向。

#### 函数原型

```
__aicore__ inline void SetMMLayoutTransform(bool mmLayoutMode)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| mmLayoutMode | 输入 | 控制Mmad 优先通过M/N的哪个方向，bool型，支持如下两种取值：- true：代表CUBE将首先通过N方向，然后通过M方向产生结果。- false：代表CUBE将首先通过M方向，然后通过N方向生成结果。 |

#### 约束说明

无

#### 调用示例

```
bool mmLayoutMode = true;
AscendC::SetMMLayoutTransform(mmLayoutMode);
```
