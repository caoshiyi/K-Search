# SetWeight

**页面ID:** atlasascendc_api_07_10073  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10073.html

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

设置权重矩阵Weight。

#### 函数原型

```
__aicore__ inline void SetWeight(const AscendC::GlobalTensor<WeightT>& weight)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| weight | 输入 | Weight在Global Memory上的地址。类型为GlobalTensor。权重矩阵Weight支持的数据类型为：half、bfloat16_t。 |

#### 约束说明

无

#### 调用示例

```
GlobalTensor<half> weightGm;
weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight));
conv3dApi.SetWeight(weightGm);
```
