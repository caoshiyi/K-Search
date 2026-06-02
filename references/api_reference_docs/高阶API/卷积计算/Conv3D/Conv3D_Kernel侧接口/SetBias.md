# SetBias

**页面ID:** atlasascendc_api_07_10074  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10074.html

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

设置偏置矩阵Bias。

#### 函数原型

```
__aicore__ inline void SetBias(const AscendC::GlobalTensor<BiasT>& bias)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| bias | 输入 | Bias在Global Memory上的地址。类型为GlobalTensor。偏置矩阵Bias支持的数据类型为：half、bfloat16_t。 |

#### 约束说明

在卷积计算中，如果涉及偏置矩阵Bias，必须调用此接口；若卷积计算不涉及Bias，则不应调用此接口。

#### 调用示例

```
GlobalTensor<float> biasGm;
biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bias));
if (biasFlag) {
    conv3dApi.SetBias(biasGm);
}
```
