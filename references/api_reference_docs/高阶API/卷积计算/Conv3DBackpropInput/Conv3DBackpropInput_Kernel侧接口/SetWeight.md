# SetWeight

**页面ID:** atlasascendc_api_07_0923  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0923.html

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

设置卷积反向计算的输入矩阵Weight。

#### 函数原型

```
__aicore__ inline void SetWeight(const AscendC::GlobalTensor<SrcT> &weight)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| weight | 输入 | Weight矩阵在Global Memory上的首地址。类型为GlobalTensor。SrcT表示Weight矩阵的数据类型，当前支持的数据类型为：half、bfloat16_t。 |

#### 约束说明

无

#### 调用示例

```
const Conv3DBackpropInputTilingData* tilingData;
// ...初始化tilingData
ConvBackpropApi::Conv3DBackpropInput<weightDxType, inputSizeDxType, gradOutputDxType, gradInputDxType> gradInput_;
// ...设置其它参数
gradInput_.SetWeight(weightGm_[offsetB_]);
```
