# SetInput

**页面ID:** atlasascendc_api_07_10072  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10072.html

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

设置特征矩阵Input。

#### 函数原型

```
__aicore__ inline void SetInput(const AscendC::GlobalTensor<InputT>& input)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| input | 输入 | Input在Global Memory上的首地址。类型为GlobalTensor。特征矩阵Input支持的数据类型为：half、bfloat16_t。 |

#### 约束说明

无

#### 调用示例

```
GlobalTensor<half> inputGm;
inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input));
conv3dApi.SetInput(inputGm);
```
