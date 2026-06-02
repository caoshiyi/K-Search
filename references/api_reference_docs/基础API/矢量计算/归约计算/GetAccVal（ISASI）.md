# GetAccVal(ISASI)

**页面ID:** atlasascendc_api_07_0225  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0225.html

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

获取ReduceSum接口（Tensor前n个数据计算接口，n为接口的count参数）的计算结果。

#### 函数原型

```
template <typename T>
__aicore__ inline T GetAccVal()
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | ReduceSum指令的数据类型，支持half、float。 |

#### 返回值说明

ReduceSum接口（Tensor前n个数据计算接口，n为接口的count参数）的计算结果。

#### 约束说明

无。

#### 调用示例

```
AscendC::LocalTensor<float> src;
AscendC::LocalTensor<float> work;
AscendC::LocalTensor<float> dst;
AscendC::ReduceSum(dst, src, work, 128);
float res = AscendC::GetAccVal<float>();
```
