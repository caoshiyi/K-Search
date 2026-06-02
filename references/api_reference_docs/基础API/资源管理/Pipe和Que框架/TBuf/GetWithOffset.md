# GetWithOffset

**页面ID:** atlasascendc_api_07_0164  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0164.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

以TBuf为基地址，向后偏移指定长度，将偏移后的地址作为起始地址，提取长度为指定值的Tensor。

#### 函数原型

```
template <typename T>
__aicore__ inline LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset)
```

#### 参数说明

**表1 **模板参数说明

| 参数名称 | 含义 |
| --- | --- |
| T | 待获取Tensor的数据类型。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| size | 输入 | 需要获取的Tensor元素个数。 |
| bufOffset | 输入 | 从起始位置的偏移长度，单位是字节，且需32字节对齐。 |

#### 约束说明

size的数值是Tensor中元素的个数，size*sizeof(T) + bufOffset不能超过TBuf初始化时的长度。

bufOffset需满足32字节对齐的要求。

#### 返回值说明

获取到的LocalTensor。

#### 调用示例

```
// 为TBuf初始化分配内存，分配内存长度为1024字节
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf; // 模板参数为TPosition中的VECCALC类型
uint32_t byteLen = 1024;
pipe.InitBuffer(calcBuf, byteLen);
// 从calcBuf偏移64字节获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
AscendC::LocalTensor<int32_t> tempTensor1 = calcBuf.GetWithOffset<int32_t>(128, 64);
```
