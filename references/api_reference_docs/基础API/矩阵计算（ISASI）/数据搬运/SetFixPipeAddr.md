# SetFixPipeAddr

**页面ID:** atlasascendc_api_07_0256  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0256.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

DataCopy（CO1->GM）过程中进行随路量化后，通过调用该接口设置Elementwise操作时LocalTensor的地址。

#### 函数原型

```
template <typename T>
__aicore__ inline void SetFixPipeAddr(const LocalTensor<T>& eleWiseData, uint16_t c0ChStride)
```

#### 参数说明

**表1 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| eleWiseData | 输入 | L1 Buffer上的源操作数。类型为LocalTensor。 支持的TPosition为A1/B1/C1。起始地址需要保证32字节对齐，仅支持half数据类型。 |
| c0ChStride | 输入 | 在L1 Buffer上的C0 channel stride，单位是C0_SIZE（32B）。 eleWiseData沿N方向以C0为单位切分得到的数据块称为C0 channel，两块C0 channel的间隔称之为C0 channel stride。 |

#### 约束说明

无

#### 调用示例

完整示例可参考完整示例。

```
__aicore__inline void SetEleSrcPara(const LocalTensor <half>& eleWiseData, uint16_t c0ChStride)
{
    AscendC::SetFixPipeAddr(eleWiseData, c0ChStride);
}
```
