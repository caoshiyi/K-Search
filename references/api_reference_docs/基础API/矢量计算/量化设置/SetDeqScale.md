# SetDeqScale

**页面ID:** atlasascendc_api_07_0099  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0099.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置DEQSCALE寄存器的值。

#### 函数原型

- 用于AddDeqRelu/Cast/CastDeq的s322f16场景

```
__aicore__ inline void SetDeqScale(half scale)
```

- 用于CastDeq（isVecDeq=false）的场景

```
__aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)
```

- 用于CastDeq（isVecDeq=true）的场景

```
template <typename T>
__aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 输入量化Tensor的数据类型。支持的数据类型为uint64_t。 |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| scale（half） | 输入 | scale量化参数，half类型。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，用于AddDeqRelu/Cast/CastDeq的s322f16场景。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，用于AddDeqRelu/Cast/CastDeq的s322f16场景。 Atlas 推理系列产品AI Core：用于AddDeqRelu或者Cast的s322f16场景。 |
| scale（float） | 输入 | scale量化参数，float类型。 用于CastDeq（isVecDeq=false）场景设置DEQSCALE寄存器的值。 |
| offset | 输入 | offset量化参数，int16_t类型，只有前9位有效。 用于CastDeq（isVecDeq=false）的场景，设置offset。 |
| signMode | 输入 | bool类型，表示量化结果是否带符号。 用于CastDeq（isVecDeq=false）的场景，设置signMode。 |
| vdeq | 输入 | 用于CastDeq（isVecDeq=true）的场景，输入量化tensor，大小为128Byte。 类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。 LocalTensor的起始地址需要32字节对齐。 |
| 存储量化tensor信息的数据结构，结构体内包含量化tensor中的16组量化参数 ``` const uint8_t VDEQ_TENSOR_SIZE = 16;  struct VdeqInfo {     __aicore__ VdeqInfo() {}     __aicore__ VdeqInfo(const float vdeqScaleIn[VDEQ_TENSOR_SIZE], const int16_t vdeqOffsetIn[VDEQ_TENSOR_SIZE],         const bool vdeqSignModeIn[VDEQ_TENSOR_SIZE])     {         for (int32_t i = 0; i < VDEQ_TENSOR_SIZE; ++i) {             vdeqScale[i] = vdeqScaleIn[i];             vdeqOffset[i] = vdeqOffsetIn[i];             vdeqSignMode[i] = vdeqSignModeIn[i];         }     }      float vdeqScale[VDEQ_TENSOR_SIZE] = { 0 };     int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = { 0 };     bool vdeqSignMode[VDEQ_TENSOR_SIZE] = { 0 }; }; ```  - vdeqScale：float类型的数组，用于存储量化tensor中的scale参数scale0-scale15。- vdeqOffset：int16_t类型的数组，用于存储量化tensor中的offset参数offset0-offset15。- vdeqSignMode：bool类型的数组，用于存储量化tensor中的signMode参数signMode0-signMode15。 |  |  |

#### 约束说明

无

#### 调用示例

```
AscendC::LocalTensor<int8_t> dstLocal;
AscendC::LocalTensor<int16_t> srcLocal;
AscendC::LocalTensor<uint64_t> tmpBuffer;
uint32_t srcSize = 256;
// Cast
AscendC::LocalTensor<half> castDstLocal;
AscendC::LocalTensor<int32_t> castSrcLocal;
half scale = 1.0;
AscendC::SetDeqScale(scale);
AscendC::Cast(castDstLocal, castSrcLocal, AscendC::RoundMode::CAST_NONE, srcSize);
// CastDeq
float scale = 1.0;
int16_t offset = 0;
bool signMode = true;
AscendC::SetDeqScale(scale, offset, signMode);
AscendC::CastDeq<int8_t, int16_t, false, false>(dstLocal, srcLocal, srcSize);
// CastVdeq
float vdeqScale[16] = { 0 };
int16_t vdeqOffset[16] = { 0 };
bool vdeqSignMode[16] = { 0 };
for (int i = 0; i < 16; i++) {
    vdeqScale[i] = 1.0;
    vdeqOffset[i] = 0;
    vdeqSignMode[i] = true;
}
AscendC::VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
AscendC::SetDeqScale<uint64_t>(tmpBuffer, vdeqInfo);
AscendC::CastDeq<int8_t, int16_t, true, false>(dstLocal, srcLocal, srcSize);
```
