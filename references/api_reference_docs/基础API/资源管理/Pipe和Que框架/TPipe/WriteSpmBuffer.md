# WriteSpmBuffer

**页面ID:** atlasascendc_api_07_0167  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0167.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

将需要溢出暂存的数据拷贝到SPM Buffer中。

#### 函数原型

- 适用于连续和不连续的数据暂存：

```
template <typename T>
__aicore__ inline void WriteSpmBuffer(const LocalTensor<T>& writeBuffer, const DataCopyParams& copyParams, int32_t writeOffset = 0)
```

- 适用于连续的数据暂存：

```
template <typename T>
__aicore__ inline void WriteSpmBuffer(const LocalTensor<T>& writeBuffer, const int32_t writeSize, int32_t writeOffset = 0)
```

#### 参数说明

**表1 **接口参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| writeBuffer | 输入 | 需要溢出暂存的Local内存。 |
| copyParams | 输入 | 搬运参数，DataCopyParams类型，DataCopyParams结构定义请参考表2。 |
| writeSize | 输入 | 拷贝的元素个数。 |
| writeOffset | 输入 | 拷贝到SPM Buffer的偏移，单位为字节。 |

**表2 **DataCopyParams结构体参数定义

| 参数名称 | 含义 |
| --- | --- |
| blockCount | 待搬运的连续传输数据块个数。uint16_t类型，取值范围：blockCount∈[1, 4095]。 |
| blockLen | 待搬运的每个连续传输数据块长度，单位为DataBlock（32字节）。uint16_t类型，取值范围：blockLen∈[1, 65535]。 特别地，当dst位于C2PIPE2GM时，单位为128B；当dst位于C2时，表示源操作数的连续传输数据块长度，单位为64B。 |
| srcGap | 源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，srcGap不要超出该数据类型的取值范围。 在L1 Buffer -> Fixpipe Buffer场景中，srcGap特指源操作数相邻连续数据块的间隔（前面一个数据块的头与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，srcGap不要超出该数据类型的取值范围。 |
| dstGap | 目的操作数相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，dstGap不要超出该数据类型的取值范围。 特别地，当dstLocal位于C2PIPE2GM时，单位为128B；当dstLocal位于C2时，单位为64B。 在L1 Buffer -> Fixpipe Buffer场景中，dstGap特指源操作数相邻连续数据块的间隔（前面一个数据块的头与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，dstGap不要超出该数据类型的取值范围。 |

#### 约束说明

- 暂存拷贝到L1时注意writeSize和writeOffset保证32字节对齐
- 拷贝的内存不要超出初始化的SPM Buffer大小，否则会存在溢出踩踏等问题。

#### 调用示例

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
int dataSize = 32; // 假设T为half类型，从ub上申请一块内存32 * sizeof(half)字节
int offset = 32; // 拷贝到spmBuffer时偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize * sizeof(half));
AscendC::LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
AscendC::DataCopyParams copyParams{1, 2, 0, 0}; // 从ub上搬运一个连续传输数据块，一个数据块的长度为2个datablock，一个datablock为32bytes
pipe.WriteSpmBuffer(writeLocal, copyParams, offset);
```

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
int dataSize = 32; // 假设T为half类型，从ub上申请一块内存32 * sizeof(half)字节
int offset = 32; // 拷贝到spmBuffer时偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize * sizeof(half));
AscendC::LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
pipe.WriteSpmBuffer(writeLocal, dataSize, offset);
```
