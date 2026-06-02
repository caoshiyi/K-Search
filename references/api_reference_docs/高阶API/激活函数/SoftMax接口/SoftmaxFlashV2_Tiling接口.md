# SoftmaxFlashV2 Tiling接口

**页面ID:** atlasascendc_api_07_0765  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0765.html

---

#### 功能说明

用于获取SoftmaxFlashV2接口所需的Tiling参数。

#### 函数原型

- 获取Kernel接口计算所需最小/最大临时空间的接口

```
uint32_t GetSoftMaxFlashV2MinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
```

```
uint32_t GetSoftMaxFlashV2MaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
```

- Tiling计算接口

  - AscendC::optiling命名空间下的计算接口

```
void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
```

  - AscendC命名空间下的计算接口

```
void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate, const bool isBasicBlock = false, const bool isFlashOutputBrc = false)
```

#### 参数说明

**表1 **GetSoftMaxFlashV2MinTmpSize/GetSoftMaxFlashV2MaxTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize1 | 输入 | 计算的源数据的数据类型大小，比如half=2。 |
| dataTypeSize2 | 输入 | 参与计算的expSumTensor和maxTensor的数据类型大小，比如half=2。 |
| isUpdate | 输入 | 是否使能刷新功能，和kernel侧SoftmaxFlashV2接口一致。 |
| isBasicBlock | 输入 | 是否要使能基本块计算。isBasicBlock参数可以通过isBasicBlockInSoftmax接口获取，与kernel侧接口的模板参数保持一致，默认false。注意，若kernel侧API使能模板参数SoftmaxConfig，即shape常量化场景，isBasicBlock参数必须通过接口isBasicBlockInSoftmax获取。 |
| isFlashOutputBrc | 输入 | 是否使能输出shape的非拓展模式。非拓展模式为不对输出数据做Broadcast，输出shape为(m, 1)。参数取值如下： - false：不使能非拓展模式，默认值。输出为float数据类型时，shape为(m，8)；输出为half数据类型时，shape为(m, 16)。- true：使能非拓展模式，输出的shape均为(m, 1)。该参数取值为true时，kernel接口的模板参数SoftmaxConfig中的mode必须配置为SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC。 |

**表2 **SoftMaxFlashV2TilingFunc接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| localWorkSpaceSize | 输入 | 剩余的可供SoftmaxFlashV2接口计算的空间大小。localWorkSpaceSize的取值必须大于GetSoftMaxFlashV2MinTmpSize接口返回的计算所需的最小临时空间大小。 |
| dataTypeSize1 | 输入 | 计算的源数据的数据类型，比如half=2。 |
| dataTypeSize2 | 输入 | 参与计算的maxTensor和sumTensor的数据类型，比如half=2。 |
| isUpdate | 输入 | 是否使能刷新功能，和kernel侧SoftmaxFlashV2接口一致。 |
| isBasicBlock | 输入 | 是否要使能基本块计算。isBasicBlock参数可以通过isBasicBlockInSoftmax接口获取，与kernel侧接口的模板参数保持一致，默认false。注意，若kernel侧API使能模板参数SoftmaxConfig，即shape常量化场景，isBasicBlock参数必须通过接口isBasicBlockInSoftmax获取。 |
| isFlashOutputBrc | 输入 | 是否使能输出shape的非拓展模式。非拓展模式为不对输出数据做Broadcast，输出shape为(m, 1)。参数取值如下： - false：不使能非拓展模式，默认值。输出为float数据类型时，shape为(m，8)；输出为half数据类型时，shape为(m, 16)。- true：使能非拓展模式，输出的shape均为(m, 1)。该参数取值为true时，kernel接口的模板参数SoftmaxConfig中的mode必须配置为SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC。 |
| softmaxFlashTiling | 输出 | 输出SoftmaxFlashV2接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。 |

#### 返回值说明

GetSoftMaxFlashV2MinTmpSize返回SoftmaxFlashV2接口能完成计算所需最小临时空间大小，单位为Byte。

GetSoftMaxFlashV2MaxTmpSize返回SoftmaxFlashV2接口能完成计算所需最大临时空间大小，单位为Byte。

#### 约束说明

无
