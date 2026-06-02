# SoftmaxFlashV3 Tiling接口

**页面ID:** atlasascendc_api_07_10002  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10002.html

---

#### 功能说明

用于获取SoftmaxFlashV3接口所需的Tiling参数。

#### 函数原型

- 获取Kernel接口计算所需最小/最大临时空间的接口

```
void GetSoftMaxFlashV3MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2, uint32_t& maxValue, uint32_t& minValue, const bool isUpdate, const bool isBasicBlock = false)
```

- Tiling计算接口

  - AscendC::optiling命名空间下的计算接口

```
void SoftMaxFlashV3TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashV3Tiling, const bool isUpdate,const bool isBasicBlock = false)
```

  - AscendC命名空间下的计算接口

```
void SoftMaxFlashV3TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashV3Tiling, const bool isUpdate,const bool isBasicBlock = false)
```

#### 参数说明

**表1 **GetSoftMaxFlashV3MaxMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize1 | 输入 | 输入srcTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数T的数据类型大小。当前模板参数T仅支持half类型，故此参数只支持取值为2。 |
| dataTypeSize2 | 输入 | 输入inMeanTensor、inExpSumTensor、inMaxTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数U的数据类型大小。当前模板参数U仅支持float类型，故此参数只支持取值为4。 |
| maxValue | 输出 | SoftMaxFlashV3接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | SoftMaxFlashV3接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |
| isUpdate | 输入 | 是否使能SoftMaxFlashV3 update为true的公式计算。该参数取值与SoftmaxFlashV3 Kernel接口的模板参数isUpdate保持一致。 |
| isBasicBlock | 输入 | 预留参数，暂未启用，必须使用默认值false。 |

**表2 **SoftMaxFlashV3TilingFunc接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize1 | 输入 | 输入srcTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数T的数据类型大小。当前模板参数T仅支持half类型，故此参数只支持取值为2。 |
| dataTypeSize2 | 输入 | 输入inMeanTensor、inExpSumTensor、inMaxTensor的数据类型大小，即对应SoftMaxFlashV3 Kernel函数中模板参数U的数据类型大小。当前模板参数U仅支持float类型，故此参数只支持取值为4。 |
| localWorkSpaceSize | 输入 | 剩余的可供SoftmaxFlashV3接口计算的空间大小。localWorkSpaceSize的取值必须大于GetSoftMaxFlashV3MaxMinTmpSize接口返回的计算所需的最小临时空间大小。 |
| isUpdate | 输入 | 是否使能SoftMaxFlashV3 update为true的公式计算。与SoftmaxFlashV3 Kernel接口的模板参数isUpdate保持一致。 |
| isBasicBlock | 输入 | 预留参数，暂未启用，必须使用默认值false。 |
| softmaxFlashV3Tiling | 输出 | 输出SoftMaxFlashV3接口所需的Tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。 |

#### 约束说明

无
