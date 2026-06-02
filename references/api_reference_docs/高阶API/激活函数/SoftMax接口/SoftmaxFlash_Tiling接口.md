# SoftmaxFlash Tiling接口

**页面ID:** atlasascendc_api_07_0763  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0763.html

---

#### 功能说明

**注意：该接口后续即将废弃，新开发内容不要使用该接口**。

用于获取SoftmaxFlash Tiling参数。

#### 函数原型

- 获取Kernel接口计算所需最小/最大临时空间的接口

```
uint32_t GetSoftMaxFlashMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate, const bool isReuseSource)
```

```
uint32_t GetSoftMaxFlashMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isUpdate, const bool isReuseSource)
```

- Tiling计算接口

  - AscendC::optiling命名空间下的计算接口

```
void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false)
```

  - AscendC命名空间下的计算接口

```
void SoftMaxFlashTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxFlashTiling, const bool isUpdate = false)
```

#### 参数说明

**表1 **SoftmaxFlash GetSoftMaxFlashMaxTmpSize/GetSoftMaxFlashMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize | 输入 | 参与计算的maxTensor和sumTensor的数据类型，比如half=2。 |
| isUpdate | 输入 | 是否使能刷新功能，和kernel侧SoftmaxFlash接口一致，默认false。 |
| isReuseSource | 输入 | 与kernel侧接口配置保持一致。 |

**表2 **SoftmaxFlash SoftMaxFlashTilingFunc接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize | 输入 | 参与计算的maxTensor和sumTensor的数据类型，比如half=2。 |
| localWorkSpaceSize | 输入 | 剩余的可供SoftmaxFlash接口计算的空间大小，单位为Byte。localWorkSpaceSize的取值必须大于GetSoftMaxFlashMinTmpSize接口返回的计算所需的最小临时空间大小。 |
| isUpdate | 输入 | 是否使能刷新功能，和kernel侧SoftmaxFlash接口一致，默认false。 |
| softmaxFlashTiling | 输出 | 输出SoftmaxFlash接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。 |

#### 返回值说明

GetSoftMaxFlashMaxTmpSize返回SoftmaxFlash接口能完成计算所需最大临时空间大小，单位为Byte。

GetSoftMaxFlashMinTmpSize返回SoftmaxFlash接口能完成计算所需最小临时空间大小，单位为Byte。

#### 约束说明

无
