# SoftmaxGrad Tiling接口

**页面ID:** atlasascendc_api_07_0764  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0764.html

---

#### 功能说明

用于获取SoftmaxGrad Tiling参数。

#### 函数原型

- 获取Kernel接口计算所需最小/最大临时空间的接口

```
uint32_t GetSoftMaxGradMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isFront, const bool isReuseSource)
```

```
uint32_t GetSoftMaxGradMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isFront, const bool isReuseSource)
```

- Tiling计算接口

  - AscendC::optiling命名空间下的计算接口

```
void SoftMaxGradTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxGradTiling, const bool isFront = false)
```

  - AscendC命名空间下的计算接口

```
void SoftMaxGradTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxGradTiling, const bool isFront = false)
```

#### 参数说明

**表1 **SoftmaxGrad GetSoftMaxGradMaxTmpSize/GetSoftMaxGradMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize | 输入 | 计算的数据类型，比如half=2。 |
| isFront | 输入 | 是否只计算![](images/atlasascendc_api_07_0764_img_001.png)，和kernel侧的SoftmaxGrad接口一致，默认false。 |
| isReuseSource | 输入 | 与kernel侧接口配置保持一致。 |

**表2 **SoftmaxGrad SoftMaxGradTilingFunc接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| localWorkSpaceSize | 输入 | 剩余的可供SoftmaxGrad接口计算的临时空间大小，单位为Byte。localWorkSpaceSize的取值必须大于GetSoftMaxGradMinTmpSize接口返回的计算所需的最小临时空间大小。 |
| dataTypeSize | 输入 | 计算的数据类型，比如half=2。 |
| isFront | 输入 | 是否只计算![](images/atlasascendc_api_07_0764_img_002.png)，和kernel侧的SoftmaxGrad接口一致，默认false。 |
| softmaxGradTiling | 输出 | 输出SoftmaxGrad接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。 |

#### 返回值说明

GetSoftMaxGradMinTmpSize返回SoftmaxGrad接口能完成计算所需最小临时空间大小，单位为Byte。

GetSoftMaxGradMaxTmpSize返回SoftmaxGrad接口能完成计算所需最大临时空间大小，单位为Byte。

#### 约束说明

无
