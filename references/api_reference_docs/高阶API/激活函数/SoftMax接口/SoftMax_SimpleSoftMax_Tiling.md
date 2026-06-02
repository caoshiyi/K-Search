# SoftMax/SimpleSoftMax Tiling

**页面ID:** atlasascendc_api_07_0762  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0762.html

---

#### 功能说明

用于获取SoftMax/SimpleSoftMax Tiling参数。

#### 函数原型

- 获取Kernel接口计算所需最大/最小临时空间的接口

```
uint32_t GetSoftMaxMaxTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
```

```
uint32_t GetSoftMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t dataTypeSize, const bool isReuseSource)
```

- Tiling计算接口

  - AscendC::optiling命名空间下的计算接口

```
void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxTiling)
```

  - AscendC命名空间下的计算接口

```
void SoftMaxTilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize, const uint32_t localWorkSpaceSize, AscendC::tiling::SoftMaxTiling& softmaxTiling)
```

#### 参数说明

**表1 **SoftMax/SimpleSoftMax GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize | 输入 | 参与计算的max和sum的数据类型，比如half=2。 |
| isReuseSource | 输入 | 与kernel侧接口配置保持一致。 |

**表2 **SoftMax/SimpleSoftMax SoftMaxTilingFunc接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| dataTypeSize | 输入 | 参与计算的max和sum的数据类型，比如half=2。 |
| localWorkSpaceSize | 输入 | 剩余的可供SoftMax接口计算的空间大小，单位为Byte。localWorkSpaceSize的取值必须大于GetSoftMaxMinTmpSize接口返回的计算所需的最小临时空间大小。 |
| softmaxTiling | 输出 | 输出SoftMax接口所需的tiling信息，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。 |

#### 返回值说明

GetSoftMaxMaxTmpSize返回SoftMax/SimpleSoftMax接口能完成计算所需最大临时空间大小，单位为Byte。

GetSoftMaxMinTmpSize返回SoftMax/SimpleSoftMax接口能完成计算所需最小临时空间大小，单位为Byte。

#### 约束说明

无
