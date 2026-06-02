# IsBasicBlockInSoftMax

**页面ID:** atlasascendc_api_07_0766  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0766.html

---

#### 功能说明

用于判断SoftMaxTiling结构是否符合基本块特征。

#### 函数原型

- AscendC::optiling命名空间下的计算接口

```
bool IsBasicBlockInSoftMax(optiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2)
```

- AscendC命名空间下的计算接口

```
bool IsBasicBlockInSoftMax(AscendC::tiling::SoftMaxTiling& tiling, const uint32_t dataTypeSize = 2)
```

#### 参数说明

**表1 **参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| tiling | 输入 | 待判断的SoftMaxTiling结构，支持optiling::SoftMaxTiling形式入参和AscendC::tiling::SoftMaxTiling形式入参。 |
| dataTypeSize | 输入 | 参与计算的srcTensor的数据类型大小，比如half=2。 |

#### 返回值说明

- 返回true表示SoftMaxTiling结构满足基本块Tiling特征。
- 返回false表示SoftMaxTiling结构不满足基本块Tiling特征。

#### 约束说明

无
