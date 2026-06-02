# GetSelectMaxMinTmpSize

**页面ID:** atlasascendc_api_07_0860  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0860.html

---

#### 功能说明

kernel侧Select接口的计算需要开发者申请临时空间，本接口用于在host侧获取申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，申请的临时空间大小不能小于最小临时空间大小；
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间申请。

#### 函数原型

> **注意:** 

GetSelectWithBytesMaskMinTmpSize、GetSelectWithBytesMaskMaxTmpSize、GetSelectWithBytesMaskMaxMinTmpSize接口废弃，并将在后续版本移除，请不要使用该接口。请使用GetSelectMinTmpSize、GetSelectMaxTmpSize、GetSelectMaxMinTmpSize接口。

- 获取最小临时空间大小

```
uint32_t GetSelectMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
```

```
uint32_t GetSelectWithBytesMaskMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
```

- 获取最大临时空间大小

```
uint32_t GetSelectMaxTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
```

```
uint32_t GetSelectWithBytesMaskMaxTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask)
```

- 获取最大和最小临时空间大小

```
void GetSelectMaxMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetSelectWithBytesMaskMaxMinTmpSize(const ge::Shape& src0Shape, const ge::Shape& src1Shape, const uint32_t srcTypeSize, const ge::Shape& maskShape, const uint32_t maskTypeSize, const bool isReuseMask, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| src0Shape | 输入 | 输入src0的shape信息。src0为scalar时，shape应为{1}。 |
| src1Shape | 输入 | 输入src1的shape信息。src1为scalar时，shape应为{1}。 |
| srcTypeSize | 输入 | 输入srcTensor的数据类型大小，比如数据类型为half，此处应传入2。 |
| maskShape | 输入 | 输入maskTensor的shape信息。 |
| maskTypeSize | 输入 | 输入maskTensor的数据类型大小，比如数据类型为bool，此处应传入1。 |
| isReuseMask | 输入 | 是否复用maskTensor输入的空间。与kernel侧保持一致。 |
| maxValue | 输出 | Select接口能完成计算所需最大临时空间大小。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Select接口能完成计算所需最小临时空间大小。 |

#### 返回值说明

GetSelectMinTmpSize返回Select接口能完成计算所需最小临时空间大小。

GetSelectMaxTmpSize返回Select接口能完成计算所需最大临时空间大小。

GetSelectMaxMinTmpSize无返回值。

#### 约束说明

无

#### 调用示例

```
std::vector<int64_t> shape0Vec = {64, 128};
std::vector<int64_t> shape1Vec = {1};
std::vector<int64_t> mask1Vec = {64, 128};
ge::Shape src0Shape(shape0Vec);
ge::Shape src1Shape(shape1Vec);
ge::Shape maskShape(mask1Vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetSelectMaxMinTmpSize(src0Shape, src1Shape, 2, maskShape, 1, false, maxValue, minValue);
```
