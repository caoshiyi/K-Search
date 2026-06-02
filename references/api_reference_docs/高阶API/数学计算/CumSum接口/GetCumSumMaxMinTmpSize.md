# GetCumSumMaxMinTmpSize

**页面ID:** atlasascendc_api_07_0606  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0606.html

---

#### 功能说明

kernel侧CumSum接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大和最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

#### 函数原型

```
void GetCumSumMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isLastAxis, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | 输入的shape信息。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间。 |
| isLastAxis | 输入 | 定义是first轴处理还是last轴处理。 |
| maxValue | 输出 | Cumsum接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Cumsum接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

#### 约束说明

- 输入input只支持二维结构。

#### 调用示例

```
// 输入shape为32*32的矩阵；算子输入的数据类型为half；isLastAxis传入默认值true，isReuseSource传入默认值false
uint32_t firstDim = 32;
uint32_t lastDim = 32;
std::vector<int64_t> srcShapeDims = {firstDim, lastDim};
auto srcShape = ge::Shape(srcShapeDims);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetCumSumMaxMinTmpSize(srcShape, 2, true, false, maxValue, minValue);
```
