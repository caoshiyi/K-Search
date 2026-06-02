# GetAscendDequantMaxMinTmpSize

**页面ID:** atlasascendc_api_07_0821  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0821.html

---

#### 功能说明

kernel侧AscendDequant接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

#### 函数原型

```
void GetAscendDequantMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | 输入srcTensor的shape信息。 |
| typeSize | 输入 | 输入srcTensor的数据类型大小，单位为字节。比如输入的数据类型为int32_t，此处应传入4。 |
| maxValue | 输出 | AscendDequant接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | AscendDequant接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

#### 约束说明

无

#### 调用示例

```
// 输入shape信息为(2,1024)
// AscendDequant接口中传入的dequantParams中m = 2, n = 1024;算子输入的数据类型为int32_t
std::vector<int64_t> shape_vec = {2, 1024};
ge::Shape srcShape(shape_vec);
uint32_t typeSize = 4;
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetAscendDequantMaxMinTmpSize(srcShape, typeSize, maxValue, minValue);
```
