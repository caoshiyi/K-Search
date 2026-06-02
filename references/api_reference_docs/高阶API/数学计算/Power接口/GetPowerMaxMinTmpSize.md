# GetPowerMaxMinTmpSize

**页面ID:** atlasascendc_api_07_0521  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0521.html

---

#### 功能说明

kernel侧Power接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大和最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

接口内部根据srcShape1、srcShape2输入判断接口为Power(dstTensor, srcTensor1, srcTensor2)、Power(dstTensor, srcTensor1, scalarValue) 或Power(dstTensor, scalarValue, srcTensor2)类型中的哪一种，进而返回对应临时空间大小。

#### 函数原型

```
void GetPowerMaxMinTmpSize(const ge::Shape& srcShape1, const ge::Shape& srcShape2, const bool typeIsInt, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape1 | 输入 | 输入srcTensor1的shape信息。 |
| srcShape2 | 输入 | 输入srcTensor2的shape信息。 |
| typeIsInt | 输入 | bool类型，true表示输入是int32_t。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与Power接口一致。 |
| maxValue | 输出 | Power接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Power接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

#### 约束说明

无

#### 调用示例

完整的调用样例请参考更多样例。

- Power(dstTensor, srcTensor1, srcTensor2)样例

```
// srcTensor1、srcTensor2输入shape信息均为512;算子输入的数据类型为float;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetPowerMaxMinTmpSize(shape, shape, false, 4, false, maxValue, minValue);
```

- Power(dstTensor, srcTensor1, scalarValue)样例

```
// srcTensor1输入shape信息为128*128，scalarValue的shape为1;算子输入的数据类型为half;不允许修改源操作数
std::vector<int64_t> shape1_vec = {128,128};
std::vector<int64_t> shape2_vec = {1};
ge::Shape shape1(shape1_vec);
ge::Shape shape2(shape2_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetPowerMaxMinTmpSize(shape1, shape2, false, 2, false, maxValue, minValue);
```

- Power(dstTensor, scalarValue, srcTensor2)样例

```
//scalarValue的shape为1，srcTensor2输入shape信息为128*128;算子输入的数据类型为float;不允许修改源操作数
std::vector<int64_t> shape1_vec = {1};
std::vector<int64_t> shape2_vec = {128,128};
ge::Shape shape1(shape1_vec);
ge::Shape shape2(shape2_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
AscendC::GetPowerMaxMinTmpSize(shape1, shape2, false, 4, false, maxValue, minValue);
```
