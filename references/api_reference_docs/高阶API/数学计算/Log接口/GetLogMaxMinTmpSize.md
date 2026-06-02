# GetLogMaxMinTmpSize

**页面ID:** atlasascendc_api_07_0513  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0513.html

---

#### 功能说明

Host侧接口，用于获取Log接口能完成计算所需最小的临时空间大小，此空间为预留空间，即需要保证预留有足够的物理空间，用于执行计算。

#### 函数原型

```
void GetLogMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetLog10MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

```
void GetLog2MaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数列表

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| srcShape | 输入 | 输入的shape信息。 |
| typeSize | 输入 | 输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，与Log接口一致。 |
| maxValue | 输出 | Log接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。 > **注意:**  说明： maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | Log接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。 |

#### 约束说明

无

#### 调用示例

完整的调用样例请参考更多样例。

- GetLogMaxMinTmpSize接口样例：

```
// 输入shape信息为1024;算子输入的数据类型为half;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
auto tmp_size = AscendC::GetLogMaxMinTmpSize(shape, 2, false, maxValue, minValue);
```

- GetLog10MaxMinTmpSize接口样例：

```
// 输入shape信息为1024;算子输入的数据类型为half;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
auto tmp_size = AscendC::GetLog10MaxMinTmpSize(shape, 2, false, maxValue, minValue);
```

- GetLog2MaxMinTmpSize接口样例：

```
// 输入shape信息为1024;算子输入的数据类型为half;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
auto tmp_size = AscendC::GetLog2MaxMinTmpSize(shape, 2, false, maxValue, minValue);
```
