# GetReduceMaxMaxMinTmpSize

**页面ID:** atlasascendc_api_07_10147  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10147.html

---

#### 功能说明

kernel侧ReduceMax接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。该接口**最大临时空间当前等于最小临时空间**。

#### 函数原型

```
void GetReduceMaxMaxMinTmpSize(const ge::Shape& srcShape, const ge::DataType dataType, ReducePattern pattern, bool isSrcInnerPad, bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| srcShape | 输入 | 输入的shape信息，参数取值与ReduceMax接口的srcShape参数保持一致。 |
| dataType | 输入 | 输入的数据类型，ge::DataType类型，该类型的具体定义请参考DataType，当前支持的数据类型与ReduceMax接口的模板参数T保持一致。 |
| 用于指定ReduceMax的计算轴。ReducePattern类型，该类型的定义如下，包括Reduce轴和Normal轴。pattern由与输入向量维度数量相同的A、R字母组合形成，字母A表示Normal轴，R表示Reduce轴。该参数的取值与ReduceMax接口的pattern参数保持一致，当前只支持取值为AscendC::ReducePattern::AR，AscendC::ReducePattern::RA。                                                                                                                           ``` enum class ReducePattern : uint32_t {     AR = 0,     RA = 1,     R,     ARA,     ARAR,     ARARA,     ARARAR,     ARARARA,     ARARARAR,     ARARARARA,     RAR,     RARA,     RARAR,     RARARA,     RARARAR,     RARARARA, }; ``` |  |  |
| isSrcInnerPad | 输入 | 表示实际需要计算的最内层轴数据是否32Bytes对齐，参数取值与ReduceMax接口的isSrcInnerPad参数保持一致。 |
| isReuseSource | 输入 | 是否复用源操作数输入的空间，参数取值与ReduceMax接口的isReuseSource参数保持一致。 |
| maxValue | 输出 | ReduceMax接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。          > **注意:**             说明：                        maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | ReduceMax接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

#### 约束说明

无

#### 调用示例

完整的调用样例请参考更多样例。

```
// 输入shape为16*32的矩阵;算子输入的数据类型为float;不允许修改源操作数
auto shape = ge::Shape({ 16, 32 });
uint32_t maxValue = 0;
uint32_t minValue = 0;
bool isSrcInnerPad = true;
bool isReuseSource = false;
AscendC::GetReduceMaxMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, isSrcInnerPad, isReuseSource, maxValue, minValue);
```
