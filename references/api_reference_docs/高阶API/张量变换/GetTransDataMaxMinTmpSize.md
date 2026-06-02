# GetTransDataMaxMinTmpSize

**页面ID:** atlasascendc_api_07_10182  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10182.html

---

#### 功能说明

kernel侧TransData接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

- 为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
- 在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。该接口**最大临时空间当前等于最小临时空间**。

#### 函数原型

```
bool GetTransDataMaxMinTmpSize(const platform_ascendc::PlatformAscendC& platform, const ge::Shape& srcShape, const ge::Shape& dstShape,const ge::DataType dataType, const TransDataConfig &config, uint32_t& maxValue, uint32_t& minValue)
```

#### 参数说明

**表1 **接口参数列表

| 接口 | 输入/输出 | 功能 |
| --- | --- | --- |
| platform | 输入 | 传入硬件平台的信息，PlatformAscendC定义请参见构造及析构函数。 |
| srcShape | 输入 | 输入源操作数的shape大小，参数取值与TransData接口的params.srcLayout参数中的shape信息保持一致。 |
| dstShape | 输入 | 输出目的操作数的shape大小，参数取值与TransData接口的params.dstLayout参数中的shape信息保持一致。 |
| dataType | 输入 | 输入的数据类型，ge::DataType类型，该类型的具体定义请参考DataType，当前只支持half/float/uint16_t/int16_t数据类型的输入。 |
| 数据格式转换的场景，参数取值与TransData接口的config参数保持一致。当前支持的转换场景有：NCDHW -> NDC1HWC0、NDC1HWC0 -> NCDHW、NCDHW -> FRACTAL_Z_3D、FRACTAL_Z_3D -> NCDHW。TransDataConfig类型，具体定义如下。                                                                                                                           ``` struct TransDataConfig {     DataFormat srcFormat;     DataFormat dstFormat; };  enum class DataFormat : uint8_t {     ND = 0,     NZ,     NCHW,     NC1HWC0,     NHWC,     NCDHW,     NDC1HWC0,     FRACTAL_Z_3D, }; ``` |  |  |
| maxValue | 输出 | TransData接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。          > **注意:**             说明：                        maxValue仅作为参考值，有可能大于Unified Buffer剩余空间的大小，该场景下，开发者需要根据Unified Buffer剩余空间的大小来选取合适的临时空间大小。 |
| minValue | 输出 | TransData接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。 |

#### 返回值说明

GetTransDataMaxMinTmpSize返回值为true/false，true表示成功获取TransData接口内部计算需要的最大和最小临时空间大小；false表示获取失败。

#### 约束说明

无

#### 调用示例

完整的调用样例请参考更多样例。

```
// 输入shape为(1,16,2,4,4)NCDHW转换为输出shape(1,2,1,4,4,16)NDC1HWC0;算子输入的数据类型为half
uint32_t maxSize;
uint32_t minSize;
int32_t n = 1, c = 16, d = 2, h = 4, w = 4, c1 = 1, c0 = 16;
auto ncdhwShape = ge::Shape({ n, c, d, h, w });
auto ndc1hwc0Shape = ge::Shape({ n, d, c1, h, w, c0});
auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
TransDataConfig config = {DataFormat::NCDHW, DataFormat::NDC1HWC0};
bool ret = GetTransDataMaxMinTmpSize(plat, ncdhwShape, ndc1hwc0Shape, ge::DataType::DT_FLOAT16, config, maxSize, minSize);
```
