# ShapeInfo

**页面ID:** atlasascendc_api_07_0008  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0008.html

---

#### 功能说明

ShapeInfo用来存放LocalTensor或GlobalTensor的shape信息。

#### 函数原型

- ShapeInfo结构定义

```
struct ShapeInfo {
public:
    __aicore__ inline ShapeInfo();
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[],
        const uint8_t inputOriginalShapeDim, const uint32_t inputOriginalShape[], const DataFormat inputFormat);
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[], const DataFormat inputFormat);
    __aicore__ inline ShapeInfo(const uint8_t inputShapeDim, const uint32_t inputShape[]);
    uint8_t shapeDim;
    uint8_t originalShapeDim;
    uint32_t shape[K_MAX_DIM];
    uint32_t originalShape[K_MAX_DIM];
    DataFormat dataFormat;
};
```

- 获取Shape中所有dim的累乘结果

```
__aicore__ inline int GetShapeSize(const ShapeInfo& shapeInfo)
```

#### 函数说明

**表1 **ShapeInfo结构参数说明

| 参数名称 | 描述 |
| --- | --- |
| shapeDim | 现有的shape维度。 |
| shape | 现有的shape。 |
| originalShapeDim | 原始的shape维度。 |
| originalShape | 原始的shape。 |
| 数据排布格式，DataFormat类型，定义如下： ``` enum class DataFormat : uint8_t {     ND = 0,     NZ,     NCHW,     NC1HWC0,     NHWC, }; ``` |  |

**表2 **GetShapeSize参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shapeInfo | 输入 | ShapeInfo类型，LocalTensor或GlobalTensor的shape信息。 |
