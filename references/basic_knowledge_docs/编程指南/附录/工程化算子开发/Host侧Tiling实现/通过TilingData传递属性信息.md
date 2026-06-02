# 通过TilingData传递属性信息

**页面ID:** atlas_ascendc_10_00022  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00022.html

---

如果算子包含属性信息，该属性信息可以通过TilingData传递到kernel侧，参与kernel侧算子核函数的计算。以ReduceMaxCustom算子为例，该算子用于对输入数据按维度dim返回最大值，并且返回索引。ReduceMaxCustom算子有两个属性，reduceDim和isKeepDim，reduceDim表示按照哪一个维度进行reduce操作；isKeepDim表示是否需要保持输出的维度与输入一样。本样例仅支持对最后一维做reduce操作，输入数据类型为half。

1. ReduceMaxCustom算子TilingData的定义如下：这里我们重点关注reduceAxisLen。参数reduceAxisLen表示获取reduceDim轴的长度，这里也就是最后一维的长度。该参数后续会通过TilingData传递到kernel侧参与计算。

```
#ifndef REDUCE_MAX_CUSTOM_TILING_H
#define REDUCE_MAX_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ReduceMaxTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, reduceAxisLen); // 添加tiling字段，reduceDim轴的长度
  //其他TilingData参数的定义
  ...
END_TILING_DATA_DEF;
// 注册算子tilingdata类到对应的ReduceMaxCustom算子
REGISTER_TILING_DATA_CLASS(ReduceMaxCustom, ReduceMaxTilingData)
}
#endif // REDUCE_MAX_CUSTOM_TILING_H
```

2. ReduceMaxCustom算子的Tiling实现如下。这里我们重点关注属性信息通过TilingData传递的过程：首先通过TilingContext上下文从attr获取reduceDim属性值；然后根据reduceDim属性值获取reduceDim轴的长度并设置到TilingData中。

```
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ReduceMaxTilingData tiling;
    // 从attr获取reduceDim属性值，因为reduceDim是第一个属性，所以GetAttrPointer传入的索引值为0
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const uint32_t* reduceDim = attrs->GetAttrPointer<uint32_t>(0);
    // 获取reduceDim轴的长度
    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    const gert::Shape& xShape = xShapePtr->GetStorageShape();
    const uint32_t reduceAxisLen = xShape.GetDim(*reduceDim);
    // 计算TilingData中除了reduceAxisLen之外其他成员变量的值
    ...
    // 将reduceAxisLen设置到tiling结构体中，传递到kernel函数使用
    tiling.set_reduceAxisLen(reduceAxisLen);
    // 设置TilingData中除了reduceAxisLen之外其他成员变量的值
    ...
    // TilingData序列化保存
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    ...
    return ge::GRAPH_SUCCESS;
}} // namespace optiling
```
