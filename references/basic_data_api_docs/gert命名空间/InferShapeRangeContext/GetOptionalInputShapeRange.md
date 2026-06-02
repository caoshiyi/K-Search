# GetOptionalInputShapeRange

**页面ID:** atlasopapi_07_00108  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00108.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的可选输入shape range指针。

#### 函数原型

```
const Range<Shape> *GetOptionalInputShapeRange(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输入索引，从0开始计数。 |

#### 返回值说明

返回shape range指针，ir_index非法，或该INPUT没有实例化时，返回空指针。

#### 约束说明

无。

#### 调用示例

```
const auto infer_shape_range_func =  -> graphStatus {
  auto input_shape_range = context->GetOptionalInputShapeRange(0U);
  auto output_shape_range = context->GetOutputShapeRange(0U);
  output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
  output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
  return GRAPH_SUCCESS;
};
```
