# GetInputShapeRange

**页面ID:** atlasopapi_07_00104  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00104.html

---

#### 函数功能

根据算子输入索引获取对应的输入shape range指针。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
const Range<Shape> *GetInputShapeRange(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

输入shape range指针，index非法时，返回空指针。

#### 约束说明

无。

#### 调用示例

```
const auto infer_shape_range_func =  -> graphStatus {
  auto input_shape_range = context->GetInputShapeRange(0U);
  auto output_shape_range = context->GetOutputShapeRange(0U);
  output_shape_range->SetMin(const_cast<gert::Shape *>(input_shape_range->GetMin()));
  output_shape_range->SetMax(const_cast<gert::Shape *>(input_shape_range->GetMax()));
  return GRAPH_SUCCESS;
};
```
