# GetRequiredInputTensorRange

**页面ID:** atlasopapi_07_00111  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00111.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应的必选输入tensor range指针。

#### 函数原型

```
const TensorRange *GetRequiredInputTensorRange(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 必选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

TensorRange类型指针，定义如下：

```
using TensorRange = Range<Tensor>;
```

ir_index非法时，返回空指针。

#### 约束说明

如果输入没有被设置为数据依赖，调用此接口获取tensor range时，只能在tensor中获取到正确的shape、format、datatype信息，无法获取到真实的tensor数据地址（获取到的地址为nullptr）。

#### 调用示例

```
const auto infer_shape_range_func =  -> graphStatus {
  auto input_shape_range = context->GetRequiredInputTensorRange(0U);
  auto output_shape_range = context->GetOutputShapeRange(0U);
  *output_shape_range->GetMin() = input_shape_range->GetMin()->GetStorageShape();
  *output_shape_range->GetMax() = input_shape_range->GetMax()->GetStorageShape();
  return GRAPH_SUCCESS;
};
```
