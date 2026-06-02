# GetOutputShape

**页面ID:** atlasopapi_07_00101  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00101.html

---

#### 函数功能

根据算子输出索引获取对应的输出Shape指针。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
Shape *GetOutputShape(const size_t index)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出索引，从0开始计数。 |

#### 返回值说明

返回指定的输出Shape指针，输入index非法时，返回空指针。

关于Shape类型的定义，请参见Shape。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferShapeForReshape(InferShapeContext *context) {
  const gert::Shape *x_shape = context->GetInputShape(0);        // 获取第0个输入的shape
  const gert::Tensor *shape_tensor = context->GetInputTensor(1); // 获取第1个输入的tensor  数据依赖
  gert::Shape *output_shape = context->GetOutputShape(0);
  if (x_shape == nullptr || shape_tensor == nullptr || output_shape == nullptr) {
    // 防御式编程，不应该出现的场景，打印错误并返回失败
    return ge::GRAPH_FAILED;
  }
  // ...
}
```
