# OutputTensorDesc

**页面ID:** atlasopapi_07_00615  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00615.html

---

#### 函数功能

设置算子输出的Tensor描述信息，用于构造InferShapeContext的基类ExtendedKernelContext中的ComputeNodeInfo等信息。

#### 函数原型

```
OpInferShapeContextBuilder &OutputTensorDesc(size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format, const gert::ExpandDimsType &expand_dims_type = {})
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 输出的索引，对应的是算子输出实例索引。 |
| dtype | 输入 | 输出Tensor的数据类型。 |
| origin_format | 输入 | 输出Tensor的原始格式。 |
| storage_format | 输入 | 输出Tensor的存储格式。 |
| expand_dims_type | 输入 | 输出Tensor的补维规则ExpandDimsType。 |

#### 返回值说明

OpInferShapeContextBuilder对象本身，用于链式调用。

#### 约束说明

在调用Build方法之前，必须调用该接口，否则构造出的InferShapeContext将包含未定义数据。
