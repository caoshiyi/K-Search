# OutputTensorDesc

**页面ID:** atlasopapi_07_00610  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00610.html

---

#### 函数功能

设置输出Tensor描述信息，用于构造InferDataTypeContext的基类ExtendedKernelContext中的ComputeNodeInfo信息。无需设置输出数据类型信息，输出数据类型由算子根据输入数据类型计算推导得到。

#### 函数原型

```
OpInferDataTypeContextBuilder &OutputTensorDesc(size_t index, ge::Format origin_format, ge::Format storage_format, const gert::ExpandDimsType &expand_dims_type = {})
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出实例索引。 |
| origin_format | 输入 | 输出Tensor的原始格式。 |
| storage_format | 输入 | 输出Tensor的存储格式。 |
| expand_dims_type | 输入 | 输出Tensor的补维规则ExpandDimsType，默认值为{}。 |

#### 返回值说明

OpInferDataTypeContextBuilder对象本身，用于链式调用。

#### 约束说明

在调用Build方法之前，必须调用该接口，否则构造出的InferDataTypeContext将包含未定义数据。
