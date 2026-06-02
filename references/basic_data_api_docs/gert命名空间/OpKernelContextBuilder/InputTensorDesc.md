# InputTensorDesc

**页面ID:** atlasopapi_07_00627  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00627.html

---

#### 函数功能

设置Tensor描述信息，用于构造KernelContext的基类ExtendedKernelContext中的ComputeNodeInfo等信息。

#### 函数原型

```
OpKernelContextBuilder &InputTensorDesc(size_t index, ge::DataType dtype, ge::Format origin_format, ge::Format storage_format, const gert::ExpandDimsType &expand_dims_type = {})
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入实例索引。 |
| dtype | 输入 | 输入Tensor的数据类型。 |
| origin_format | 输入 | 输入Tensor的原始格式。 |
| storage_format | 输入 | 输入Tensor的存储格式。 |
| expand_dims_type | 输入 | 输入Tensor的补维规则ExpandDimsType，默认值为{}。 |

#### 返回值说明

OpKernelContextBuilder对象引用，用于链式调用。

#### 约束说明

在调用Build方法之前，必须调用该接口，否则构造出的KernelContext将包含未定义数据。
