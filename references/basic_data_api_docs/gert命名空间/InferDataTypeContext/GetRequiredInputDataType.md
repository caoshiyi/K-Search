# GetRequiredInputDataType

**页面ID:** atlasopapi_07_00091  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00091.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应必选输入的数据类型。

#### 函数原型

```
ge::DataType GetRequiredInputDataType(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 必选输入在算子IR原型定义中的索引，从0开始计数。 |

#### 返回值说明

返回指定输入的数据类型，若输入的ir_index非法，返回DT_UNDEFINED。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferDataTypeForXXX(InferDataTypeContext *context) {
  auto data_type = context->GetRequiredInputDataType(1);
  ...
}
```
