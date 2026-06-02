# SetOutputDataType

**页面ID:** atlasopapi_07_00093  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00093.html

---

#### 函数功能

根据输出索引，设置指定输出的数据类型。

#### 函数原型

```
ge::graphStatus SetOutputDataType(const size_t index, const ge::DataType datatype)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子IR原型定义中的输出索引，从0开始计数。 |
| datatype | 输入 | 需要设置的输出数据类型。 关于DataType的说明，请参见DataType。 |

#### 返回值说明

返回设置的结果状态，状态说明请参见ge::graphStatus。

index非法时，返回失败。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferDataTypeForXXX(InferDataTypeContext *context) {
  auto ret = context->SetOutputDataType(0, ge::DataType::DT_FLOAT);
  ...
}
```
