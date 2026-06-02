# GetInputDataType

**页面ID:** atlasopapi_07_00088  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00088.html

---

#### 函数功能

根据算子输入索引获取对应输入的数据类型。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
ge::DataType GetInputDataType(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子的输入索引，从0开始计数。 |

#### 返回值说明

返回指定输入的数据类型。

若输入index非法，返回DT_UNDEFINED。

#### 约束说明

无。

#### 调用示例

```
ge::graphStatus InferDataTypeForXXX(InferDataTypeContext *context) {
  auto data_type = context->GetInputDataType(0);
  ...
}
```
