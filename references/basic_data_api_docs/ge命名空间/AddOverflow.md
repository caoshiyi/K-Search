# AddOverflow

**页面ID:** atlasopapi_07_00562  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00562.html

---

#### 函数功能

该模板函数用于判断两个数值相加是否会发生溢出，并在不溢出的情况下返回正确的计算结果。

#### 函数原型

```
template<typename TLhs, typename TRhs, typename TRet>
bool AddOverflow(TLhs lhs, TRhs rhs, TRet &ret)
```

#### 参数说明

**表1 **模板参数说明

| 参数 | 说明 |
| --- | --- |
| TLhs | 加法左操作数的数据类型。 |
| TRhs | 加法右操作数的数据类型。 |
| TRet | 加法计算结果的数据类型。 |

**表2 **参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| lhs | 输入 | 加法左操作数。 |
| rhs | 输入 | 加法右操作数。 |
| ret | 输出 | 左右操作数相加的结果，只有函数返回值为true时，该结果才有效。 |

#### 返回值

true表示计算失败，ret返回值无效；false表示计算成功，ret返回值有效。

#### 约束说明

无。

#### 调用示例

```
// ...
ge::DataType out_data_type = ge::DT_FLOAT;
GE_ASSERT_GRAPH_SUCCESS(GetOutputDataType(context, out_data_type), "get data type failed");
GE_ASSERT_TRUE(out_data_type == ge::DataType::DT_INT32 || out_data_type == ge::DataType::DT_INT64,
               "only support DT_INT32 and DT_INT64, but out_data_type is %s",
               ge::TypeUtils::DataTypeToSerialString(out_data_type).c_str());
const auto is_malloc = (out_data_type == ge::DataType::DT_INT32);
const auto data_type_size = ge::GetSizeByDataType(out_data_type);
if (data_type_size <= 0) {
  // 报错
}
size_t malloc_buffer_size = 0U;
if (ge::MulOverflow(static_cast<size_t>(data_type_size), Shape::kMaxDimNum, malloc_buffer_size)) {
  // 报错
}
if (ge::AddOverflow(malloc_buffer_size, sizeof(GertTensorData), malloc_buffer_size)) {
  // 报错
}
```
