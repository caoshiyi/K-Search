# SetExpandDimsType

**页面ID:** atlasopapi_07_00020  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00020.html

---

#### 函数功能

设置原始Format向运行时Format转换时的补维规则。

#### 函数原型

```
void SetExpandDimsType(const ExpandDimsType &expand_dims_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| expand_dims_type | 输入 | 需要设置的补维规则。 关于ExpandDimsType的说明，请参见ExpandDimsType。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
auto dtype_ = ge::DataType::DT_INT32;
StorageFormat fmt_(ge::Format::FORMAT_NC, ge::FORMAT_NCHW, {});
ExpandDimsType type_("1001");
gert::CompileTimeTensorDesc td;
td.SetDataType(dtype_);
auto dtype = td.GetDataType(); // ge::DataType::DT_INT32;
td.SetStorageFormat(fmt_.GetStorageFormat());
auto storage_fmt = td.GetStorageFormat(); // ge::FORMAT_NCHW
td.SetOriginFormat(fmt_.GetOriginFormat());
auto origin_fmt = td.GetOriginFormat(); // ge::Format::FORMAT_NC
td.SetExpandDimsType(type_);
auto type = td.GetExpandDimsType(); // type_("1001")
```
