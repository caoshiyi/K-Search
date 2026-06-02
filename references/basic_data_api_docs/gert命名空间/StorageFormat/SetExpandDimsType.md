# SetExpandDimsType

**页面ID:** atlasopapi_07_00169  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00169.html

---

#### 函数功能

设置补维规则。

#### 函数原型

```
void SetExpandDimsType(const ExpandDimsType &expand_dims_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| expand_dims_type | 输入 | 补维规则。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
ExpandDimsType new_dim_type("1010");
format.SetExpandDimsType(new_dim_type);
```
