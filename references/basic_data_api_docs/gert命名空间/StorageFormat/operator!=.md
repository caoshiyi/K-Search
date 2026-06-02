# operator!=

**页面ID:** atlasopapi_07_00172  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00172.html

---

#### 函数功能

判断格式是否不相同。

#### 函数原型

```
bool operator!=(const StorageFormat &other) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| other | 输入 | 另一种格式。 |

#### 返回值说明

true代表格式不相同，false代表格式相同。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
StorageFormat another_format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NC, dim_type);
bool is_diff_fmt = format != another_format; // true
```
