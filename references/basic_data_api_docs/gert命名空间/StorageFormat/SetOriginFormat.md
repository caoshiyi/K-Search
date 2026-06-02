# SetOriginFormat

**页面ID:** atlasopapi_07_00165  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00165.html

---

#### 函数功能

设置原始format。

#### 函数原型

```
void SetOriginFormat(const ge::Format origin_format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| origin_format | 输入 | 原始format。 |

#### 返回值说明

无。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
format.SetOriginFormat(ge::Format::FORMAT_NC);
```
