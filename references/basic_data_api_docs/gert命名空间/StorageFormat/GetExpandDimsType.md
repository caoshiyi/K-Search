# GetExpandDimsType

**页面ID:** atlasopapi_07_00168  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00168.html

---

#### 函数功能

获取补维规则。

#### 函数原型

```
ExpandDimsType GetExpandDimsType() const
```

#### 参数说明

无。

#### 返回值说明

补维规则。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
auto fmt_dim_type = format.GetExpandDimsType();
```
