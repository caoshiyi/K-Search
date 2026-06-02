# MutableExpandDimsType

**页面ID:** atlasopapi_07_00170  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00170.html

---

#### 函数功能

获取可写的补维规则。

#### 函数原型

```
ExpandDimsType &MutableExpandDimsType()
```

#### 参数说明

无。

#### 返回值说明

补维规则引用。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
ExpandDimsType new_dim_type("1010");
format.SetExpandDimsType(new_dim_type);
auto &fmt_dim_type = format.MutableExpandDimsType();
```
