# GetOriginFormat

**页面ID:** atlasopapi_07_00164  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00164.html

---

#### 函数功能

获取原始format。

#### 函数原型

```
ge::Format GetOriginFormat() const
```

#### 参数说明

无。

#### 返回值说明

原始format。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
auto origin_format = format.GetOriginFormat();  // Format::FORMAT_NCHW
```
