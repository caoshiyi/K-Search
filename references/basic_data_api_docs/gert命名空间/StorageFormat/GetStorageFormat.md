# GetStorageFormat

**页面ID:** atlasopapi_07_00166  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00166.html

---

#### 函数功能

获取运行时format。

#### 函数原型

```
ge::Format GetStorageFormat() const
```

#### 参数说明

无。

#### 返回值说明

运行时format。

#### 约束说明

无。

#### 调用示例

```
ExpandDimsType dim_type("1100");
StorageFormat format(ge::Format::FORMAT_NCHW, ge::Format::FORMAT_C1HWNC0, dim_type);
auto storage_format = format.GetStorageFormat();  // Format::FORMAT_C1HWNC0
```
