# GetStorageFormat

**页面ID:** atlasopapi_07_00752  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00752.html

---

#### 函数功能

获取运行时Tensor的数据格式。

#### 函数原型

```
ge::Format GetStorageFormat() const
```

#### 参数说明

无

#### 返回值说明

返回运行时数据格式。

关于ge::Format类型的定义，请参见Format。

#### 约束说明

无

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
t.SetOriginFormat(ge::FORMAT_NHWC);
t.SetStorageFormat(ge::FORMAT_NC1HWC0);
auto fmt = t.GetStorageFormat(); // ge::FORMAT_NC1HWC0
```
