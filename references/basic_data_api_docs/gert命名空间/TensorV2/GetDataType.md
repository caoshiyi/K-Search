# GetDataType

**页面ID:** atlasopapi_07_00744  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00744.html

---

#### 函数功能

获取Tensor的数据类型。

#### 函数原型

```
ge::DataType GetDataType() const
```

#### 参数说明

无。

#### 返回值说明

返回Tensor中的数据类型。

关于ge::DataType的定义，请参见DataType。

#### 约束说明

无。

#### 调用示例

```
StorageShape sh({1, 2, 3}, {1, 2, 3});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto dt = t.GetDataType(); //  ge::DT_FLOAT
```
