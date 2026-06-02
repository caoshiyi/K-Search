# GetOriginShape

**页面ID:** atlasopapi_07_00749  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00749.html

---

#### 函数功能

获取Tensor的原始Shape。

#### 函数原型

```
const Shape &GetOriginShape() const
```

#### 参数说明

无。

#### 返回值说明

只读的原始Shape引用。

关于Shape类型的定义，请参见Shape。

#### 约束说明

无。

#### 调用示例

```
StorageShape sh({1, 2, 3}, {2, 1, 3});
TensorV2 t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto shape = t.GetOriginShape(); // 1,2,3
```
