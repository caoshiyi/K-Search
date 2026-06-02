# MutableOriginShape

**页面ID:** atlasopapi_07_00209  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00209.html

---

#### 函数功能

获取Tensor的原始shape。

#### 函数原型

**Shape &MutableOriginShape()**

#### 参数说明

无。

#### 返回值说明

原始shape引用。

关于Shape类型的定义，请参见Shape。

#### 约束说明

无。

#### 调用示例

```
StorageShape sh({1, 2, 3}, {2, 1, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto shape = t.MutableOriginShape(); // 1,2,3
```
