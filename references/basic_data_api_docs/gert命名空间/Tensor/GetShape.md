# GetShape

**页面ID:** atlasopapi_07_00210  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00210.html

---

#### 函数功能

获取Tensor的shape，包含运行时和原始shape。

#### 函数原型

**const StorageShape &GetShape() const**

**StorageShape &GetShape()**

#### 参数说明

无。

#### 返回值说明

- const StorageShape &GetShape() const：返回只读的shape引用。
- StorageShape &GetShape()：返回shape引用。

关于StorageShape类型的定义，请参见StorageShape。

#### 约束说明

无。

#### 调用示例

```
StorageShape sh({1, 2, 3}, {2, 1, 3});
Tensor t = {sh, {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}}, kOnHost, ge::DT_FLOAT, nullptr};
auto shape = t.GetShape(); // sh
```
