# GetShape

**页面ID:** atlasopapi_07_00733  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00733.html

---

#### 函数功能

获取原始shape。

#### 函数原型

```
const Shape &GetShape() const
```

#### 参数说明

无。

#### 返回值说明

原始shape。

#### 约束说明

无。

#### 调用示例

```
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto origin_shape = shape.GetShape(); // 3,256,256
```
