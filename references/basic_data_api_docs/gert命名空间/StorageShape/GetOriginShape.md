# GetOriginShape

**页面ID:** atlasopapi_07_00176  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00176.html

---

#### 函数功能

获取原始shape。

#### 函数原型

```
const Shape &GetOriginShape() const
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
auto origin_shape = shape.GetOriginShape(); // 3,256,256
```
