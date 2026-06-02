# MutableOriginShape

**页面ID:** atlasopapi_07_00178  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00178.html

---

#### 函数功能

获取可写的原始shape。

#### 函数原型

```
Shape &MutableOriginShape()
```

#### 参数说明

无。

#### 返回值说明

可写的原始shape。

#### 约束说明

无。

#### 调用示例

```
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto origin_shape = shape.MutableOriginShape(); // 3,256,256
```
