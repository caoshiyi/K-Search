# MutableShape

**页面ID:** atlasopapi_07_00734  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00734.html

---

#### 函数功能

获取可写的原始shape。

#### 函数原型

```
Shape &MutableShape()
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
auto origin_shape = shape.MutableShape(); // 3,256,256
```
