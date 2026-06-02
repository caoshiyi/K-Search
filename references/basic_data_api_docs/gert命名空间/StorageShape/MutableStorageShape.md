# MutableStorageShape

**页面ID:** atlasopapi_07_00179  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00179.html

---

#### 函数功能

获取可写的运行时shape。

#### 函数原型

```
Shape &MutableStorageShape()
```

#### 参数说明

无。

#### 返回值说明

可写的运行时shape。

#### 约束说明

无。

#### 调用示例

```
StorageShape shape({3, 256, 256}, {256, 256, 3});
auto storage_shape = shape.MutableStorageShape(); // 256,256,3
```
