# Expand（更改原shape）

**页面ID:** atlasopapi_07_00068  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00068.html

---

#### 函数功能

对shape做补维，并将补维后的结果直接更新原shape对象。

#### 函数原型

```
ge::graphStatus Expand(Shape &shape) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 既是输入，又是输出 | 需要进行补维操作的shape对象。 |

#### 返回值说明

补维成功返回ge::GRAPH_SUCCESS。

关于ge::graphStatus类型的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
Shape shape({3, 256, 256}); // 设置原始shape 3,256,256
ExpandDimsType type1("1000");
auto ret = type1.Expand(shape); // ret = ge::GRAPH_SUCCESS, shape = 1,3,256,256
```
