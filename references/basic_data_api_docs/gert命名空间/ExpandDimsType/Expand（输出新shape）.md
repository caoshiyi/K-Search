# Expand（输出新shape）

**页面ID:** atlasopapi_07_00067  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00067.html

---

#### 函数功能

对shape做补维，并将补维后的结果写入指定的输出shape对象。

#### 函数原型

```
ge::graphStatus Expand(const Shape &shape, Shape &out_shape) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 输入shape，补维前shape。 |
| out_shape | 输出 | 输出shape，补维后shape。 |

#### 返回值说明

补维成功返回ge::GRAPH_SUCCESS；失败则返回ge::GRAPH_FAILED。

关于ge::graphStatus类型的定义，请参见ge::graphStatus。

#### 约束说明

无。

#### 调用示例

```
Shape origin_shape({3, 256, 256}); // 设置原始shape 3x256x256
Shape out_shape;
ExpandDimsType type1("1000");
ExpandDimsType type2("10000");
ExpandDimsType type3("1001");
auto ret = type1.Expand(origin_shape, out_shape); // ret = ge::GRAPH_SUCCESS, out_shape = 1,3,256,256
ret = type2.Expand(origin_shape, out_shape); // ret = ge::GRAPH_FAILED
ret = type3.Expand(origin_shape, out_shape); // ret = ge::GRAPH_SUCCESS, out_shape = 1,3,256,1,256
```
