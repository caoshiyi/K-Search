# SetDim

**页面ID:** atlasopapi_07_00427  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00427.html

---

#### 函数功能

将Shape中第idx维度的值设置为value。

#### 函数原型

```
graphStatus SetDim(size_t idx, int64_t value)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| idx | 输入 | Shape维度的索引，索引从0开始。 |
| value | 输入 | 需设置的值。 |

#### 返回值

graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

使用SetDim接口前，只能使用Shape(const std::vector<int64_t>& dims)构造shape对象。如果使用Shape()构造shape对象，使用SetDim接口将返回失败。
