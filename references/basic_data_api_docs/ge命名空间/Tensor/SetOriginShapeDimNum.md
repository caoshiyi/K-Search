# SetOriginShapeDimNum

**页面ID:** atlasopapi_07_00480  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00480.html

---

#### 函数功能

设置原始shape的维度大小，即rank大小。

#### 函数原型

```
graphStatus SetOriginShapeDimNum(const size_t dim_num)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dim_num | 输入 | 原始shape的维度大小，即原始shape的rank。 |

#### 返回值

graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

无。
