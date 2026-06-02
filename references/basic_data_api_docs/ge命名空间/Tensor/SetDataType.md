# SetDataType

**页面ID:** atlasopapi_07_00475  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00475.html

---

#### 函数功能

设置Tensor的Datatype。

#### 函数原型

```
graphStatus SetDataType(const ge::DataType &dtype)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dtype | 输入 | 需设置的DataType值。 关于DataType类型，请参见DataType。 |

#### 返回值

graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

无。
