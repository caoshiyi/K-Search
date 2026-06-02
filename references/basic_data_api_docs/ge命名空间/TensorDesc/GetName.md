# GetName

**页面ID:** atlasopapi_07_00435  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00435.html

---

#### 函数功能

获取TensorDesc所描述Tensor的名称。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
std::string GetName() const
graphStatus GetName(AscendString &name)
graphStatus GetName(AscendString &name) const
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输出 | 算子名称。 |

#### 返回值

graphStatus类型：获取name成功，返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

无。
