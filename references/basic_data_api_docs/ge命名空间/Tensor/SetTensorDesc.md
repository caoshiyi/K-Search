# SetTensorDesc

**页面ID:** atlasopapi_07_00484  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00484.html

---

#### 函数功能

设置Tensor的描述符（TensorDesc）。

#### 函数原型

```
graphStatus SetTensorDesc(const TensorDesc &tensor_desc)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor_desc | 输入 | 需设置的Tensor描述符。 |

#### 返回值

graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

无。
