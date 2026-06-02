# GetDataType

**页面ID:** atlasopapi_07_00432  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00432.html

---

#### 函数功能

获取TensorDesc所描述Tensor的数据类型。

#### 函数原型

```
DataType GetDataType() const
```

#### 参数说明

无。

#### 返回值

DataType类型，TensorDesc所描述的Tensor的数据类型。

#### 异常处理

无。

#### 约束说明

由于返回的DataType信息为值拷贝，因此修改返回的DataType信息，不影响TensorDesc中已有的DataType信息。
