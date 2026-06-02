# GetShape

**页面ID:** atlasopapi_07_00440  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00440.html

---

#### 函数功能

获取TensorDesc所描述Tensor的Shape。

#### 函数原型

```
Shape GetShape() const
```

#### 参数说明

无。

#### 返回值

Shape类型，TensorDesc描述的Shape。

#### 异常处理

无。

#### 约束说明

由于返回的Shape信息为值拷贝，因此修改返回的Shape信息，不影响TensorDesc中已有的Shape信息。
