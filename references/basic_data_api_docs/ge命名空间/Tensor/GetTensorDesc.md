# GetTensorDesc

**页面ID:** atlasopapi_07_00473  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00473.html

---

#### 函数功能

获取Tensor的描述符。

#### 函数原型

```
TensorDesc GetTensorDesc() const
```

#### 参数说明

无。

#### 返回值

返回当前Tensor的描述符，TensorDesc类型。

#### 异常处理

无。

#### 约束说明

修改返回的TensorDesc信息，不影响Tensor对象中已有的TensorDesc信息。
