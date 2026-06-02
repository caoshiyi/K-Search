# GetFormat

**页面ID:** atlasopapi_07_00434  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00434.html

---

#### 函数功能

获取TensorDesc所描述的Tensor的Format。

#### 函数原型

```
Format GetFormat() const
```

#### 参数说明

无。

#### 返回值

Format类型，TensorDesc所描述的Tensor的Format信息。

#### 异常处理

无。

#### 约束说明

由于返回的Format信息为值拷贝，因此修改返回的Format信息，不影响TensorDesc中已有的Format信息。
