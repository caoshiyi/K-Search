# FrameworkType

**页面ID:** atlasopapi_07_00387  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00387.html

---

#### 函数功能

设置原始模型的框架类型。

#### 函数原型

```
OpRegistrationData &FrameworkType(const domi::FrameworkType &fmk_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 框架类型： - CAFFE- TENSORFLOW- ONNX  FrameworkType枚举值的定义如下： ``` enum FrameworkType { CAFFE = 0, MINDSPORE = 1, TENSORFLOW = 3, ANDROID_NN, ONNX, FRAMEWORK_RESERVED, }; ``` |  |  |
