# ImplyType

**页面ID:** atlasopapi_07_00395  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00395.html

---

#### 函数功能

设置算子执行方式。

#### 函数原型

```
OpRegistrationData &ImplyType(const domi::ImplyType &imply_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 算子执行方式。 ``` enum class ImplyType : unsigned int { BUILDIN = 0,// 内置算子，由OME正常执行 TVM,        // 编译成tvm bin文件执行 CUSTOM,     // 由用户自定义计算逻辑，通过CPU执行 AI_CPU,      // AI CPU 自定义算子类型 INVALID = 0xFFFFFFFF, }; ``` |  |  |
