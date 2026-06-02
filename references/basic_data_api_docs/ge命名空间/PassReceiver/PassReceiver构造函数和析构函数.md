# PassReceiver构造函数和析构函数

**页面ID:** atlasgeapi_07_0067  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasgeapi_07_0067.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

#### 功能说明

PassReceiver构造函数和析构函数。

#### 函数原型

```
PassReceiver(PassRegistrationData &reg_data)
~PassReceiver() = default
```

#### 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| reg_data | 输入 | 需要注册的pass信息。 |

#### 返回值说明

PassReceiver构造函数返回PassReceiver类型的对象。

#### 约束说明

无
