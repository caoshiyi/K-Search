# OpRegistrationData构造函数和析构函数

**页面ID:** atlasopapi_07_00385  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00385.html

---

#### 函数功能

OpRegistrationData构造函数和析构函数。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
OpRegistrationData(const std::string &om_optype)
OpRegistrationData(const char_t *om_optype)
~OpRegistrationData()
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| om_optype | 输入 | 指定适配昇腾AI处理器的模型支持的算子类型。 |

#### 返回值

OpRegistrationData构造函数返回OpRegistrationData类型的对象。

#### 异常处理

无。

#### 约束说明

无。
