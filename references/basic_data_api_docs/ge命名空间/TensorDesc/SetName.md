# SetName

**页面ID:** atlasopapi_07_00447  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00447.html

---

#### 函数功能

向TensorDesc中设置Tensor的名称。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
void SetName(const std::string &name)
void SetName(const char_t *name)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 需设置的Tensor的名称。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
