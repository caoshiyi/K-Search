# AscendString构造函数和析构函数

**页面ID:** atlasopapi_07_00283  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00283.html

---

#### 函数功能

AscendString构造函数和析构函数。

#### 函数原型

```
AscendString() = default
~AscendString() = default
AscendString(const char_t *const name)
AscendString(const char_t *const name, size_t length)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 字符串名称。 |
| length | 输入 | 字符串长度。 |

#### 返回值

AscendString构造函数返回AscendString类型的对象。

#### 异常处理

无。

#### 约束说明

无。
