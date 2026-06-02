# GetPrimaryFormat

**页面ID:** atlasopapi_07_00507  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00507.html

---

#### 函数功能

从实际format中解析出主format信息。

#### 函数原型

```
inline int32_t GetPrimaryFormat(int32_t format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 实际format（4字节大小，第1个字节的高四位为预留字段，低四位为C0 format，第2-3字节为子format信息，第4字节为主format信息）。 |

#### 返回值

实际format中包含的主format。

#### 异常处理

无。

#### 约束说明

无。
