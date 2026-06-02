# HasC0Format

**页面ID:** atlasopapi_07_00511  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00511.html

---

#### 函数功能

判断实际format中是否包含C0 format。

#### 函数原型

```
inline bool HasC0Format(int32_t format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 实际format（4字节大小，第1个字节的高四位为预留字段，低四位为C0 format，第2-3字节为子format信息，第4字节为主format信息）。 |

#### 返回值

- true：实际format中包含C0 format。
- false：实际format中不包含C0 format。

#### 异常处理

无。

#### 约束说明

无。
