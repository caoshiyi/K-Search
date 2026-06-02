# GetFormatName

**页面ID:** atlasopapi_07_00506  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00506.html

---

#### 函数功能

根据传入的Format类型，获取Format的字符串描述。

#### 函数原型

```
const char_t *GetFormatName(Format format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | Format枚举值。 |

#### 返回值

该Format所对应的字符串描述，若Format不合法或不被识别，则返回nullptr。

#### 异常处理

无。

#### 约束说明

返回的字符串不可被修改。
