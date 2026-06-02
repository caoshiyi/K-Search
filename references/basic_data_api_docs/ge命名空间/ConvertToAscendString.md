# ConvertToAscendString

**页面ID:** atlasopapi_07_00499  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00499.html

---

#### 函数功能

模板函数，接受一个模板参数T，并将其转换为AscendString类型。这个函数的主要功能是将不同类型的字符串转换为AscendString类型。

#### 函数原型

```
template<typename T> ge::AscendString ConvertToAscendString(T str)
```

支持以下几种拓展：

- template<> inline ge::AscendString ConvertToAscendString<const char *>(const char *str)

对于const char *类型的字符串，直接使用AscendString的构造函数进行转换。

- template<> inline ge::AscendString ConvertToAscendString<std::string>(std::string str)

对于std::string类型的字符串，先将其转换为const char *类型，然后再进行转换。

- template<> inline ge::AscendString ConvertToAscendString<ge::AscendString>(ge::AscendString str)

对于AscendString类型的字符串，直接返回AscendString类型字符串。

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| str | 输入 | 待转换的字符串。 |

#### 返回值

转换后的AscendString类型字符串。

#### 异常处理

无。

#### 约束说明

无。
