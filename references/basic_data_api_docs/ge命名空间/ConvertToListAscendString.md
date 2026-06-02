# ConvertToListAscendString

**页面ID:** atlasopapi_07_00500  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00500.html

---

#### 函数功能

定义了一个模板函数ConvertToListAscendString，用于将不同类型的字符串列表转换为AscendString类型的列表。

#### 函数原型

```
template<typename T> std::vector<ge::AscendString> ConvertToListAscendString(T strs)
```

支持以下两种拓展：

- template<> inline std::vector<ge::AscendString> ConvertToListAscendString(std::vector<std::string> strs)

对于std::vectorstd::string类型的字符串列表，先将其转换为std::vector<const char *>类型，然后再进行转换。

- template<> inline std::vector<ge::AscendString> ConvertToListAscendString(std::vector<ge::AscendString> strs)

对于std::vector<ge::AscendString>类型的字符串列表，直接返回原列表。

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| strs | 输入 | 待转换的字符串列表。 |

#### 返回值

转换后的AscendString类型字符串列表。

#### 异常处理

无。

#### 约束说明

无。
