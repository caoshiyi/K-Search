# DataFormatToFormat

**页面ID:** atlasopapi_07_00495  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00495.html

---

#### 函数功能

将数据格式字符串转化为Format类型值。

使用该接口需要包含type_utils.h头文件。

```
#include "graph/utils/type_utils.h"
```

#### 函数原型

```
static Format DataFormatToFormat(const AscendString &str)
static Format DataFormatToFormat(const std::string &str)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| str | 输入 | 待转换的Format字符串形式。 从GCC 5.1版本开始，libstdc++为了更好的实现C++11规范，更改了std::string和std::list的一些接口，导致新老版本ABI不兼容。所以推荐使用入参为**AscendString**类型的接口。 |

#### 返回值说明

输入合法时，返回转换后的Format enum值，枚举定义请参考Format，仅支持部分格式，详见约束说明；输入不合法时，返回FORMAT_RESERVED，并打印报错信息。

#### 约束说明

只支持以下几种格式：

"NCHW": FORMAT_NCHW

"NHWC": FORMAT_NHWC

"NDHWC": FORMAT_NDHWC

"NCDHW": FORMAT_NCDHW

"ND": FORMAT_ND

#### 调用示例

```
// 如果使用的是AscendString入参的接口(建议使用)
ge::AscendString format_str = "NHWC";
auto format = ge::TypeUtils::DataFormatToFormat(format_str); // 1

// 如果使用的是std:string入参的接口(不建议使用)
std::string format_str = "NHWC";
auto format = ge::TypeUtils::DataFormatToFormat(format_str); // 1
```
