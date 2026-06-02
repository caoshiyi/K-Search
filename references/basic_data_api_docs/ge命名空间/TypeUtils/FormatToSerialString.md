# FormatToSerialString

**页面ID:** atlasopapi_07_00493  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00493.html

---

#### 函数功能

将Format类型值转化为字符串表达。

从GCC 5.1版本开始，libstdc++为了更好的实现C++11规范，更改了std::string和std::list的一些接口，导致新老版本ABI不兼容。所以推荐使用FormatToAscendString替代本接口。

使用该接口需要包含type_utils.h头文件。

```
#include "graph/utils/type_utils.h"
```

#### 函数原型

```
static std::string FormatToSerialString(const Format format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 待转换的Format，支持的Format请参考Format。 |

#### 返回值说明

转换后的Format字符串。

#### 约束说明

无。

#### 调用示例

```
ge::Format format = ge::Format::FORMAT_NHWC;
auto format_str = ge::TypeUtils::FormatToSerialString(format); // "NHWC"
```
