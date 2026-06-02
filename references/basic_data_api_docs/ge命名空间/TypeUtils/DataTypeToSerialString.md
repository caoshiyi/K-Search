# DataTypeToSerialString

**页面ID:** atlasopapi_07_00491  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00491.html

---

#### 函数功能

将DataType类型值转化为字符串表达。

从GCC 5.1版本开始，libstdc++为了更好的实现C++11规范，更改了std::string和std::list的一些接口，导致新老版本ABI不兼容。所以推荐使用DataTypeToAscendString替代本接口。

使用该接口需要包含type_utils.h头文件。

```
#include "graph/utils/type_utils.h"
```

#### 函数原型

```
static std::string DataTypeToSerialString(const DataType data_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data_type | 输入 | 待转换的DataType，支持的DataType请参考DataType。 |

#### 返回值说明

转换后的DataType字符串。

#### 约束说明

无。

#### 调用示例

```
DataType data_type = ge::DT_UINT32;
auto type_str = ge::TypeUtils::DataTypeToSerialString(data_type); // "DT_UINT32"
```
