# DataTypeToAscendString

**页面ID:** atlasopapi_07_00487  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00487.html

---

#### 函数功能

将DataType类型值转化为字符串表达。

使用该接口需要包含type_utils.h头文件。

```
#include "graph/utils/type_utils.h"
```

#### 函数原型

```
static AscendString DataTypeToAscendString(const DataType &data_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data_type | 输入 | 待转换的DataType，支持的DataType请参考DataType。 |

#### 返回值说明

转换后的DataType字符串，AscendString类型。

#### 约束说明

无。

#### 调用示例

```
DataType data_type = ge::DT_UINT32;
auto type_str = ge::TypeUtils::DataTypeToAscendString(data_type); // "DT_UINT32"
const char *ptr = type_str.GetString();  // 获取char*指针
```
