# GetDataTypeLength

**页面ID:** atlasopapi_07_00496  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00496.html

---

#### 函数功能

获取数据类型所占内存大小。

使用该接口需要包含type_utils.h头文件。

```
#include "graph/utils/type_utils.h"
```

#### 函数原型

```
static bool GetDataTypeLength(const ge::DataType data_type, uint32_t &length)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data_type | 输入 | 数据类型。 |
| length | 输出 | 数据类型所占内存大小，单位：字节。 |

#### 返回值说明

获取成功时返回true；data_type不支持时返回false。

#### 约束说明

无。

#### 调用示例

```
uint32_t type_length;
ge::DataType data_type = ge::DT_INT8;
const bool ret = ge::TypeUtils::GetDataTypeLength(data_type, type_length); // type_length 1
if (!ret) {
  // ...
}
```
