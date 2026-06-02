# Compat

**页面ID:** atlasopapi_07_00561  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00561.html

---

#### 函数功能

检查入参的数值是否超过指定数据类型T所能表示的数值范围。

#### 函数原型

**template<typename T1>**

**static bool Compat(const T1 v)**

#### 参数说明

**表1 **模板参数说明

| 参数 | 说明 |
| --- | --- |
| T1 | 表示需要检查的数值类型。支持有符号整型和无符号整型，例如：int64_t、uint32_t等。 |

**表2 **参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| v | 输入 | 需要检查的数值，类型为T1。 |

#### 返回值说明

返回bool类型：

- 如果数值v在数据类型T的表示范围内，返回true。
- 如果数值v超出数据类型T的表示范围，返回false。

#### 约束说明

无。

#### 调用示例

```
const int64_t value = XXX;
if (!ge::IntegerChecker<int32_t>::Compat(value)) {
   // 报错
}
```
