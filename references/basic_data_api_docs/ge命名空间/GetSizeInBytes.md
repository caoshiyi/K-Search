# GetSizeInBytes

**页面ID:** atlasopapi_07_00509  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00509.html

---

#### 函数功能

根据传入的element_count和data_type，获取element_count个该data_type所占用的内存总大小。

#### 函数原型

```
int64_t GetSizeInBytes(int64_t element_count, DataType data_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| element_count | 输入 | 用于标识多少个data_type。 |
| data_type | 输入 | 数据类型，请参见DataType。 |

#### 返回值

如果传入个数为非法值或传入不支持的数据类型，返回-1。

#### 异常处理

无。

#### 约束说明

无。
