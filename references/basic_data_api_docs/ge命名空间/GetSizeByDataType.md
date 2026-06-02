# GetSizeByDataType

**页面ID:** atlasopapi_07_00508  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00508.html

---

#### 函数功能

根据传入的data_type，获取该data_type所占用的内存大小。如果要获取多个元素的内存大小，请使用GetSizeInBytes。

#### 函数原型

```
inline int GetSizeByDataType(DataType data_type)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| data_type | 输入 | 数据类型，请参见DataType。 |

#### 返回值

该data_type所占用的内存大小（单位为bytes）。

如果该data_type所占用的内存小于1byte，返回1000+该数据类型的bit位数，比如DT_INT4数据类型，返回1004。

如果传入非法值或不支持的数据类型，返回-1。

#### 异常处理

无。

#### 约束说明

无。
