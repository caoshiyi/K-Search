# GetFormatFromSub

**页面ID:** atlasopapi_07_00504  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00504.html

---

#### 函数功能

根据传入的主format和子format信息得到实际的format。

实际format为4字节大小，第1个字节的高四位为预留字段，低四位为C0 format，第2-3字节为子format信息，第4字节为主format信息，如下：

/*

* ---------------------------------------------------

* |     4bits     |      4bits    |        2bytes     | 1byte  |

* |------------|------------|----------------|--------|

* |  reserved  | C0 format |   Sub format | format |

* ---------------------------------------------------

*/

#### 函数原型

```
inline int32_t GetFormatFromSub(int32_t primary_format, int32_t sub_format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| primary_format | 输入 | 主format信息，值不超过0xffU。 |
| sub_format | 输入 | 子format信息，值不超过0xffffU。 |

#### 返回值

指定的主format和子format对应的实际format。

#### 异常处理

无。

#### 约束说明

无。
