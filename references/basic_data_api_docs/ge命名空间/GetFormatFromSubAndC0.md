# GetFormatFromSubAndC0

**页面ID:** atlasopapi_07_00505  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00505.html

---

#### 函数功能

根据传入的主format，子format和C0 format信息得到实际的format。

实际format为4字节大小，第1个字节的高四位为预留字段，低四位为C0 format，第2-3字节为子format信息，第4字节为主format信息，如下：

/*

* ---------------------------------------------------

* |     4bits    |      4bits      |        2bytes    | 1byte  |

* |------------|-------------|----------------|--------|

* |  reserved  | C0 format |   Sub format   | format |

* ---------------------------------------------------

*/

#### 函数原型

```
inline int32_t GetFormatFromSubAndC0(int32_t primary_format, int32_t sub_format, int32_t c0_format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| primary_format | 输入 | 主format信息，值不超过0xffU。 |
| sub_format | 输入 | 子format信息，值不超过0xffffU。 |
| c0_format | 输入 | C0 format信息，值不超过0xfU。 |

#### 返回值

指定的primary_format、sub_format和c0_format对应的实际format。

#### 异常处理

无。

#### 约束说明

无。
