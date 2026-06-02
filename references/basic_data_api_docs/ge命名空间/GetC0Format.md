# GetC0Format

**页面ID:** atlasopapi_07_00501  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00501.html

---

#### 函数功能

根据实际format获取C0 format的值。C0值是昇腾AI处理器特有数据格式所需要的参数。例如NC1HWC0，最后一维即表示C0值，C0值在不同的昇腾AI处理器中数值不同，通过C0 format来承载。

实际format为4字节大小，第1个字节的高四位为预留字段，低四位为C0 format，第2-3字节为子format信息，第4字节为主format信息，如下所示：

/*

* ---------------------------------------------------

* |     4bits     |      4bits     |        2bytes    | 1byte  |

* |------------|-------------|----------------|--------|

* |  reserved  | C0 format |   Sub format | format |

* ---------------------------------------------------

*/

#### 函数原型

```
inline int32_t GetC0Format(int32_t format)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| format | 输入 | 实际format（4字节大小，第1个字节的高四位为预留字段，低四位为C0 format段，第2-3字节为子format信息，第4字节为主format信息）。 |

#### 返回值

- 如果包含C0 format，返回C0 format的值。
- 如果不包含C0 format，返回0。

#### 异常处理

无。

#### 约束说明

无。
