# GetConstData

**页面ID:** atlasopapi_07_00431  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00431.html

---

#### 函数功能

如果TensorDesc是常量节点的描述，获取TensorDesc中的权重值。

#### 函数原型

```
bool GetConstData(uint8_t **const_data_buffer, size_t &const_data_len) const
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| const_data_buffer | 输出 | 权重地址。 |
| const_data_len | 输出 | 权重长度。 |

#### 返回值

获取成功，返回true。

获取失败，返回false。

#### 异常处理

无。

#### 约束说明

无。
