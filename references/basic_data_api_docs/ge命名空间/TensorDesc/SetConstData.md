# SetConstData

**页面ID:** atlasopapi_07_00443  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00443.html

---

#### 函数功能

如果TensorDesc是常量节点的描述，向TensorDesc中设置权重值。

#### 函数原型

```
void SetConstData(std::unique_ptr<uint8_t[]> const_data_buffer, const size_t &const_data_len)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| const_data_buffer | 输入 | 权重地址。 |
| const_data_len | 输入 | 权重长度。 |

#### 返回值

无。

#### 异常处理

无。

#### 约束说明

无。
