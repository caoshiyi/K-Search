# Tensor构造函数和析构函数

**页面ID:** atlasopapi_07_00459  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00459.html

---

#### 函数功能

Tensor构造函数和析构函数。

#### 函数原型

```
Tensor()
~Tensor() = default
explicit Tensor(const TensorDesc &tensor_desc)
Tensor(const TensorDesc &tensor_desc, const std::vector<uint8_t> &data)
Tensor(const TensorDesc &tensor_desc, const uint8_t *data, size_t size)
Tensor(TensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor_desc | 输入 | TensorDesc对象，需设置的Tensor描述符。 |
| data | 输入 | 需设置的数据。 |
| size | 输入 | 数据的长度，单位为字节。 |

#### 返回值

Tensor构造函数返回Tensor类型的对象。

#### 异常处理

无。

#### 约束说明

无。
