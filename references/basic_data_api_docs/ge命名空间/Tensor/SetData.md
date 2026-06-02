# SetData

**页面ID:** atlasopapi_07_00474  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00474.html

---

#### 函数功能

向Tensor中设置数据。

#### 函数原型

> **注意:** 

数据类型为string的接口后续版本会废弃，建议使用数据类型为非string的接口。

```
graphStatus SetData(std::vector<uint8_t> &&data)
graphStatus SetData(const std::vector<uint8_t> &data)
graphStatus SetData(const uint8_t *data, size_t size)
graphStatus SetData(const std::string &data)
graphStatus SetData(const char_t *data)
graphStatus SetData(const std::vector<std::string> &data)
graphStatus SetData(const std::vector<AscendString> &datas)
graphStatus SetData(uint8_t *data, size_t size, const Tensor::DeleteFunc &deleter_func)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| data/datas | 输入 | 需设置的数据。 |
| size | 输入 | 数据的长度，单位为字节。 |
| deleter_func | 输入 | 用于释放data数据。 ``` using DeleteFunc = std::function<void(uint8_t *)>; ``` |

#### 返回值

graphStatus类型：设置成功返回GRAPH_SUCCESS，否则，返回GRAPH_FAILED。

#### 异常处理

无。

#### 约束说明

无。
