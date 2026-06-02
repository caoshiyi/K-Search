# TilingData

**页面ID:** atlasopapi_07_00637  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00637.html

---

#### 函数功能

设置算子的TilingData指针，用于算子Tiling计算时，向该指针中写入TilingData数据。

#### 函数原型

```
OpTilingContextBuilder &TilingData(const gert::TilingData *tiling_data, gert::Chain::Deleter deleter = nullptr)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_data | 输入 | TilingData数据指针。 |
| deleter | 输入 | Tiling数据的删除器，如果用户显式传入删除器deleter，ContextHolder析构时会调用删除器释放Tiling数据，默认不传入。建议使用TilingDataSize接口。 |

#### 返回值说明

OpTilingContextBuilder对象本身，用于链式调用。

#### 约束说明

- 在调用Build方法之前，必须设置TilingData或TilingDataSize，否则构造出的TilingContext将包含未定义数据。
- 通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在ContextHolder对象的生命周期内有效。
