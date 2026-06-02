# OutputTensors

**页面ID:** atlasopapi_07_00641  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00641.html

---

#### 函数功能

设置输出Tensor 。

#### 函数原型

```
OpTilingContextBuilder &OutputTensors(const std::vector<gert::Tensor *> &outputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| outputs | 输入 | 输出Tensor的指针数组。 |

#### 返回值说明

OpTilingContextBuilder对象本身，用于链式调用。

#### 约束说明

- 在调用Build方法之前，必须设置OutputTensors，否则构造出的TilingContext将包含未定义数据。
- 通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在ContextHolder对象的生命周期内有效。
