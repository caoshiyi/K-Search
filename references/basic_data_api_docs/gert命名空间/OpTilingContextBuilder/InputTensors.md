# InputTensors

**页面ID:** atlasopapi_07_00640  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00640.html

---

#### 函数功能

设置输入Tensor指针，用于在Tiling计算时，可通过该Builder类构造的上下文TilingContext获取相应的输入Tensor指针。

#### 函数原型

```
OpTilingContextBuilder &InputTensors(const std::vector<gert::Tensor *> &inputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 设置输入Tensor指针。 |

#### 返回值说明

OpTilingContextBuilder对象本身，用于链式调用。

#### 约束说明

- 在调用Build方法之前，必须设置InputTensors，否则构造出的TilingContext将包含未定义数据。
- 通过指针传入的参数（void*），其内存所有权归调用者所有；调用者必须确保指针在ContextHolder对象的生命周期内有效。
