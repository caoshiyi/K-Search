# InputTensorsRange

**页面ID:** atlasopapi_07_00622  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00622.html

---

#### 函数功能

设置输入Tensor的Range指针，用于在Shape Range推导时，可通过该Builder类构造的上下文InferShapeRangeContext获取相应的输入Tensor Range指针，即可以获得最大Shape的Tensor和最小Shape的Tensor。

#### 函数原型

```
OpInferShapeRangeContextBuilder &InputTensorsRange(const std::vector<gert::Range<gert::Tensor> *> &inputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | gert::Range<gert::Tensor> *类型的数组，存储各算子输入的Tensor Range指针，Tensor Range包含最大Shape的Tensor和最小Shape的Tensor。 |

#### 返回值说明

OpInferShapeRangeContextBuilder对象本身，用于链式调用。

#### 约束说明

- 在调用Build方法之前，必须调用该接口，否则构造出的InferShapeRangeContext将包含未定义数据。
- 通过指针传入的参数 (gert::Tensor *)，其内存所有权归调用者所有；调用者必须确保指针在ContextHolder对象的生命周期内有效。
