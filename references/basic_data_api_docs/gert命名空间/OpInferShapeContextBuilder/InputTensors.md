# InputTensors

**页面ID:** atlasopapi_07_00616  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00616.html

---

#### 函数功能

设置输入Tensor指针。Shape推导时，可通过该Builder类构造的InferShapeContext获取相应的输入Tensor指针。对于数据依赖的算子，对应数据依赖的输入Tensor中的TensorData需要设置为正确的Host地址；对于非数据依赖算子，Tensor的TensorData需要设置为空指针。

#### 函数原型

```
OpInferShapeContextBuilder &InputTensors(const std::vector<gert::Tensor *> &inputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 输入Tensor指针数组，所有权归调用者管理，调用者需要保证输入指针生命周期长于Build产生的ContextHolder对象。 |

#### 返回值说明

OpInferShapeContextBuilder对象本身，用于链式调用。

#### 约束说明

在调用Build方法之前，必须调用该接口，否则构造出的InferShapeContext将包含未定义数据。
