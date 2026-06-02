# InferShape

**页面ID:** atlasopapi_07_00115  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00115.html

---

#### 函数功能

注册算子的InferShape函数。

用户需要为算子编写一个InferShapeKernelFunc类型的函数，并使用该接口进行注册。

InferShapeKernelFunc类型定义如下：

```
using InferShapeKernelFunc = UINT32 (*)(InferShapeContext *);
```

InferShape函数的原型是确定的，其接受一个InferShapeContext类型作为输入，在此context上，可以获取到输入、输出的shape指针等内容（算子原型定义上的输入、输出、属性信息）。InferShape成功后，返回ge::GRAPH_SUCCESS，其他返回值被认为推导失败。推导失败后，执行过程结束退出。

#### 函数原型

```
OpImplRegisterV2 &InferShape(InferShapeKernelFunc infer_shape_func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| infer_shape_func | 输入 | 要注册的自定义InferShape函数，类型为InferShapeKernelFunc。 |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了InferShape函数infer_shape_func。

#### 约束说明

无。
