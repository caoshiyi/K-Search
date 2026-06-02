# Inputs

**页面ID:** atlasopapi_07_00629  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00629.html

---

#### 函数功能

设置Context的values的输入指针，values承载的类型为void*的指针数组。

#### 函数原型

```
OpKernelContextBuilder &Inputs(std::vector<void *> inputs)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| inputs | 输入 | 输入指针数组，所有权归调用者管理，调用者需要保证输入指针生命周期长于Build产生的ContextHolder对象。 |

#### 返回值说明

OpKernelContextBuilder对象本身，用于链式调用。

#### 约束说明

在调用Build方法之前，必须调用该接口，否则构造出的KernelContext将包含未定义数据。
