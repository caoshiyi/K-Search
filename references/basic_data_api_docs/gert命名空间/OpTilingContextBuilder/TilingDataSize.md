# TilingDataSize

**页面ID:** atlasopapi_07_00638  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00638.html

---

#### 函数功能

设置算子TilingData的大小，设置该大小后，会申请相应大小的内存用于存储算子的TilingData。相较于TilingData，调用此接口时生成的TilingData指针所有权归属ContextHolder，调用者无需关注TilingData的生命周期。

注意该接口与TilingData互斥，如果同时调用TilingDataSize和TilingData接口，后调用的接口会覆盖前一次调用的结果，以最新的为准。

#### 函数原型

```
OpTilingContextBuilder &TilingDataSize(size_t tiling_data_size)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_data_size | 输出 | Tiling数据大小。 |

#### 返回值说明

TilingContextBuilder对象用于链式调用。

#### 约束说明

在调用Build方法之前，必须设置TilingData或TilingDataSize，否则构造出的TilingContext将包含未定义数据。
