# OpExecuteFunc

**页面ID:** atlasopapi_07_00124  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00124.html

---

#### 函数功能

单个算子包含多kernel组合执行逻辑的场景下，算子可以通过该接口设置算子级的回调函数，回调函数内实现多kernel的下发。该功能为预留特性，暂不支持。

#### 函数原型

```
OpImplRegisterV2 &OpExecuteFunc(OpExecFunc op_execute_func)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| 注册的自定义OpExecuteFunc函数，类型为OpExecFunc。 OpExecFunc类型定义如下： ``` using OpExecFunc = UINT32 (*)(OpExecuteContext *); ``` |  |  |

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了算子级的回调函数。

#### 约束说明

无。
