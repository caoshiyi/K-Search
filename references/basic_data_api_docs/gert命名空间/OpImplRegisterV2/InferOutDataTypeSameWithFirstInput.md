# InferOutDataTypeSameWithFirstInput

**页面ID:** atlasopapi_07_00122  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00122.html

---

#### 函数功能

注册一种datatype推导规则，该规则将算子第一个输入的datatype作为所有输出的datatype。

#### 函数原型

```
OpImplRegisterV2 &InferOutDataTypeSameWithFirstInput()
```

#### 参数说明

无

#### 返回值说明

返回算子的OpImplRegisterV2对象，该对象新增注册了算子datatype推导规则。

#### 约束说明

- 注册此规则，可以不用再注册自定义推导规则。若同时注册了InferDataType和InferOutDataTypeByFirstInput，将使能最后注册的规则。
- 若算子无输入或第一个输入datatype为未定义（DT_UNDEFINED），推导将报错。
