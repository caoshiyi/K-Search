# GetDynamicInputDesc

**页面ID:** atlasopapi_07_00074  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00074.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应动态输入的tensor描述信息。

#### 函数原型

```
const CompileTimeTensorDesc *GetDynamicInputDesc(const size_t ir_index, const size_t relative_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输入索引，从0开始计数。 |
| relative_index | 输入 | 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]。 |

#### 返回值说明

CompileTimeTensorDesc指针，index或relative_index非法时，返回空指针。

关于CompileTimeTensorDesc的定义，请参见CompileTimeTensorDesc。

#### 约束说明

无。
