# GetOptionalInputDesc

**页面ID:** atlasopapi_07_00073  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00073.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应可选输入的tensor描述信息。

#### 函数原型

```
const CompileTimeTensorDesc *GetOptionalInputDesc(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输入索引，从0开始计数。 |

#### 返回值说明

输入TensorDesc的指针，当输入index非法时，或该INPUT没有实例化时，返回空指针。

关于CompileTimeTensorDesc的定义，请参见CompileTimeTensorDesc。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
// case 1假设某个算子的IR原型的第0个输入是可选输入，且实际有1个输入
auto optional_input_td = extend_context->GetOptionalInputDesc(0); // 拿到第0个可选输入的tensor描述
// case 2假设某个算子的IR原型的第0个输入是可选输入，且实际没有输入
auto optional_input_td = extend_context->GetOptionalInputDesc(0); // nullptr
```
