# GetRequiredInputDesc

**页面ID:** atlasopapi_07_00075  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00075.html

---

#### 函数功能

根据算子原型定义中的输入索引获取对应必选输入的tensor描述信息。

#### 函数原型

```
const CompileTimeTensorDesc *GetRequiredInputDesc(const size_t ir_index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 算子IR原型定义中的输入索引，从0开始计数。 |

#### 返回值说明

CompileTimeTensorDesc指针，index非法时，返回空指针。

关于CompileTimeTensorDesc的定义，请参见CompileTimeTensorDesc。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
// 假设某个算子的IR原型的第0个输入是普通输入，且实际有1个输入
auto optional_input_td = extend_context->GetRequiredInputDesc(0); // 拿到第0个输入的tensor描述
```
