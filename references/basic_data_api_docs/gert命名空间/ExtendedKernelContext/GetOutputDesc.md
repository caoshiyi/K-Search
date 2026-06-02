# GetOutputDesc

**页面ID:** atlasopapi_07_00072  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00072.html

---

#### 函数功能

根据算子输出索引获取对应输出的tensor描述信息。这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
const CompileTimeTensorDesc *GetOutputDesc(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出索引，从0开始计数。 |

#### 返回值说明

输出TensorDesc的指针，当输入index非法时，返回空指针。

关于CompileTimeTensorDesc的定义，请参见CompileTimeTensorDesc。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
for (size_t idx = 0; idx < extend_context->GetComputeNodeInfo()->GetOutputsNum(); ++idx) {
  auto output_td = extend_context->GetOutputDesc(idx);
  ...
}
```
