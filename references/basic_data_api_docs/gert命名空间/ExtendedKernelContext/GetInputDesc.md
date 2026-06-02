# GetInputDesc

**页面ID:** atlasopapi_07_00071  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00071.html

---

#### 函数功能

根据算子输入索引获取对应输入的tensor描述信息。这里的输入索引是指算子实例化后实际的索引，不是原型定义中的索引。

#### 函数原型

```
const CompileTimeTensorDesc *GetInputDesc(const size_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输入索引，从0开始计数。 |

#### 返回值说明

输入TensorDesc的指针，当输入index非法时，返回空指针。

关于CompileTimeTensorDesc的定义，请参见CompileTimeTensorDesc。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
for (size_t idx = 0; idx < extend_context->GetComputeNodeInputNum(); ++idx) {
  auto input_td = extend_context->GetInputDesc(idx);
  ...
}
```
