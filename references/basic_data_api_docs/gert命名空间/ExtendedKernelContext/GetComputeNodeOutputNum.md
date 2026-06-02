# GetComputeNodeOutputNum

**页面ID:** atlasopapi_07_00079  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00079.html

---

#### 函数功能

获取算子的输出个数。

#### 函数原型

```
size_t GetComputeNodeOutputNum() const
```

#### 参数说明

无。

#### 返回值说明

算子的输出个数。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
for (size_t idx = 0; idx < extend_context->GetComputeNodeOutputNum(); ++idx) {
  auto input_td = extend_context->GetOutputDesc(idx);
  ...
}
```
