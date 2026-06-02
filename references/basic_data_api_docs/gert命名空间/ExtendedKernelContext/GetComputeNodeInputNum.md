# GetComputeNodeInputNum

**页面ID:** atlasopapi_07_00078  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00078.html

---

#### 函数功能

获取算子的输入个数。

#### 函数原型

```
size_t GetComputeNodeInputNum() const
```

#### 参数说明

无。

#### 返回值说明

算子的输入个数。

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
