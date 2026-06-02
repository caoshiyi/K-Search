# GetKernelName

**页面ID:** atlasopapi_07_00084  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00084.html

---

#### 函数功能

获取本kernel的名称。

#### 函数原型

```
const ge::char_t *GetKernelName() const
```

#### 参数说明

无。

#### 返回值说明

本kernel的name。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto kernel_name = extend_context->GetKernelName();
```
