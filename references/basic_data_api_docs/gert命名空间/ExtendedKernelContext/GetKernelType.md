# GetKernelType

**页面ID:** atlasopapi_07_00085  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00085.html

---

#### 函数功能

获取本kernel的类型。

#### 函数原型

```
const ge::char_t *GetKernelType() const
```

#### 参数说明

无。

#### 返回值说明

本kernel的type。

#### 约束说明

无。

#### 调用示例

```
// 假设已存在KernelContext *context
auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
auto kernel_type = extend_context->GetKernelType();
```
