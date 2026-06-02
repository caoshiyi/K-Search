# BuildKernelRunContext

**页面ID:** atlasascendc_api_07_1013  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1013.html

---

#### 功能说明

构造KernelRunContext并返回KernelRunContextHolder的智能指针，该对象可通过GetContext接口获取KernelContext类型的对象。

#### 函数原型

```
std::shared_ptr<KernelRunContextHolder> BuildKernelRunContext()
```

#### 参数说明

无

#### 返回值说明

KernelRunContextHolder的共享指针，可通过GetContext<gert::KernelContext>()函数获取KernelContext对象。

#### 约束说明

无

#### 调用示例

```
auto kernelContextHolder = context_ascendc::ContextBuilder().Inputs().Outputs().BuildKernelRunContext();
gert::KernelContext* tilingParseContext = kernelContextHolder->GetContext<gert::KernelContext>();
```
