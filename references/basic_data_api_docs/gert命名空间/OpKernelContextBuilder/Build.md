# Build

**页面ID:** atlasopapi_07_00630  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00630.html

---

#### 函数功能

根据之前的设置，构建KernelContext，返回一个ContextHolder<KernelContext>对象。

#### 函数原型

```
ContextHolder<KernelContext> Build()
```

#### 参数说明

无

#### 返回值说明

返回一个ContextHolder<KernelContext>对象。通过调用GetContext()方法可获取KernelContext指针。

#### 约束说明

- 所有通过指针传入的参数，其内存所有权归调用者所有；调用者必须确保这些指针在ContextHolder对象的整个生命周期内有效。
- ContextHolder析构时会自动释放内部上下文资源。请勿手动释放GetContext() 返回的指针。

#### 调用示例

```
#include "base/context_builder/op_kernel_run_context_builder.h"
OpKernelContextBuilder ctx_builder;
gert::StorageShape shape0 = {{10, 20}, {10, 20}};
auto holder = ctx_builder.OpType("Add")
                  .OpName("add_1")
                  .IONum(2, 1)
                  .InputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                  .InputTensorDesc(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                  .OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                  .Inputs({&shape0, &shape0})
                  .Outputs({&shape0})
                  .Build();
auto ctx = reinterpret_cast<KernelContext *>(holder.GetContext());
EXPECT_NE(ctx, nullptr);
auto ctx_compute_node_info = static_cast<const gert::ComputeNodeInfo *>(ctx->GetComputeNodeExtend());
EXPECT_NE(ctx_compute_node_info, nullptr);
EXPECT_EQ(std::string(ctx_compute_node_info->GetNodeType()), std::string("Add"));
EXPECT_EQ(std::string(ctx_compute_node_info->GetNodeName()), std::string("add_1"));
EXPECT_EQ(ctx_compute_node_info->GetIrInputsNum(), 2);
EXPECT_EQ(ctx_compute_node_info->GetIrOutputsNum(), 1);
EXPECT_EQ(ctx_compute_node_info->GetInputsNum(), 2);
EXPECT_EQ(ctx_compute_node_info->GetOutputsNum(), 1);
const CompileTimeTensorDesc *info_input_0 = ctx_compute_node_info->GetInputTdInfo(0);
EXPECT_NE(info_input_0, nullptr);
EXPECT_EQ(info_input_0->GetStorageFormat(), ge::FORMAT_FRACTAL_NZ);
EXPECT_EQ(info_input_0->GetOriginFormat(), ge::FORMAT_ND);
EXPECT_EQ(ctx->GetInput(0)->GetPointer<gert::StorageShape>(), &shape0);
EXPECT_EQ(ctx->GetInput(1)->GetPointer<gert::StorageShape>(), &shape0);
EXPECT_EQ(ctx->GetOutput(0)->GetPointer<gert::StorageShape>(), &shape0);
```
