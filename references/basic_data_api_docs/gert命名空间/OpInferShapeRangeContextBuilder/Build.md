# Build

**页面ID:** atlasopapi_07_00623  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00623.html

---

#### 函数功能

根据之前的设置，构建InferShapeRangeContext，返回一个ContextHolder<InferShapeRangeContext>对象。

#### 函数原型

```
ContextHolder<InferShapeRangeContext> Build()
```

#### 参数说明

无

#### 返回值说明

返回一个ContextHolder<InferShapeRangeContext>对象。通过调用GetContext()方法可获取InferShapeRangeContext指针。

#### 约束说明

- 所有通过指针传入的参数，其内存所有权归调用者所有；调用者必须确保这些指针在ContextHolder对象的整个生命周期内有效。
- ContextHolder析构时会自动释放内部上下文资源。请勿手动释放GetContext()返回的指针。

#### 调用示例

```
#include "base/context_builder/op_infer_shape_range_context_builder.h"
OpInferShapeRangeContextBuilder ctx_builder;
gert::StorageShape xShapeMin{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
gert::StorageShape xShapeMax{{10, 10, 10, 10, 20}, {10, 10, 10, 10, 20}};
gert::StorageShape wShapeMin{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
gert::StorageShape wShapeMax{{10, 10, 10, 10, 20}, {10, 10, 10, 10, 20}};
gert::Shape yShapeMinNull{1, 1, 1, 1, 1};
gert::Shape yShapeMaxNull{10, 10, 10, 10, 20};
gert::StorageShape yShapeMin{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
gert::StorageShape yShapeMax{{10, 10, 10, 10, 20}, {10, 10, 10, 10, 20}};
StorageFormat format{FORMAT_NCDHW, FORMAT_RESERVED, {}};
gert::Tensor xTensorMin{xShapeMin, format, ge::DT_INT8};
gert::Tensor xTensorMax{xShapeMax, format, ge::DT_INT8};
gert::Range<gert::Tensor> xShapeRange(&xTensorMin, &xTensorMax);
gert::Tensor wTensorMin{wShapeMin, format, ge::DT_INT8};
gert::Tensor wTensorMax{wShapeMax, format, ge::DT_INT8};
gert::Range<gert::Tensor> wShapeRange(&wTensorMin, &wTensorMax);
gert::Range<gert::Shape> yShapeRange(&yShapeMinNull, &yShapeMaxNull);
auto holder = ctx_builder.IONum(2, 1)
                  .OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                  .OpType("DIY")
                  .OpName("diy_1")
                  .InputTensorsRange({&xShapeRange, &wShapeRange})
                  .Build();
auto ctx = holder.GetContext();
EXPECT_NE(ctx, nullptr);
const CompileTimeTensorDesc *info_input_0 = ctx->GetInputDesc(0);
EXPECT_NE(info_input_0, nullptr);
EXPECT_EQ(info_input_0->GetDataType(), ge::DT_INT8);
EXPECT_EQ(info_input_0->GetOriginFormat(), ge::FORMAT_NCDHW);
EXPECT_EQ(info_input_0->GetStorageFormat(), ge::FORMAT_RESERVED);
const CompileTimeTensorDesc *info_input_1 = ctx->GetInputDesc(1);
EXPECT_NE(info_input_1, nullptr);
EXPECT_EQ(info_input_1->GetDataType(), ge::DT_INT8);
EXPECT_EQ(info_input_1->GetOriginFormat(), ge::FORMAT_NCDHW);
EXPECT_EQ(info_input_1->GetStorageFormat(), ge::FORMAT_RESERVED);
const CompileTimeTensorDesc *info_output_0 = ctx->GetOutputDesc(0);
EXPECT_NE(info_output_0, nullptr);
EXPECT_EQ(info_output_0->GetDataType(), ge::DT_FLOAT16);
EXPECT_EQ(info_output_0->GetOriginFormat(), ge::FORMAT_NCDHW);
EXPECT_EQ(info_output_0->GetStorageFormat(), ge::FORMAT_RESERVED);
EXPECT_NE(ctx->GetInputShapeRange(0), nullptr);
EXPECT_EQ(*(ctx->GetInputShapeRange(0)->GetMin()), xShapeMin.GetOriginShape());
EXPECT_EQ(*(ctx->GetInputShapeRange(0)->GetMax()), xShapeMax.GetOriginShape());
EXPECT_NE(ctx->GetInputShapeRange(1), nullptr);
EXPECT_EQ(*(ctx->GetInputShapeRange(1)->GetMin()), wShapeMin.GetOriginShape());
EXPECT_EQ(*(ctx->GetInputShapeRange(1)->GetMax()), wShapeMax.GetOriginShape());
EXPECT_NE(ctx->GetOutputShapeRange(0), nullptr);
EXPECT_NE(ctx->GetOutputShapeRange(0)->GetMin(), nullptr);
EXPECT_NE(ctx->GetOutputShapeRange(0)->GetMax(), nullptr);
EXPECT_EQ(ctx->GetOutputShapeRange(0)->GetMin()->GetDimNum(), 0);
EXPECT_EQ(ctx->GetOutputShapeRange(0)->GetMax()->GetDimNum(), 0);
ctx->GetOutputShapeRange(0)->GetMin()->SetDimNum(5);
ctx->GetOutputShapeRange(0)->GetMax()->SetDimNum(5);
for (size_t i = 0; i < 5; i++) {
  ctx->GetOutputShapeRange(0)->GetMin()->SetDim(i, yShapeMin.GetOriginShape()[i]);
  ctx->GetOutputShapeRange(0)->GetMax()->SetDim(i, yShapeMax.GetOriginShape()[i]);
}
EXPECT_EQ(*(ctx->GetOutputShapeRange(0)->GetMin()), yShapeMin.GetOriginShape());
EXPECT_EQ(*(ctx->GetOutputShapeRange(0)->GetMax()), yShapeMax.GetOriginShape());
```
