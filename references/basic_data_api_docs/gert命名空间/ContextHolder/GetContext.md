# GetContext

**页面ID:** atlasopapi_07_00598  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00598.html

---

#### 函数功能

根据指定类型获取对应的Context指针。

#### 函数原型

```
template<typename ContextTypeT>
ContextTypeT *GetContext()
```

#### 参数说明

ContextTypeT是一个泛型，包括以下类型的Context：

- InferDataTypeContext；
- InferShapeContext；
- InferShapeRangeContext；
- KernelContext；
- TilingContext；
- TilingParseContext。

#### 返回值说明

返回指向Context的指针，ContextBuilder、ContextHolder以及Context之间的关系如下表所示：

**表1 **ContextBuilder、ContextHolder以及Context之间的关系

| ContextBuilder | Holder | Context |
| --- | --- | --- |
| OpInferDataTypeContextBuilder | ContextHolder<InferDataTypeContext> | InferDataTypeContext |
| OpInferShapeContextBuilder | ContextHolder<InferShapeContext> | InferShapeContext |
| OpInferShapeRangeContextBuilder | ContextHolder<InferShapeRangeContext> | InferShapeRangeContext |
| OpKernelContextBuilder | ContextHolder<KernelContext> | KernelContext |
| OpTilingContextBuilder | ContextHolder<TilingContext> | TilingContext |
| OpTilingParseContextBuilder | ContextHolder<TilingParseContext> | TilingParseContext |

#### 约束说明

无

#### 调用示例

```
OpInferDataTypeContextBuilder ctx_builder;
ge::DataType dtype0 = ge::DT_FLOAT;
ge::DataType dtype1 = ge::DT_FLOAT16;
ge::DataType dtype2 = ge::DT_FLOAT;
ge::DataType dtype3 = ge::DT_FLOAT16;
ge::DataType dtype4 = ge::DT_FLOAT16;
std::vector<ge::DataType> input_dtype_ref = {dtype0, dtype1, dtype2, dtype3};
std::vector<ge::DataType *> output_dtype_ref = {&dtype4};
auto holder = ctx_builder.OpType("Concat")
                  .OpName("concat_1")
                  .IOInstanceNum({4}, {1})
                  .InputTensorDesc(0, dtype0, ge::FORMAT_ND, ge::FORMAT_ND)
                  .InputTensorDesc(1, dtype1, ge::FORMAT_ND, ge::FORMAT_ND)
                  .InputTensorDesc(2, dtype2, ge::FORMAT_ND, ge::FORMAT_ND)
                  .InputTensorDesc(3, dtype3, ge::FORMAT_ND, ge::FORMAT_ND)
                  .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND)
                  .Build();
auto ctx = holder.GetContext();
EXPECT_NE(ctx, nullptr);
```
