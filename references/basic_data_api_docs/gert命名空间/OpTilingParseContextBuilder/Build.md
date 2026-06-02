# Build

**页面ID:** atlasopapi_07_00651  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00651.html

---

#### 函数功能

根据之前的设置，构建TilingParseContext，返回一个ContextHolder<TilingParseContext>对象。

#### 函数原型

```
ContextHolder<TilingParseContext> Build()
```

#### 参数说明

无

#### 返回值说明

返回一个ContextHolder<TilingParseContext>对象，通过其GetContext()方法可获取TilingParseContext指针。

#### 约束说明

- 所有通过指针传入的参数，其内存所有权归调用者所有；调用者必须确保这些指针在ContextHolder对象的整个生命周期内有效。
- ContextHolder析构时会自动释放内部上下文资源。请勿手动释放GetContext()返回的指针。

#### 调用示例

```
#include "base/context_builder/op_tiling_parse_context_builder.h"
const char* json_data = R"({"block_dim": 16, "stream_num": 1})";
uint8_t tmp_platform_info[] = {1, 2, 3, 4, 5, 6, 7}; // fake数据
uint8_t tmp_compile_info[] = {1, 2, 3, 4, 5, 6, 7}; // fake数据
OpTilingParseContextBuilder ctx_builder;
auto holder = ctx_builder.OpName("tmp")
                  .OpType("DIY")
                  .IONum(4, 1)
                  .InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                  .InputTensorDesc(1, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                  .InputTensorDesc(2, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                  .InputTensorDesc(3, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                  .OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_NCDHW, ge::FORMAT_RESERVED)
                  .CompiledJson(json_data)
                  .CompiledInfo(tmp_compile_info)
                  .PlatformInfo(tmp_platform_info)
                  .Build();
auto ctx = holder.GetContext();
EXPECT_NE(ctx, nullptr);
auto ctx_compute_node_info = ctx->GetComputeNodeInfo();
EXPECT_NE(ctx_compute_node_info, nullptr);
EXPECT_EQ(ctx->GetInputDesc(0)->GetOriginFormat(), ge::FORMAT_NCDHW);
EXPECT_EQ(ctx->GetInputDesc(0)->GetStorageFormat(), ge::FORMAT_RESERVED);
EXPECT_EQ((void *) ctx->GetPlatformInfo(), (void *) tmp_platform_info);
EXPECT_EQ((void *) ctx->GetPlatformInfo(), (void *) tmp_platform_info);
EXPECT_EQ(std::string(ctx->GetCompiledJson()), std::string(json_data.c_str()));
EXPECT_EQ(ctx->GetCompiledInfo<uint8_t>(), tmp_compile_info);
```
