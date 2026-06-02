# 简介

**页面ID:** atlasopapi_07_00550  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00550.html

---

OpImplRegisterV2是一个算子实现注册工具类，用于自定义算子开发。它支持以下功能：

- 注册算子的shape推导、数据类型推导、Range推导等实现。
- 注册算子的Tiling计算实现。
- 注册算子的属性信息。
- 指定Shape推导和Tiling计算过程中所依赖的输入索引。

在自定义算子开发过程中，用户基于OP_ADD注册算子原型时会间接使用到该类。

#### 需要包含的头文件

```
#include "register/op_impl_registry.h"
```

#### Public成员函数

```
explicit OpImplRegisterV2(const ge::char_t *op_type)
OpImplRegisterV2(OpImplRegisterV2 &&register_data) noexcept
OpImplRegisterV2(const OpImplRegisterV2 &register_data)
OpImplRegisterV2 &operator=(const OpImplRegisterV2 &) = delete
OpImplRegisterV2 &operator=(OpImplRegisterV2 &&) = delete
~OpImplRegisterV2()
OpImplRegisterV2 &InferShape(InferShapeKernelFunc infer_shape_func)
OpImplRegisterV2 &InferShapeRange(InferShapeRangeKernelFunc infer_shape_range_func)
OpImplRegisterV2 &InferDataType(InferDataTypeKernelFunc infer_datatype_func)
OpImplRegisterV2 &Tiling(TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, int64_t private_attr_val)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const std::vector<int64_t> &private_attr_val)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const ge::char_t *private_attr_val)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, ge::float32_t private_attr_val)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, bool private_attr_val)
OpImplRegisterV2 &PrivateAttr(const ge::char_t *private_attr, const std::vector<ge::float32_t> &private_attr_val)
template<typename T>
OpImplRegisterV2 &TilingParse(KernelFunc const tiling_parse_func)
template<typename T>
OpImplRegisterV2 &TilingParse(TilingParseFunc const tiling_parse_func)
OpImplRegisterV2 &InputsDataDependency(std::initializer_list<int32_t> inputs)
OpImplRegisterV2 &InferOutDataTypeSameWithFirstInput()
OpImplRegisterV2 &GenSimplifiedKey(GenSimplifiedKeyKernelFunc gen_simplifiedkey_func)
OpImplRegisterV2 &OpExecuteFunc(OpExecFunc op_execute_func)
OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs)
OpImplRegisterV2 &TilingInputsDataDependency(std::initializer_list<int32_t> inputs, std::initializer_list<TilingPlacement> placements)
OpImplRegisterV2 &HostInputs(std::initializer_list<int32_t> inputs)
OpImplRegisterV2 &OutputShapeDependOnCompute(std::initializer_list<int32_t> outputs)
```
