# TensorTrait简介

**页面ID:** atlasascendc_api_07_00120  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00120.html

---

TensorTrait数据结构是描述Tensor相关信息的基础模板类，包含Tensor的数据类型、逻辑位置和Layout内存布局。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。

#### 需要包含的头文件

```
#include "kernel_operator_tensor_trait.h"
```

#### 原型定义

```
template <typename T, TPosition pos = TPosition::GM, typename LayoutType = Layout<Shape<>, Stride<>>>
struct TensorTrait {
    using LiteType = T;
    using LiteLayoutType = LayoutType;
    static constexpr const TPosition tPos = pos; // 该常量成员为后续功能扩展做预留
public:
    __aicore__ inline TensorTrait(const LayoutType& t = {});

    __aicore__ inline LayoutType& GetLayout();
    __aicore__ inline const LayoutType& GetLayout() const;

    __aicore__ inline void SetLayout(const LayoutType& t);

};
```

#### 模板参数

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 只支持如下基础数据类型：int4b_t、uint8_t、int8_t、int16_t、uint16_t、bfloat16_t、int32_t、uint32_t、int64_t、uint64_t、float、half 。 在TensorTrait结构体内部，使用using关键字定义了一个类型别名LiteType，与模板参数T类型一致。 通过TensorTrait定义的LocalTensor/GlobalTensor不包含ShapeInfo信息。 例如：LocalTensor<float>对应的不含ShapeInfo信息的Tensor为LocalTensor<TensorTrait<float>>。 |
| pos | 数据存放的逻辑位置，Tposition类型，默认为TPosition::GM。 |
| LayoutType | Layout数据类型，默认为空类型，即Layout<Shape<>, Stride<>>。 输入的数据类型LayoutType，需满足约束说明。 |

#### 成员函数

```
__aicore__ inline TensorTrait(const LayoutType& t = {})
__aicore__ inline LayoutType& GetLayout()
__aicore__ inline const LayoutType& GetLayout() const
__aicore__ inline void SetLayout(const LayoutType& t)
```

#### 相关接口

```
// TensorTrait结构构造方法
template <typename T, TPosition pos, typename LayoutType>
__aicore__ inline constexpr auto MakeTensorTrait(const LayoutType& t)

// is_tensorTrait原型定义
template <typename T> struct is_tensorTrait
```

#### 约束说明

- 同一接口不支持同时输入TensorTrait类型的GlobalTensor/LocalTensor和非TensorTrait类型的GlobalTensor/LocalTensor。
- 非TensorTrait类型和TensorTrait类型的GlobalTensor/LocalTensor相互之间不支持拷贝构造和赋值运算符。
- TensorTrait特性当前仅支持如下接口：

> **注意:** 

  - 和API配合使用时，当前暂不支持TensorTrait结构配置pos、LayoutType模板参数，需要使用构造函数构造TensorTrait，pos、LayoutType保持默认值即可。
  - DataCopy切片数据搬运接口需要ShapeInfo信息，不支持输入TensorTrait类型的GlobalTensor/LocalTensor。

**表2 **TensorTrait特性支持的接口列表

| 接口分类 | 接口名称 |
| --- | --- |
| 基础API>资源管理>TQue/TQueBind | AllocTensor、FreeTensor、EnQue、DeQue |
| 基础API>矢量计算>基础算术 | Exp、Ln、Abs、Reciprocal、Sqrt、Rsqrt、Relu、Add、Sub、Mul、Div、Max、Min、Adds、Muls、Maxs、Mins、VectorPadding、BilinearInterpolation、LeakyRelu |
| 基础API>矢量计算>逻辑计算 | And、Or |
| 基础API>矢量计算>复合计算 | CastDeq、AddRelu、AddDeqRelu、SubRelu、MulAddDst、FusedMulAdd、FusedMulAddRelu、AddReluCast、SubReluCast、MulCast |
| 基础API>数据搬运 | DataCopy、Copy |
| 基础API>矩阵计算 | InitConstValue、LoadData、LoadDataWithTranspose、SetAippFunctions、LoadImageToLocal、LoadUnzipIndex、LoadDataUnzip、LoadDataWithSparse、Mmad、MmadWithSparse、BroadCastVecToMM、Gemm、Fixpipe |
| 基础API>矢量计算>比较与选择 | Compare、GetCmpMask、SetCmpMask、Select、GatherMask |
| 基础API>矢量计算>类型转换 | Cast |
| 基础API>矢量计算>归约计算 | ReduceMax、BlockReduceMax、WholeReduceMax、ReduceMin、BlockReduceMin、WholeReduceMin、ReduceSum、BlockReduceSum、WholeReduceSum、RepeatReduceSum、PairReduceSum |
| 基础API>矢量计算>数据转换 | Transpose、TransDataTo5HD |
| 基础API>矢量计算>数据填充 | Brcb |
| 基础API>矢量计算>离散与聚合 | Gather、Gatherb、Scatter |
| 基础API>矢量计算>排序组合（ISASI） | ProposalConcat、ProposalExtract、RpSort16、MrgSort4、Sort32 |
