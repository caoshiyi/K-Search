# TensorTrait简介<a name="ZH-CN_TOPIC_0000002554423777"></a>

TensorTrait数据结构是描述Tensor相关信息的基础模板类，包含Tensor的数据类型、逻辑位置和Layout内存布局。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。

## 需要包含的头文件<a name="zh-cn_topic_0000002213064918_section78885814919"></a>

```
#include "kernel_operator_tensor_trait.h"
```

## 原型定义<a name="section10580930144614"></a>

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

## 模板参数<a name="section116801320102618"></a>

**表 1**  模板参数说明

<a name="table13588175515344"></a>
<table><thead align="left"><tr id="row1160915519346"><th class="cellrowborder" valign="top" width="21.8%" id="mcps1.2.3.1.1"><p id="p9609105553412"><a name="p9609105553412"></a><a name="p9609105553412"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="78.2%" id="mcps1.2.3.1.2"><p id="p156091955143419"><a name="p156091955143419"></a><a name="p156091955143419"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row260915573419"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p2060925573411"><a name="p2060925573411"></a><a name="p2060925573411"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p1466165112529"><a name="p1466165112529"></a><a name="p1466165112529"></a>只支持如下基础数据类型：int4b_t、uint8_t、int8_t、int16_t、uint16_t、bfloat16_t、int32_t、uint32_t、int64_t、uint64_t、float、half 。</p>
<p id="p9673541185614"><a name="p9673541185614"></a><a name="p9673541185614"></a><span>在TensorTrait结构体内部，使用</span>using<span>关键字定义了一个类型别名</span>LiteType<span>，与模板参数T类型一致</span>。</p>
<p id="p17381434135715"><a name="p17381434135715"></a><a name="p17381434135715"></a><span>通过TensorTrait定义的</span>LocalTensor/GlobalTensor不包含<a href="ShapeInfo.md">ShapeInfo</a>信息。</p>
<p id="p18609195511344"><a name="p18609195511344"></a><a name="p18609195511344"></a>例如：LocalTensor&lt;float&gt;对应的不含ShapeInfo信息的Tensor为LocalTensor&lt;TensorTrait&lt;float&gt;&gt;。</p>
</td>
</tr>
<tr id="row1545073919457"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p1745103924512"><a name="p1745103924512"></a><a name="p1745103924512"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p1401735165413"><a name="p1401735165413"></a><a name="p1401735165413"></a>数据存放的逻辑位置，<a href="TPosition.md">Tposition</a>类型，默认为TPosition::GM。</p>
</td>
</tr>
<tr id="row1076563718543"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p167661637135419"><a name="p167661637135419"></a><a name="p167661637135419"></a>LayoutType</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p1776623725413"><a name="p1776623725413"></a><a name="p1776623725413"></a><a href="Layout.md">Layout</a>数据类型，默认为空类型，即Layout&lt;Shape&lt;&gt;, Stride&lt;&gt;&gt;。</p>
<p id="p183241815132114"><a name="p183241815132114"></a><a name="p183241815132114"></a>输入的数据类型LayoutType，需满足<a href="Layout构造函数.md#zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section19165124931511">约束说明</a>。</p>
</td>
</tr>
</tbody>
</table>

## 成员函数<a name="zh-cn_topic_0000002213064918_section1173524710"></a>

```
__aicore__ inline TensorTrait(const LayoutType& t = {})
__aicore__ inline LayoutType& GetLayout()
__aicore__ inline const LayoutType& GetLayout() const
__aicore__ inline void SetLayout(const LayoutType& t)
```

## 相关接口<a name="section104554349817"></a>

```
// TensorTrait结构构造方法
template <typename T, TPosition pos, typename LayoutType>
__aicore__ inline constexpr auto MakeTensorTrait(const LayoutType& t)

// is_tensorTrait原型定义
template <typename T> struct is_tensorTrait
```

## 约束说明<a name="section1253618294449"></a>

-   同一接口不支持同时输入TensorTrait类型的GlobalTensor/LocalTensor和非TensorTrait类型的GlobalTensor/LocalTensor。
-   非TensorTrait类型和TensorTrait类型的GlobalTensor/LocalTensor相互之间不支持拷贝构造和赋值运算符。
-   TensorTrait特性当前仅支持如下接口：

    > **说明：** 
    >-   和API配合使用时，当前暂不支持TensorTrait结构配置pos、LayoutType模板参数，需要使用构造函数构造TensorTrait，pos、LayoutType保持默认值即可。
    >-   DataCopy切片数据搬运接口需要ShapeInfo信息，不支持输入TensorTrait类型的GlobalTensor/LocalTensor。

    **表 2**  TensorTrait特性支持的接口列表

    <a name="table5536122919441"></a>
    <table><thead align="left"><tr id="row9536132974410"><th class="cellrowborder" valign="top" width="36.69%" id="mcps1.2.3.1.1"><p id="p053720299442"><a name="p053720299442"></a><a name="p053720299442"></a>接口分类</p>
    </th>
    <th class="cellrowborder" valign="top" width="63.31%" id="mcps1.2.3.1.2"><p id="p15537112974412"><a name="p15537112974412"></a><a name="p15537112974412"></a>接口名称</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row2053714299447"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p145371529164420"><a name="p145371529164420"></a><a name="p145371529164420"></a>基础API&gt;资源管理&gt;TQue/TQueBind</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p65371629164414"><a name="p65371629164414"></a><a name="p65371629164414"></a>AllocTensor、FreeTensor、EnQue、DeQue</p>
    </td>
    </tr>
    <tr id="row10537182994411"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p10537152994413"><a name="p10537152994413"></a><a name="p10537152994413"></a>基础API&gt;矢量计算&gt;基础算术</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p1353752917440"><a name="p1353752917440"></a><a name="p1353752917440"></a>Exp、Ln、Abs、Reciprocal、Sqrt、Rsqrt、Relu、Add、Sub、Mul、Div、Max、Min、Adds、Muls、Maxs、Mins、VectorPadding、BilinearInterpolation、Prelu、Mull、LeakyRelu</p>
    </td>
    </tr>
    <tr id="row753710292448"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p953782918440"><a name="p953782918440"></a><a name="p953782918440"></a>基础API&gt;矢量计算&gt;逻辑计算</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p1453712918442"><a name="p1453712918442"></a><a name="p1453712918442"></a>And、Or<span id="ph14188188114718"><a name="ph14188188114718"></a><a name="ph14188188114718"></a>、ShiftRight、ShiftLeft</span></p>
    </td>
    </tr>
    <tr id="row1353742918445"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p1653742954413"><a name="p1653742954413"></a><a name="p1653742954413"></a>基础API&gt;矢量计算&gt;复合计算</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p1453742917447"><a name="p1453742917447"></a><a name="p1453742917447"></a>CastDequant、AddRelu、AddDeqRelu、SubRelu、MulAddDst、FusedMulAdd、MulAddRelu、AddReluCast、ExpSub、AbsSub、SubReluCast、MulCast</p>
    </td>
    </tr>
    <tr id="row3537129184413"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p15537329144411"><a name="p15537329144411"></a><a name="p15537329144411"></a>基础API&gt;数据搬运</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p753742915441"><a name="p753742915441"></a><a name="p753742915441"></a>DataCopy、Copy</p>
    </td>
    </tr>
    <tr id="row1153772954413"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p35371729114419"><a name="p35371729114419"></a><a name="p35371729114419"></a>基础API&gt;矩阵计算</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p15538192954414"><a name="p15538192954414"></a><a name="p15538192954414"></a>Fill、LoadData、LoadDataWithTranspose、SetAippFunctions、LoadImageToLocal、LoadUnzipIndex、LoadDataUnzip、LoadDataWithSparse、Mmad、MmadWithSparse、BroadCastVecToMM、Gemm、Fixpipe</p>
    </td>
    </tr>
    <tr id="row61987635612"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p519919617562"><a name="p519919617562"></a><a name="p519919617562"></a>基础API&gt;矢量计算&gt;比较与选择</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p171999635617"><a name="p171999635617"></a><a name="p171999635617"></a>Compare、GetCmpMask、SetCmpMask、Select、GatherMask</p>
    </td>
    </tr>
    <tr id="row17369134075319"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p19751786543"><a name="p19751786543"></a><a name="p19751786543"></a>基础API&gt;矢量计算&gt;类型转换</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p13370194016534"><a name="p13370194016534"></a><a name="p13370194016534"></a>Cast、Truncate</p>
    </td>
    </tr>
    <tr id="row2480185119548"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p1748015518541"><a name="p1748015518541"></a><a name="p1748015518541"></a>基础API&gt;矢量计算&gt;归约计算</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p948010513543"><a name="p948010513543"></a><a name="p948010513543"></a>ReduceMax、BlockReduceMax、WholeReduceMax、ReduceMin、BlockReduceMin、WholeReduceMin、ReduceSum、BlockReduceSum、WholeReduceSum、RepeatReduceSum、PairReduceSum</p>
    </td>
    </tr>
    <tr id="row19583512577"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p45841411577"><a name="p45841411577"></a><a name="p45841411577"></a>基础API&gt;矢量计算&gt;数据转换</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p135844110574"><a name="p135844110574"></a><a name="p135844110574"></a>Transpose、TransDataTo5HD</p>
    </td>
    </tr>
    <tr id="row5272114215616"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p1272134245615"><a name="p1272134245615"></a><a name="p1272134245615"></a>基础API&gt;矢量计算&gt;数据填充</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p122721642185615"><a name="p122721642185615"></a><a name="p122721642185615"></a>Brcb、Duplicate（仅支持不带scalar参数的接口）</p>
    </td>
    </tr>
    <tr id="row2866124135816"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p786712415810"><a name="p786712415810"></a><a name="p786712415810"></a>基础API&gt;矢量计算&gt;离散与聚合</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p286711411582"><a name="p286711411582"></a><a name="p286711411582"></a>Gather、Gatherb、Scatter</p>
    </td>
    </tr>
    <tr id="row673710289580"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p773717281588"><a name="p773717281588"></a><a name="p773717281588"></a>基础API&gt;矢量计算&gt;数据重排（ISASI）</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p1473732895810"><a name="p1473732895810"></a><a name="p1473732895810"></a>Interleave、DeInterleave</p>
    </td>
    </tr>
    <tr id="row49061051165820"><td class="cellrowborder" valign="top" width="36.69%" headers="mcps1.2.3.1.1 "><p id="p490625113583"><a name="p490625113583"></a><a name="p490625113583"></a>基础API&gt;矢量计算&gt;排序组合（ISASI）</p>
    </td>
    <td class="cellrowborder" valign="top" width="63.31%" headers="mcps1.2.3.1.2 "><p id="p490655195819"><a name="p490655195819"></a><a name="p490655195819"></a>ProposalConcat、ProposalExtract、RpSort16、MrgSort4、Sort32</p>
    </td>
    </tr>
    </tbody>
    </table>

