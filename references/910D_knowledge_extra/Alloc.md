# Alloc<a name="ZH-CN_TOPIC_0000002523304742"></a>

## 产品支持情况<a name="section73648168211"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

根据用户指定的逻辑位置、数据类型、数据长度返回对应的LocalTensor对象。

## 函数原型<a name="section620mcpsimp"></a>

-   原型1：tileSize为模板参数

    ```
    // 当tileSize为常量时，建议使用此接口，以获得更优的性能
    template <class DataType, uint32_t tileSize> LocalTensor<DataType> __aicore__ inline Alloc()
    template <TPosition pos, class DataType, uint32_t tileSize> __aicore__ inline LocalTensor<DataType> Alloc()
    ```

-   原型2：tileSize为接口入参

    ```
    // 当tileSize为动态参数时使用此接口
    template <class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize)
    template <TPosition pos, class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize)
    ```

-   原型3：使用TensorTrait时使用此接口

    ```
    template <class DataType> LocalTensor<DataType> __aicore__ inline Alloc()
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  原型1和原型2模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="17.43%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.57%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row18835145716587"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="p18689719202918"><a name="p18689719202918"></a><a name="p18689719202918"></a><a href="TPosition.md">TPosition</a>位置，需要符合<a href="LocalMemAllocator简介.md">LocalMemAllocator</a>中指定的Hardware物理位置（静态Tensor编程场景下，此参数可以省略）。</p>
</td>
</tr>
<tr id="row199212475441"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="p352033512453"><a name="p352033512453"></a><a name="p352033512453"></a>DataType</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="p4921114784410"><a name="p4921114784410"></a><a name="p4921114784410"></a>LocalTensor的数据类型，只支持基础数据类型，不支持TensorTrait类型。</p>
</td>
</tr>
<tr id="row187531218114513"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="p18753151854519"><a name="p18753151854519"></a><a name="p18753151854519"></a>tileSize</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="p16753518194519"><a name="p16753518194519"></a><a name="p16753518194519"></a>LocalTensor的元素个数，其数量不应超过当前物理位置剩余的内存空间。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  原型2参数说明

<a name="table10918948849"></a>
<table><thead align="left"><tr id="row159181048845"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.2.4.1.1"><p id="p89187484413"><a name="p89187484413"></a><a name="p89187484413"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.4.1.2"><p id="p179183480411"><a name="p179183480411"></a><a name="p179183480411"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.2.4.1.3"><p id="p1291894812414"><a name="p1291894812414"></a><a name="p1291894812414"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14918104812413"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.2.4.1.1 "><p id="p391811481242"><a name="p391811481242"></a><a name="p391811481242"></a>tileSize</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.4.1.2 "><p id="p1591812481240"><a name="p1591812481240"></a><a name="p1591812481240"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.2.4.1.3 "><p id="p191810481240"><a name="p191810481240"></a><a name="p191810481240"></a>LocalTensor的元素个数，其数量不应超过当前物理位置剩余的内存空间。</p>
<p id="p3875183944919"><a name="p3875183944919"></a><a name="p3875183944919"></a><span>剩余的内存空间可以通过物理内存最大值与当前可用内存地址（</span><a href="GetCurAddr-92.md">GetCurAddr</a><span>返回值）的差值来计算。</span></p>
</td>
</tr>
</tbody>
</table>

**表 3**  原型3模板参数说明

<a name="table645894610463"></a>
<table><thead align="left"><tr id="row12458246134618"><th class="cellrowborder" valign="top" width="27.839999999999996%" id="mcps1.2.3.1.1"><p id="p045817465464"><a name="p045817465464"></a><a name="p045817465464"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="72.16%" id="mcps1.2.3.1.2"><p id="p15458446154614"><a name="p15458446154614"></a><a name="p15458446154614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row945854624614"><td class="cellrowborder" valign="top" width="27.839999999999996%" headers="mcps1.2.3.1.1 "><p id="p8458184610468"><a name="p8458184610468"></a><a name="p8458184610468"></a>TensorTraitType</p>
</td>
<td class="cellrowborder" valign="top" width="72.16%" headers="mcps1.2.3.1.2 "><p id="p10458194684613"><a name="p10458194684613"></a><a name="p10458194684613"></a>只支持传入<a href="TensorTrait.md">TensorTrait</a>类型，TensorTrait的数据类型/逻辑位置/Shape大小需要匹配LocalMemAllocator中指定的物理位置及其剩余空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

根据用户输入构造的LocalTensor对象。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="zh-cn_topic_0000002078486173_zh-cn_topic_0000001576727153_zh-cn_topic_0000001389787297_section320753512363"></a>

```
template <uint32_t v>
using UIntImm = Std::integral_constant<uint32_t, v>;
...
AscendC::LocalMemAllocator allocator;
// 原型1：float类型，Tensor中有1024个元素，用户可以指定逻辑位置(或者不指定，由Alloc函数根据物理位置给出默认值，不影响功能)
auto tensor1 = allocator.Alloc<AscendC::TPosition::VECIN, float, 1024>();
auto tensor1 = allocator.Alloc<float, 1024>();

// 原型2：float类型，Tensor中有tileLength个元素，用户可以指定逻辑位置(或者不指定，由Alloc函数根据物理位置给出默认值，不影响功能)
auto tensor1 = allocator.Alloc<AscendC::TPosition::VECIN, float>(tileLength);

// 原型3：用户指定逻辑位置VECIN，数据类型为float，Tensor中元素个数为16*16*16
auto shape = AscendC::MakeShape(UIntImm<16>{}, UIntImm<16>{}, UIntImm<16>{});
auto stride = AscendC::MakeStride(UIntImm<0>{}, UIntImm<0>{}, UIntImm<0>{});
auto layoutMake = AscendC::MakeLayout(shape, stride);
auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);
auto tensor3 = allocator.Alloc<decltype(tensorTraitMake)>();
```

