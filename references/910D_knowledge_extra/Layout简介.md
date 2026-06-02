# Layout简介<a name="ZH-CN_TOPIC_0000002523303724"></a>

Layout<Shape, Stride\>数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。

Layout包含两个核心组成部分：

-   Shape：定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。
-   Stride：定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。

例如，一个二维矩阵的Shape为\(4, 2\)，Stride为\(4, 1\)，表示：

-   矩阵有4行、2列。
-   列方向上的步长为1，即每行中相邻元素在内存中的间隔为1个元素；行方向上的步长为4，即相邻行的起始地址间隔为4个元素。

[表1](#table12525201245615)中给出了一维内存地址空间视图，[表2](#table17519406546)中给出了该二维矩阵的逻辑视图。

**表 1** **线性地址视图**

<a name="table12525201245615"></a>
<table><thead align="left"><tr id="row20525012185616"><th class="cellrowborder" valign="top" width="8.080808080808083%" id="mcps1.2.13.1.1"><p id="p101677148592"><a name="p101677148592"></a><a name="p101677148592"></a>地址</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.2"><p id="p2983172525618"><a name="p2983172525618"></a><a name="p2983172525618"></a>0</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.3"><p id="p9983142535617"><a name="p9983142535617"></a><a name="p9983142535617"></a>1</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.4"><p id="p29830252564"><a name="p29830252564"></a><a name="p29830252564"></a>2、3</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.5"><p id="p12983162525611"><a name="p12983162525611"></a><a name="p12983162525611"></a>4</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.6"><p id="p199831425105617"><a name="p199831425105617"></a><a name="p199831425105617"></a>5</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.7"><p id="p998312555616"><a name="p998312555616"></a><a name="p998312555616"></a>6、7</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.8"><p id="p159832025135614"><a name="p159832025135614"></a><a name="p159832025135614"></a>8</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.9"><p id="p6983102585615"><a name="p6983102585615"></a><a name="p6983102585615"></a>9</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.10"><p id="p29832025165615"><a name="p29832025165615"></a><a name="p29832025165615"></a>10、11</p>
</th>
<th class="cellrowborder" valign="top" width="8.500850085008503%" id="mcps1.2.13.1.11"><p id="p169831525195612"><a name="p169831525195612"></a><a name="p169831525195612"></a>12</p>
</th>
<th class="cellrowborder" valign="top" width="6.910691069106911%" id="mcps1.2.13.1.12"><p id="p134212147586"><a name="p134212147586"></a><a name="p134212147586"></a>13</p>
</th>
</tr>
</thead>
<tbody><tr id="row12525141216566"><td class="cellrowborder" valign="top" width="8.080808080808083%" headers="mcps1.2.13.1.1 "><p id="p01671214135912"><a name="p01671214135912"></a><a name="p01671214135912"></a>元素</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.2 "><p id="p498442515619"><a name="p498442515619"></a><a name="p498442515619"></a>a00</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.3 "><p id="p69841025115617"><a name="p69841025115617"></a><a name="p69841025115617"></a>a01</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.4 "><p id="p1498472517567"><a name="p1498472517567"></a><a name="p1498472517567"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.5 "><p id="p1098452595612"><a name="p1098452595612"></a><a name="p1098452595612"></a>a10</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.6 "><p id="p1098412510562"><a name="p1098412510562"></a><a name="p1098412510562"></a>a11</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.7 "><p id="p1984132565617"><a name="p1984132565617"></a><a name="p1984132565617"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.8 "><p id="p49841025105613"><a name="p49841025105613"></a><a name="p49841025105613"></a>a20</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.9 "><p id="p498442512566"><a name="p498442512566"></a><a name="p498442512566"></a>a21</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.10 "><p id="p1198415250563"><a name="p1198415250563"></a><a name="p1198415250563"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.500850085008503%" headers="mcps1.2.13.1.11 "><p id="p3984725145619"><a name="p3984725145619"></a><a name="p3984725145619"></a>a30</p>
</td>
<td class="cellrowborder" valign="top" width="6.910691069106911%" headers="mcps1.2.13.1.12 "><p id="p1642914105818"><a name="p1642914105818"></a><a name="p1642914105818"></a>a31</p>
</td>
</tr>
</tbody>
</table>

**表 2** **矩阵逻辑视图**

<a name="table17519406546"></a>
<table><thead align="left"><tr id="row375940195412"><th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.1"><p id="p1337044595410"><a name="p1337044595410"></a><a name="p1337044595410"></a>索引</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.2"><p id="p4370194535410"><a name="p4370194535410"></a><a name="p4370194535410"></a>列 0</p>
</th>
<th class="cellrowborder" valign="top" width="33.33333333333333%" id="mcps1.2.4.1.3"><p id="p16370154513542"><a name="p16370154513542"></a><a name="p16370154513542"></a>列 1</p>
</th>
</tr>
</thead>
<tbody><tr id="row3761740175413"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p537084575410"><a name="p537084575410"></a><a name="p537084575410"></a>行 0</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p1370134535418"><a name="p1370134535418"></a><a name="p1370134535418"></a>a00 (地址 0)</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p237024517543"><a name="p237024517543"></a><a name="p237024517543"></a>a01 (地址 1)</p>
</td>
</tr>
<tr id="row676104005417"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p2370645145412"><a name="p2370645145412"></a><a name="p2370645145412"></a>行 1</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p53701345165410"><a name="p53701345165410"></a><a name="p53701345165410"></a>a10 (地址 4)</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p1937034518545"><a name="p1937034518545"></a><a name="p1937034518545"></a>a11 (地址 5)</p>
</td>
</tr>
<tr id="row207654015545"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p23701045135412"><a name="p23701045135412"></a><a name="p23701045135412"></a>行 2</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p137019452548"><a name="p137019452548"></a><a name="p137019452548"></a>a20 (地址 8)</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p4370845165416"><a name="p4370845165416"></a><a name="p4370845165416"></a>a21 (地址 9)</p>
</td>
</tr>
<tr id="row187694055418"><td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.1 "><p id="p183701345125411"><a name="p183701345125411"></a><a name="p183701345125411"></a>行 3</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.2 "><p id="p0370945105417"><a name="p0370945105417"></a><a name="p0370945105417"></a>a30 (地址 12)</p>
</td>
<td class="cellrowborder" valign="top" width="33.33333333333333%" headers="mcps1.2.4.1.3 "><p id="p9370145125416"><a name="p9370145125416"></a><a name="p9370145125416"></a>a31 (地址 13)</p>
</td>
</tr>
</tbody>
</table>

## 需要包含的头文件<a name="zh-cn_topic_0000002213064918_section78885814919"></a>

```
#include "kernel_operator_layout.h"
```

## 原型定义<a name="section10580930144614"></a>

```
template <typename ShapeType, typename StrideType>
struct Layout : private Std::tuple<ShapeType, StrideType> {
    __aicore__ inline constexpr Layout(const ShapeType& shape  = {}, const StrideType& stride = {}) : Std::tuple<ShapeType, StrideType>(shape, stride) {}

    __aicore__ inline constexpr decltype(auto) layout() {}
    __aicore__ inline constexpr decltype(auto) layout() const {}
    
    __aicore__ inline constexpr decltype(auto) GetShape() {}   
    __aicore__ inline constexpr decltype(auto) GetShape() const {}
    
    __aicore__ inline constexpr decltype(auto) GetStride() {}    
    __aicore__ inline constexpr decltype(auto) GetStride() const {}
    
    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType& coord) const {}
}
```

## 模板参数<a name="section116801320102618"></a>

**表 3**  模板参数说明

<a name="table13588175515344"></a>
<table><thead align="left"><tr id="row1160915519346"><th class="cellrowborder" valign="top" width="21.8%" id="mcps1.2.3.1.1"><p id="p9609105553412"><a name="p9609105553412"></a><a name="p9609105553412"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="78.2%" id="mcps1.2.3.1.2"><p id="p156091955143419"><a name="p156091955143419"></a><a name="p156091955143419"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row260915573419"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p2060925573411"><a name="p2060925573411"></a><a name="p2060925573411"></a>ShapeType</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p823866165711"><a name="p823866165711"></a><a name="p823866165711"></a><span id="ph184621011705"><a name="ph184621011705"></a><a name="ph184621011705"></a><a href="容器函数.md">Std::tuple</a>结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。</span></p>
</td>
</tr>
<tr id="row1545073919457"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p1745103924512"><a name="p1745103924512"></a><a name="p1745103924512"></a>StrideType</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p64517398452"><a name="p64517398452"></a><a name="p64517398452"></a><span id="ph292255305"><a name="ph292255305"></a><a name="ph292255305"></a><a href="容器函数.md">Std::tuple</a>结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。</span></p>
</td>
</tr>
</tbody>
</table>

## 成员函数<a name="zh-cn_topic_0000002213064918_section1173524710"></a>

```
__aicore__ inline constexpr [Layout](Layout构造函数.md)(const ShapeType& shape  = {}, const StrideType& stride = {}) : Std::tuple<ShapeType, StrideType>(shape, stride) 
__aicore__ inline constexpr decltype(auto) [layout](layout.md)()
__aicore__ inline constexpr decltype(auto) [layout](layout.md)() const
__aicore__ inline constexpr decltype(auto) [GetShape](GetShape.md)()  
__aicore__ inline constexpr decltype(auto) [GetShape](GetShape.md)() const
__aicore__ inline constexpr decltype(auto) [GetStride](GetStride.md)()    
__aicore__ inline constexpr decltype(auto) [GetStride](GetStride.md)() const
template <typename CoordType> __aicore__ inline constexpr auto operator()(const CoordType& coord) const {}
```

## 相关接口<a name="section104554349817"></a>

```
// Shape结构构造方法
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> [MakeShape](MakeShape.md)(const Ts&... t)

// Stride结构构造方法
template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> [MakeStride](MakeStride.md)(const Ts&... t)

// Layout结构构造方法
template <typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto [MakeLayout](MakeLayout.md)(const ShapeType& shape, const StrideType& stride)

// is_layout原型定义
template <T>
struct [is_layout](is_layout.md);
```

