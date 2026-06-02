# Coordinate简介<a name="ZH-CN_TOPIC_0000002554423521"></a>

Coordinate本质上是一个元组（tuple），用于表示张量在不同维度的位置信息，即坐标值。Coordinate（坐标）和[Layout](Layout.md)（布局）、Index（内存位置索引）之间存在紧密的关联：

-   从Coordinate到Index的转换：Layout定义了张量的形状和各维度的步长，根据这些信息和给定的Coordinate，可以计算出该坐标在内存中的位置索引。
-   从Index到Coordinate的转换：基于Layout中定义的形状和步长信息，对于一个已知的内存位置索引，通过相应的计算可以得到该索引对应的Coordinate。

## 原型定义<a name="section10580930144614"></a>

```
template <typename... Coords>
using Coord = Std::tuple<Coords...>
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
<tbody><tr id="row260915573419"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p2060925573411"><a name="p2060925573411"></a><a name="p2060925573411"></a>Coords</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p18109358203112"><a name="p18109358203112"></a><a name="p18109358203112"></a>表示输入数据类型的形参包，参数个数范围为[0，64]。</p>
<p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>输入的数据类型支持size_t和Std::<a href="integral_constant.md">Int</a>。</p>
</td>
</tr>
</tbody>
</table>

## 相关接口<a name="section104554349817"></a>

```
// Coord结构构造方法
template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(Ts const&... t)

// Layout输入，Coordinate转换为内存位置索引Index
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const Layout<ShapeType, StrideType>& layout)

// Shape和Stride输入，Coordinate转换为内存位置索引Index
template <typename CoordType, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto Crd2Idx(const CoordType& coord, const ShapeType& shape, const StrideType& stride)
```

