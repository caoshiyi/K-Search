# make\_tuple<a name="ZH-CN_TOPIC_0000002554344535"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section7376114729"></a>

make\_tuple是一个实用的函数模板，其作用在于便捷地创建tuple对象。它可以自动推断元素的类型，使代码更简洁，也可以构造元素列表。

## 函数原型<a name="section126881859101617"></a>

```
template <typename ...Tps>
__aicore__ inline constexpr tuple<unwrap_decay_t<Tps>...> make_tuple(Tps&& ...args)
```

## 参数说明<a name="section121562129312"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p139231650195511"><a name="p139231650195511"></a><a name="p139231650195511"></a>Tps...</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p1583113235518"><a name="p1583113235518"></a><a name="p1583113235518"></a>Tps...为传入tuple的模板参数包，代表传递给make_tuple的参数类型，参数个数范围为[0, 64]。</p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、int4b_t、int8_t、uint8_t、fp8_e8m0_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t、LocalTensor、GlobalTensor。</p>
</td>
</tr>
<tr id="row184944345313"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p48491143195319"><a name="p48491143195319"></a><a name="p48491143195319"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p896332005618"><a name="p896332005618"></a><a name="p896332005618"></a>args...是函数参数包，代表传递给make_tuple的实际参数，参数个数范围为[0, 64]。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

-   tuple实例化深度为64，即支持64个元素以内的基础数据类型的聚合。
-   make\_tuple中需要特别指定数据类型的元素，必须对该元素的数据类型增加强转，否则由编译器自行推断，可能与预期不符。
-   不支持数组等可变长度的数据类型。
-   不支持隐式转换构造函数。

## 返回值说明<a name="section62431148556"></a>

一个tuple对象，此对象包含传递的参数副本。

## 调用示例<a name="section1193764916212"></a>

```
AscendC::Std::tuple<uint32_t, float, bool> test = AscendC::Std::make_tuple(22, 3.3, true);
```

更多调用示例请参见[示例](tuple.md#section1193764916212)。

