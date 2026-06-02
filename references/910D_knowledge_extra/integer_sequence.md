# integer\_sequence<a name="ZH-CN_TOPIC_0000002523344832"></a>

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

index\_sequence是Ascend C提供的一个类模板，用于生成一个编译时的整数序列，适用于模板元编程。

make\_index\_sequence是Ascend C提供的一个模板，通常使用make\_index\_sequence创建一个index\_sequence类型的对象，用于生成一个从0到N-1的整数序列。

## 函数原型<a name="section126881859101617"></a>

```
template<size_t... Idx>
using index_sequence = IntegerSequence<size_t, Idx...>;
```

```
template<size_t N>
using make_index_sequence = MakeIntegerSequence<size_t, N>;
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
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>...Idx</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p9731042114910"><a name="p9731042114910"></a><a name="p9731042114910"></a>表示序列的形参包。</p>
<p id="p2190193714595"><a name="p2190193714595"></a><a name="p2190193714595"></a>size_t，在64位系统中为long unsigned int，非64位系统中为unsigned int。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>N</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p1354016543390"><a name="p1354016543390"></a><a name="p1354016543390"></a>生成的整数序列的大小。</p>
<p id="p1895681215432"><a name="p1895681215432"></a><a name="p1895681215432"></a>size_t，在64位系统中为long unsigned int，非64位系统中为unsigned int。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

-   N的范围为\[0, 64\]。
-   index\_sequence作为序列，长度最大为64。

## 返回值说明<a name="section62431148556"></a>

无

## 调用示例<a name="section1193764916212"></a>

生成并打印一个长度为5的整数序列。

```
template<size_t... Is> 
__aicore__  inline void PrintIndexSequence(AscendC::Std::index_sequence<Is...>) {
   ((AscendC::printf(" Is:%lu", Is)), ...);
}
__aicore__ inline void Process()
{
    PrintIndexSequence(AscendC::Std::make_index_sequence<5>{}); // 打印结果: 0，1，2，3，4
    PrintIndexSequence(AscendC::Std::index_sequence<0,1,2,10,8000>{}); // 打印结果: 0，1，2，10, 8000
}
```

