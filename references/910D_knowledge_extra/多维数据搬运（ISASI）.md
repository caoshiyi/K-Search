# 多维数据搬运（ISASI）<a name="ZH-CN_TOPIC_0000002523303892"></a>

## 产品支持情况<a name="section1550532418810"></a>

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

## 功能说明<a name="section474617392321"></a>

多维数据搬运接口，相比于基础数据搬运接口，可更加自由配置搬入的维度信息以及对应的Stride。

## 函数原型<a name="section1954364615315"></a>

-   Global Memory-\> Local Memory ，支持多维度搬运

    ```
    template <typename T, uint8_t dim, const NdDmaConfig& config = kDefaultNdDmaConfig>
    __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const NdDmaParams<T, dim>& params)
    ```

-   NdDma DataCache刷新，在使用DataCopy接口进行数据搬运前，需要使用NdDmaDci接口刷新缓存保证DataCache为最新状态。

    ```
    __aicore__ inline void NdDmaDci()
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table49614585413"></a>
<table><thead align="left"><tr id="row996115820412"><th class="cellrowborder" valign="top" width="14.729999999999999%" id="mcps1.2.3.1.1"><p id="p10961458104110"><a name="p10961458104110"></a><a name="p10961458104110"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.27%" id="mcps1.2.3.1.2"><p id="p6961155817415"><a name="p6961155817415"></a><a name="p6961155817415"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row4961205811416"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p8499192715109"><a name="p8499192715109"></a><a name="p8499192715109"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p15499112721010"><a name="p15499112721010"></a><a name="p15499112721010"></a>源操作数或者目的操作数的数据类型。</p>
</td>
</tr>
<tr id="row63251529183414"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p049932711012"><a name="p049932711012"></a><a name="p049932711012"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p12499132714109"><a name="p12499132714109"></a><a name="p12499132714109"></a>搬运的数据维度, 数据类型为uint8_t，支持的维度为[1, 5]。</p>
</td>
</tr>
<tr id="row3964131831012"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p2499627101017"><a name="p2499627101017"></a><a name="p2499627101017"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p18499327181016"><a name="p18499327181016"></a><a name="p18499327181016"></a>搬运配置选项，NdDmaConfig类型，定义如下，具体参数说明请参考<a href="#table9182515919">表 NdDmaConfig结构体参数定义</a>。</p>
<a name="screen2499192718104"></a><a name="screen2499192718104"></a><pre class="screen" codetype="Cpp" id="screen2499192718104">struct NdDmaConfig {
    static constexpr uint16_t unsetPad = 0xffff;
    bool isNearestValueMode = false;
    uint16_t loopLpSize = unsetPad; // Left padding size of all dimensions, must be less than 256.
    uint16_t loopRpSize = unsetPad; // Right padding size of all dimensions, must be less than 256.
    bool ascOptimize = false;       // used for Ascend C optimization on special senario.
};</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table917mcpsimp"></a>
<table><thead align="left"><tr id="row923mcpsimp"><th class="cellrowborder" valign="top" width="15.02%" id="mcps1.2.4.1.1"><p id="p925mcpsimp"><a name="p925mcpsimp"></a><a name="p925mcpsimp"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="9.86%" id="mcps1.2.4.1.2"><p id="p927mcpsimp"><a name="p927mcpsimp"></a><a name="p927mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.12%" id="mcps1.2.4.1.3"><p id="p929mcpsimp"><a name="p929mcpsimp"></a><a name="p929mcpsimp"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row930mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p2925016172518"><a name="p2925016172518"></a><a name="p2925016172518"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p199251416112517"><a name="p199251416112517"></a><a name="p199251416112517"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p292591672516"><a name="p292591672516"></a><a name="p292591672516"></a>目的操作数，类型为LocalTensor。</p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p49261616142516"><a name="p49261616142516"></a><a name="p49261616142516"></a>源操作数，类型为GlobalTensor。</p>
</td>
</tr>
<tr id="row4726155915388"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p21111014132612"><a name="p21111014132612"></a><a name="p21111014132612"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p71115140262"><a name="p71115140262"></a><a name="p71115140262"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p964217483714"><a name="p964217483714"></a><a name="p964217483714"></a>搬运参数 NdDmaParams类型，定义如下，具体参数说明请参考<a href="#table1247813351789">表 NdDmaParams结构体参数定义</a>。</p>
<a name="screen1929612197911"></a><a name="screen1929612197911"></a><pre class="screen" codetype="Cpp" id="screen1929612197911">template &lt;typename T, uint8_t dim&gt;
struct NdDmaParams  {
    NdDmaLoopInfo&lt;dim&gt; loopInfo;
    T constantValue;  // 若有左右Padding，且不使能NearestValueMode时，该值将作为Padding值填充。
};</pre>
<p id="p395104375712"><a name="p395104375712"></a><a name="p395104375712"></a>NdDmaLoopInfo类型，定义如下，具体参数说明请参考<a href="#table725271775813">表 NdDmaLoopInfo结构体参数定义</a>。</p>
<a name="screen84611628155618"></a><a name="screen84611628155618"></a><pre class="screen" codetype="Cpp" id="screen84611628155618">template &lt;uint8_t dim&gt;
struct NdDmaLoopInfo  {
    uint64_t loopSrcStride[dim] = {0}; // src stride info per loop.
    uint32_t loopDstStride[dim] = {0}; // dst stride info per loop.
    uint32_t loopSize[dim] = {0}; // Loop size per loop.
    uint8_t loopLpSize[dim] = {0}; // Left padding size per loop.
    uint8_t loopRpSize[dim] = {0}; // Right padding size per loop.
};
// 注意: dim的有效范围为[1,5]</pre>
</td>
</tr>
</tbody>
</table>

**表 3**  NdDmaConfig结构体参数定义

<a name="table9182515919"></a>
<table><thead align="left"><tr id="row151816516917"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.3.1.1"><p id="p18182513916"><a name="p18182513916"></a><a name="p18182513916"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85%" id="mcps1.2.3.1.2"><p id="p41815515920"><a name="p41815515920"></a><a name="p41815515920"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1818105113916"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p13842123718586"><a name="p13842123718586"></a><a name="p13842123718586"></a>unsetPad</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p478014752618"><a name="p478014752618"></a><a name="p478014752618"></a>表示不设置PaddingSize，固定为0xFFFF。</p>
</td>
</tr>
<tr id="row2968131992515"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p0180154115816"><a name="p0180154115816"></a><a name="p0180154115816"></a>isNearestValueMode</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p4353141191111"><a name="p4353141191111"></a><a name="p4353141191111"></a>表示Padding值填取方式，类型为bool。</p>
<p id="p17325184218514"><a name="p17325184218514"></a><a name="p17325184218514"></a>True：使能最近值填充方式，即左右Padding值会选取当前维度最左或最右的值进行填充，可参考<a href="#fig10722115123919">图1</a>。</p>
<p id="p1582110246532"><a name="p1582110246532"></a><a name="p1582110246532"></a>False：使能常数填充方式，即所有Padding值填充为固定值NdDmaParams::constantValue。</p>
<p id="p10561101719473"><a name="p10561101719473"></a><a name="p10561101719473"></a>当数据类型为b64时，参数isNearestValueMode的值应为False。</p>
</td>
</tr>
<tr id="row1589112062510"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p10132194415583"><a name="p10132194415583"></a><a name="p10132194415583"></a>loopLpSize</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p19780547162614"><a name="p19780547162614"></a><a name="p19780547162614"></a>表示每个维度内的PaddingSize，当该值不为unsetPad时，则表示所有循环里的左PaddingSize为该值，且会使NdDmaLoopInfo::loopLpSize不生效。默认值为unsetPad，开发者可填的范围为默认值或[0,255]。</p>
</td>
</tr>
<tr id="row3593192082512"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p156644745812"><a name="p156644745812"></a><a name="p156644745812"></a>loopRpSize</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p3426397216"><a name="p3426397216"></a><a name="p3426397216"></a>表示每个维度内的PaddingSize，当该值不为unsetPad时，则表示所有循环里的右PaddingSize为该值，且会使NdDmaLoopInfo::loopRpSize不生效。默认值为unsetPad，开发者可填的范围为默认值或[0,255]。</p>
</td>
</tr>
<tr id="row1185111015588"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p13184115010581"><a name="p13184115010581"></a><a name="p13184115010581"></a>ascOptimize</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p188517014587"><a name="p188517014587"></a><a name="p188517014587"></a>预留参数，暂不支持。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  NdDmaParams结构体参数定义

<a name="table1247813351789"></a>
<table><thead align="left"><tr id="row1147863511819"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.3.1.1"><p id="p1147813514810"><a name="p1147813514810"></a><a name="p1147813514810"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85.00999999999999%" id="mcps1.2.3.1.2"><p id="p9478135782"><a name="p9478135782"></a><a name="p9478135782"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1847873513818"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.3.1.1 "><p id="p12121135141012"><a name="p12121135141012"></a><a name="p12121135141012"></a>loopInfo</p>
</td>
<td class="cellrowborder" valign="top" width="85.00999999999999%" headers="mcps1.2.3.1.2 "><p id="p1453710521859"><a name="p1453710521859"></a><a name="p1453710521859"></a>每维进行搬运的信息，类型为NdDmaLoopInfo&lt;dim&gt;。</p>
<p id="p184785351286"><a name="p184785351286"></a><a name="p184785351286"></a>NdDmaLoopInfo结构中数组类型的参数，其数组索引值对应实际维度信息，索引0 - 4对应1 - 5维。具体参数介绍可参考<a href="#table725271775813">表 NdDmaLoopInfo结构体参数定义</a>。</p>
</td>
</tr>
<tr id="row12478163518812"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.3.1.1 "><p id="p194781351084"><a name="p194781351084"></a><a name="p194781351084"></a>constantValue</p>
</td>
<td class="cellrowborder" valign="top" width="85.00999999999999%" headers="mcps1.2.3.1.2 "><p id="p1724511071219"><a name="p1724511071219"></a><a name="p1724511071219"></a>数据类型为T的数值，当存在维度左右Padding，且不使能NearestValueMode时，该值将作为Padding值填充。</p>
<p id="p113651418121810"><a name="p113651418121810"></a><a name="p113651418121810"></a>当数据类型为b64时，参数constantValue的值应为0。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  NdDmaLoopInfo结构体参数定义

<a name="table725271775813"></a>
<table><thead align="left"><tr id="row32523179589"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.3.1.1"><p id="p62531117195812"><a name="p62531117195812"></a><a name="p62531117195812"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85%" id="mcps1.2.3.1.2"><p id="p625320178587"><a name="p625320178587"></a><a name="p625320178587"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row11253121712582"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p102538173580"><a name="p102538173580"></a><a name="p102538173580"></a>loopSrcStride</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p13815527776"><a name="p13815527776"></a><a name="p13815527776"></a>表示每个维度内，该源操作数元素与下一个元素间的间隔。</p>
<p id="p425391785813"><a name="p425391785813"></a><a name="p425391785813"></a>单位为元素个数。数据类型为uint64_t，srcStride需在[0, 2<sup id="sup34471540362"><a name="sup34471540362"></a><a name="sup34471540362"></a>40</sup>)。</p>
</td>
</tr>
<tr id="row925316174582"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p225351713586"><a name="p225351713586"></a><a name="p225351713586"></a>loopDstStride</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p618412466348"><a name="p618412466348"></a><a name="p618412466348"></a>表示每个维度内，该目的操作数元素与下一个元素间的间隔。</p>
<p id="p2061714315343"><a name="p2061714315343"></a><a name="p2061714315343"></a>单位为元素个数。数据类型为uint32_t，dstStride需在[0, 2<sup id="sup1035914461219"><a name="sup1035914461219"></a><a name="sup1035914461219"></a>20</sup>)。</p>
</td>
</tr>
<tr id="row125321765817"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p8253617175816"><a name="p8253617175816"></a><a name="p8253617175816"></a>loopSize</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p1340117222355"><a name="p1340117222355"></a><a name="p1340117222355"></a>表示每个维度内，处理的元素个数（不包含Padding元素）。</p>
<p id="p134881163163"><a name="p134881163163"></a><a name="p134881163163"></a>单位为元素个数。数据类型为uint32_t，dstStride需在[0, 2<sup id="sup137185506115"><a name="sup137185506115"></a><a name="sup137185506115"></a>20</sup>)。</p>
</td>
</tr>
<tr id="row18253181717586"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p18253617125812"><a name="p18253617125812"></a><a name="p18253617125812"></a>loopLpSize</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p68901728103610"><a name="p68901728103610"></a><a name="p68901728103610"></a>表示每个维度内，左侧需要补齐的元素个数。</p>
<p id="p1023018153350"><a name="p1023018153350"></a><a name="p1023018153350"></a>单位为元素个数。数据类型为uint8_t，srcStride不要超出该数据类型的取值范围。</p>
</td>
</tr>
<tr id="row1425331775814"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="p1925319178583"><a name="p1925319178583"></a><a name="p1925319178583"></a>loopRpSize</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="p32734437364"><a name="p32734437364"></a><a name="p32734437364"></a>表示每个维度内，右侧需要补齐的元素个数。</p>
<p id="p10328102411211"><a name="p10328102411211"></a><a name="p10328102411211"></a>单位为元素个数。数据类型为uint8_t，srcStride不要超出该数据类型的取值范围。</p>
</td>
</tr>
</tbody>
</table>

以下以2维的例子介绍几个典型使用场景。

**图 1**  2D Padding场景<a name="fig10722115123919"></a>  
<!-- img2text -->
```
2D Padding场景

使用NearestValue模式进行填充。dst任意字符串分为padding值，若为constant填充模式，则padding值为用户的所提供的固定值。

NdDmaLoopInfo<2> Info {
  loopSrcStride = {1, 3}
  loopDstStride = {1, 6}
  loopSize      = {3, 4}
  loopLpSize    = {2, 2}
  loopRpSize    = {1, 3}
}

src                                      dst

┌───────┬───────┬───────┐               ┌───────┬───────┬───────┬───────┬───────┬───────┐
│   0   │   1   │   2   │               │   0   │   0   │   0   │   1   │   2   │   2   │ →
├───────┼───────┼───────┤               ├───────┼───────┼───────┼───────┼───────┼───────┤
│   3   │   4   │   5   │               │   0   │   0   │   0   │   1   │   2   │   2   │
├───────┼───────┼───────┤               ├───────┼───────┼───────┼───────┼───────┼───────┤
│   6   │   7   │   8   │               │   0   │   0   │   0   │   1   │   2   │   2   │
├───────┼───────┼───────┤               ├───────┼───────┼───────┼───────┼───────┼───────┤
│   9   │  10   │  11   │ →             │   3   │   3   │   3   │   4   │   5   │   5   │
└───────┴───────┴───────┘               ├───────┼───────┼───────┼───────┼───────┼───────┤
      ↗                                 │   6   │   6   │   6   │   7   │   8   │   8   │
     ╱                                  ├───────┼───────┼───────┼───────┼───────┼───────┤
    ╱                                   │   9   │   9   │   9   │  10   │  11   │  11   │
   ╱                                    ├───────┼───────┼───────┼───────┼───────┼───────┤
  ╱                                     │   9   │   9   │   9   │  10   │  11   │  11   │
 ╱                                      ├───────┼───────┼───────┼───────┼───────┼───────┤
╱                                       │   9   │   9   │   9   │  10   │  11   │  11   │ →
                                        └───────┴───────┴───────┴───────┴───────┴───────┘
                                        ↗
                                       ╱
                                      ╱
                                     ╱
                                    ╱
                                   ╱
```

说明:
- src 为 4×3 数据：
  - 第1行: 0 1 2
  - 第2行: 3 4 5
  - 第3行: 6 7 8
  - 第4行: 9 10 11
- dst 为 8×6 Padding 结果：
  - 上 padding 2 行，下 padding 2 行，左 padding 2 列，右 padding 1 列
  - 采用 NearestValue 模式，边界使用最近值填充
- 中间浅蓝区域对应原始 src 映射到 dst 的有效区域；外围区域为 padding 后的扩展区域
- 图中黑色斜箭头表示按二维遍历/搬运方向从 src 到 dst 的对应关系

**图 2**  2D Transpose场景<a name="fig104451143104510"></a>  
<!-- img2text -->
```
2D Transpose场景

Nd2DParam offsetInfo {
  loopSrcStride = {1, 3};
  loopDstStride = {4, 1};
  loopSize = {3, 4};
  loopUpSize = {0, 0};
  loopRpSize = {0, 0};
}

src                                   dst

┌──────────┬──────────┬──────────┐     ┌──────────┬──────────┬──────────┬──────────┐
│    0     │    1     │    2     │     │    0     │    3     │    6     │    9     │
├──────────┼──────────┼──────────┤     ├──────────┼──────────┼──────────┼──────────┤
│    3     │    4     │    5     │     │    1     │    4     │    7     │    10    │
├──────────┼──────────┼──────────┤     ├──────────┼──────────┼──────────┼──────────┤
│    6     │    7     │    8     │  →  │    2     │    5     │    8     │    11    │
├──────────┼──────────┼──────────┤     └──────────┴──────────┴──────────┴──────────┘
│    9     │    10    │    11    │
└──────────┴──────────┴──────────┘

src访问顺序:
0 → 1 → 2
          ↘
            3 → 4 → 5
                      ↘
                        6 → 7 → 8
                                  ↘
                                    9 → 10 → 11

dst排布结果:
0 → 3 → 6 → 9
↓
1 → 4 → 7 → 10
↓
2 → 5 → 8 → 11
```

**图 3**  2D BroadCast场景<a name="fig4654326132913"></a>  
<!-- img2text -->
```
2D BroadCast场景

Nd2DTransposeInfo<T>
{
    loopSrcStride = {1, 0};
    loopDstStride = {1, 3};
    loopSize = {3, 4};
    loopLpSize = {0, 0};
    loopRpSize = {0, 0};
}

src                              dst

┌─────────┬─────────┬─────────┐   ┌─────────┬─────────┬─────────┐
│    0    │    1    │    2    │   │    0    │    1    │    2    │
└─────────┴─────────┴─────────┘   ├─────────┼─────────┼─────────┤
      ───────────────────→        │    0    │    1    │    2    │
                                   ├─────────┼─────────┼─────────┤
                                   │    0    │    1    │    2    │
                                   ├─────────┼─────────┼─────────┤
                                   │    0    │    1    │    2    │
                                   └─────────┴─────────┴─────────┘
                                     ↑
                                     │
                                     └────────────────────────↗

                                   ───────────────────→
```

**图 4**  2D Slice场景<a name="fig13061435132915"></a>  
<!-- img2text -->
```
2D Slice场景

将4*5 Shape的2D矩阵裁剪为一个3*3的2D矩阵
NdDmaLoopInfo<> Info {
    loopSrcStride = {1, 4};
    loopDstStride = {1, 3};
    loopSize = {3, 3};
    loopLpSize = {0, 0};
    loopRpSize = {0, 0};
}

src                              dst

┌─────┬─────┬─────┬─────┐         ┌─────┬─────┬─────┐
│  0  │  1  │  2  │  3  │         │  0  │  1  │  2  │
├─────┼─────┼─────┼─────┤         ├─────┼─────┼─────┤
│  4  │  5  │  6  │  7  │         │  4  │  5  │  6  │
├─────┼─────┼─────┼─────┤         ├─────┼─────┼─────┤
│  8  │  9  │ 10  │ 11  │         │  8  │  9  │ 10  │
├─────┼─────┼─────┼─────┤         └─────┴─────┴─────┘
│ 12  │ 13  │ 14  │ 15  │
├─────┼─────┼─────┼─────┤
│ 16  │ 17  │ 18  │ 19  │
└─────┴─────┴─────┴─────┘

src访问/裁剪路径:                     dst写入路径:

0 ─────────────→ 3                    0 ─────────→ 2
                  ↘                                   ↘
                    ↘                                   ↘
                      ↘                                   ↘
                        ↘                                   ↘
                          ↘                                   ↘
                            ↘                                   ↘
                              ↘                                   ↘
                                ↘                                   ↘
                                  16 ─────────→ 19                  8 ─────────→ 10
```

说明:
- src 为 4×5 矩阵，元素按行编号 0~19。
- dst 为 3×3 矩阵，元素按行编号 0~10（图中仅展示位置编号，对应裁剪后的3列×3行区域）。
- 图中斜箭头表示按二维循环参数从 src 中裁剪出 3×3 区域并写入 dst。
- 参数含义按图中文字保留：
  - loopSrcStride = {1, 4}
  - loopDstStride = {1, 3}
  - loopSize = {3, 3}
  - loopLpSize = {0, 0}
  - loopRpSize = {0, 0}

## 通路说明<a name="section631mcpsimp"></a>

**表 6**  数据通路和数据类型

<a name="table14255161718545"></a>
<table><thead align="left"><tr id="row3255181710543"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.2.4.1.1"><p id="p52550177546"><a name="p52550177546"></a><a name="p52550177546"></a>支持型号</p>
</th>
<th class="cellrowborder" valign="top" width="23.93%" id="mcps1.2.4.1.2"><p id="p13255191735420"><a name="p13255191735420"></a><a name="p13255191735420"></a>数据通路（通过<a href="TPosition.md#table5376122715308">TPosition</a>表达）</p>
</th>
<th class="cellrowborder" valign="top" width="65.03%" id="mcps1.2.4.1.3"><p id="p6255617175419"><a name="p6255617175419"></a><a name="p6255617175419"></a>源操作数和目的操作数的数据类型 (两者保持一致)</p>
</th>
</tr>
</thead>
<tbody><tr id="row1125561715416"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.1 "><p id="p539445910177"><a name="p539445910177"></a><a name="p539445910177"></a><span id="ph1016119599116"><a name="ph1016119599116"></a><a name="ph1016119599116"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.93%" headers="mcps1.2.4.1.2 "><p id="p925571775417"><a name="p925571775417"></a><a name="p925571775417"></a>GM -&gt; VECIN</p>
</td>
<td class="cellrowborder" valign="top" width="65.03%" headers="mcps1.2.4.1.3 "><p id="p16499827151015"><a name="p16499827151015"></a><a name="p16499827151015"></a>b8、b16、b32、b64</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section7607175220218"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   一条指令所能获取的所有数据的地址范围宽度不能超过40位（1TB），即：

    源操作数的每一次循环的大小为：\(loopLpSize + loopSize + loopRpSize -1 \) \* loopSrcSize，目的操作数的每一次循环的大小为：\(loopLpSize + loopSize + loopRpSize -1 \) \* loopDstSize，所有的循环的大小加起来不超过2的40次方位。

-   当每层循环的dstStride为升序序列，则不同循环间的地址空间不能交织或者重叠。以一个2D Padding场景为例，loopSrcStride、loopDstStride第二个维度的stride值最小是3，数据3不能落在维度1的循环中。

    <!-- img2text -->
```
2D Padding场景，NearestValue模式，dst红色部分为padding值。若为constant模式，则padding为用户的所提供的固定值
NdDmaLoopInfo<> Info {
    loopSrcStride = {1, 3};
    loopDstStride = {1, 3};
    loopSize = {3, 2};
    loopLpSize = {0, 0};
    loopRpSize = {0, 0};
}

                    维度1                                 维度1

src                                               dst
                    ┌─────────┬─────────┬─────────┐      ┌─────────┬─────────┬─────────┐
                    │    0    │    1    │    2    │      │    0    │    1    │    2    │
维度2                ├─────────┼─────────┼─────────┤  维度2├─────────┼─────────┼─────────┤
                    │    3    │    4    │    5    │      │    3    │    4    │    5    │
                    └─────────┴─────────┴─────────┘      └─────────┴─────────┴─────────┘
                      ↑                                 ↑
                      └──────────────→──────────────────┘
                                   2

                      ←───────────────────────────────
                                   3

                      ───────────────────────────────→
                                   3
```

-   该接口通过NDDMA进行数据搬运，对应的NDDMA Cache大小为32KB，在使用DataCopy接口进行数据搬运前，需要使用NdDmaDci接口刷新缓存，否则多核场景下读写同一块GM地址可能会导致部分核读取数据错误。

## 调用示例<a name="section122101199486"></a>

```
#include "kernel_operator.h"
template <typename T>
class Kernel {
public:
    __aicore__ inline Kernel() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t count, AscendC::TPipe* pipeIn)
    {
        this->pipe = pipeIn;
        this->count = count;
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        this->pipe->InitBuffer(inQueueX, 1, sizeof(T) * count);
        this->pipe->InitBuffer(outQueueY, 1, sizeof(T) * count);
    }
    // 使用nddma搬运二维数据
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::Duplicate<T>(xLocal, 0, count);
        AscendC::NdDmaLoopInfo<2> loopInfo{{1, 8}, {1, 16}, {8, 2}, {3, 1}, {5, 1}};
        AscendC::NdDmaParams<T, 2> params = {loopInfo, 0};
        // 刷新cache
        AscendC::NdDmaDci();
        // 使用默认参数即可
        static constexpr AscendC::NdDmaConfig config;
        AscendC::DataCopy<T, 2, config>(xLocal, xGm, params);
        inQueueX.EnQue<T>(xLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::DataCopy(yLocal, xLocal, count);
        outQueueY.EnQue<T>(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm, yLocal, count);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    AscendC::TPipe* pipe = nullptr;
    uint32_t count;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
};
extern "C" __global__ __aicore__ void kernel_nddma(GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    Kernel<float> kernel;
    kernel.Init(x, y, 1024, &pipe);
    kernel.Process();
}
```

结果示例：

```
输入数据(srcGlobal): [
[1 2  3  4  5  6  7  8]
[9 10 11 12 13 14 15 16 0 0 ...]]
输出数据(dstGlobal):[
[0 0 0 0 0  0  0  0  0  0  0  0 0 0 0 0]
[0 0 0 1 2  3  4  5  6  7  8  0 0 0 0 0]
[0 0 0 9 10 11 12 13 14 15 16 0 0 0 0 0]
[0 0 0 0 0  0  0  0  0  0  0  0 0 0 0 0 0 0 ...]]
```

