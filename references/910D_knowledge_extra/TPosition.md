# TPosition<a name="ZH-CN_TOPIC_0000002523344712"></a>

Ascend C管理不同层级的物理内存时，用一种抽象的逻辑位置（TPosition）来表达各级别的存储，代替了片上物理存储的概念，达到隐藏硬件架构的目的。主要的TPosition类型包括：VECIN、VECOUT、VECCALC、A1、A2、B1、B2、C1、C2、CO1、CO2，其中VECIN、VECCALC、VECOUT主要用于矢量编程，A1、A2、B1、B2、C1、C2、CO1、CO2用于矩阵编程。您可以参考[编程范式](编程范式.md)了解TPosition的基础概念，通过[表1](通用说明和约束.md#table07372185712)了解TPosition和物理存储的映射关系。

TPosition定义如下：

```
enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    C1,
    C2,
    CO1,
    CO2,
    VECIN,
    VECOUT,
    VECCALC,
    LCM = VECCALC,
    SPM,
    SHM = SPM,
    TSCM,
    C2PIPE2GM,
    C2PIPE2LOCAL,
    MAX,
};
```

TPosition枚举值的具体定义如下：

**表 1**  TPosition枚举值含义说明

<a name="table5376122715308"></a>
<table><thead align="left"><tr id="row1337716275309"><th class="cellrowborder" valign="top" width="17.53%" id="mcps1.2.3.1.1"><p id="p1537762711305"><a name="p1537762711305"></a><a name="p1537762711305"></a>枚举值</p>
</th>
<th class="cellrowborder" valign="top" width="82.47%" id="mcps1.2.3.1.2"><p id="p153771127123013"><a name="p153771127123013"></a><a name="p153771127123013"></a>具体含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row175152367479"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p1351520369474"><a name="p1351520369474"></a><a name="p1351520369474"></a>GM</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1251517360479"><a name="p1251517360479"></a><a name="p1251517360479"></a>Global Memory，对应<span id="zh-cn_topic_0000001588832845_ph519622414417"><a name="zh-cn_topic_0000001588832845_ph519622414417"></a><a name="zh-cn_topic_0000001588832845_ph519622414417"></a>AI Core</span>的外部存储。</p>
</td>
</tr>
<tr id="row19377627133012"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p2714429161019"><a name="p2714429161019"></a><a name="p2714429161019"></a>VECIN</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p13377122733010"><a name="p13377122733010"></a><a name="p13377122733010"></a>用于矢量计算，搬入数据的存放位置，在数据搬入Vector计算单元时使用此位置。</p>
</td>
</tr>
<tr id="row13377162793019"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p16117319129"><a name="p16117319129"></a><a name="p16117319129"></a>VECOUT</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p12377122712304"><a name="p12377122712304"></a><a name="p12377122712304"></a>用于矢量计算，搬出数据的存放位置，在将Vector计算单元结果搬出时使用此位置。</p>
</td>
</tr>
<tr id="row14563209291"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p95631897919"><a name="p95631897919"></a><a name="p95631897919"></a>VECCALC</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1956359598"><a name="p1956359598"></a><a name="p1956359598"></a>用于矢量计算/矩阵计算，在计算需要临时变量时使用此位置。</p>
</td>
</tr>
<tr id="row28295254916"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p158298253916"><a name="p158298253916"></a><a name="p158298253916"></a>A1</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p68298253918"><a name="p68298253918"></a><a name="p68298253918"></a>用于矩阵计算，存放整块A矩阵，可类比CPU多级缓存中的二级缓存。</p>
</td>
</tr>
<tr id="row696513258910"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p896516251990"><a name="p896516251990"></a><a name="p896516251990"></a>B1</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p139658256916"><a name="p139658256916"></a><a name="p139658256916"></a>用于矩阵计算，存放整块B矩阵，可类比CPU多级缓存中的二级缓存。</p>
</td>
</tr>
<tr id="row21718485391"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p101720489392"><a name="p101720489392"></a><a name="p101720489392"></a>C1</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1976331084020"><a name="p1976331084020"></a><a name="p1976331084020"></a>用于矩阵计算，存放整块Bias矩阵，可类比CPU多级缓存中的二级缓存。</p>
</td>
</tr>
<tr id="row201031261391"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p31036261791"><a name="p31036261791"></a><a name="p31036261791"></a>A2</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1610362610912"><a name="p1610362610912"></a><a name="p1610362610912"></a>用于矩阵计算，存放切分后的小块A矩阵，可类比CPU多级缓存中的一级缓存。</p>
</td>
</tr>
<tr id="row22337267913"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p2023314263913"><a name="p2023314263913"></a><a name="p2023314263913"></a>B2</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p323312261998"><a name="p323312261998"></a><a name="p323312261998"></a>用于矩阵计算，存放切分后的小块B矩阵，可类比CPU多级缓存中的一级缓存。</p>
</td>
</tr>
<tr id="row2087225123915"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p1687225193913"><a name="p1687225193913"></a><a name="p1687225193913"></a>C2</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p250583044018"><a name="p250583044018"></a><a name="p250583044018"></a>用于矩阵计算，存放切分后的小块Bias矩阵，可类比CPU多级缓存中的一级缓存。</p>
</td>
</tr>
<tr id="row14365326693"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p7365162618910"><a name="p7365162618910"></a><a name="p7365162618910"></a>CO1</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p43656266915"><a name="p43656266915"></a><a name="p43656266915"></a>用于矩阵计算，存放小块结果C矩阵，可理解为Cube Out。</p>
</td>
</tr>
<tr id="row194957263919"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p4495142613918"><a name="p4495142613918"></a><a name="p4495142613918"></a>CO2</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p949611261196"><a name="p949611261196"></a><a name="p949611261196"></a>用于矩阵计算，存放整块结果C矩阵，可理解为Cube Out。</p>
</td>
</tr>
<tr id="row1682573410509"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p1082533425015"><a name="p1082533425015"></a><a name="p1082533425015"></a>LCM</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p58251234135017"><a name="p58251234135017"></a><a name="p58251234135017"></a>Local Cache Memory，代表临时共享的Unified Buffer空间，VECCALC的别名，与VECCALC实现同样的功能。</p>
</td>
</tr>
<tr id="row18899424276"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p9889942112716"><a name="p9889942112716"></a><a name="p9889942112716"></a>SPM</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p18891442102718"><a name="p18891442102718"></a><a name="p18891442102718"></a>当Unified Buffer内存有溢出风险时，用于Unified Buffer的数据暂存。</p>
</td>
</tr>
<tr id="row121471975507"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p181474775015"><a name="p181474775015"></a><a name="p181474775015"></a>SHM</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1314714710505"><a name="p1314714710505"></a><a name="p1314714710505"></a>SPM的别名。</p>
</td>
</tr>
<tr id="row12441532142714"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p1845163262710"><a name="p1845163262710"></a><a name="p1845163262710"></a>TSCM</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1245133252711"><a name="p1245133252711"></a><a name="p1245133252711"></a>Temp Swap Cache Memory，用于临时把数据交换到额外空间，进行Matmul运算。</p>
</td>
</tr>
<tr id="row1854865064818"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p1454925074819"><a name="p1454925074819"></a><a name="p1454925074819"></a>C2PIPE2GM</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1159617463492"><a name="p1159617463492"></a><a name="p1159617463492"></a>用于存放FIXPIPE量化参数。</p>
</td>
</tr>
<tr id="row16016386275"><td class="cellrowborder" valign="top" width="17.53%" headers="mcps1.2.3.1.1 "><p id="p11073816276"><a name="p11073816276"></a><a name="p11073816276"></a>C2PIPE2LOCAL</p>
</td>
<td class="cellrowborder" valign="top" width="82.47%" headers="mcps1.2.3.1.2 "><p id="p1601381273"><a name="p1601381273"></a><a name="p1601381273"></a>预留参数。为后续的功能做保留，开发者暂时无需关注。</p>
</td>
</tr>
</tbody>
</table>

