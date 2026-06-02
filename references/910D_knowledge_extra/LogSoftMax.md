# LogSoftMax<a name="ZH-CN_TOPIC_0000002554424279"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

对输入tensor做LogSoftmax计算。计算公式如下 ：

<!-- img2text -->
$$
\operatorname{LogSoftmax}(x_i)=\log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)
$$

<!-- img2text -->
$$
\operatorname{LogSoftmax}(x_i)=\log \left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right)=x_i-\log \left(\sum_j \exp(x_j)\right)
$$

为方便理解，通过Python脚本实现的方式表达计算公式如下，其中src是源操作数（输入），dst、sum、max为目的操作数（输出）。

```
def log_softmax(src):
    #基于last轴进行rowmax(按行取最大值)处理
    max = np.max(src, axis=-1, keepdims=True)
    sub = src - max
    exp = np.exp(sub)
    #基于last轴进行rowsum(按行求和)处理
    sum = np.sum(exp, axis=-1, keepdims=True)
    dst = exp / sum
    dst = np.log10(dst)
    return dst, max, sum
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[m, k\]的输入Tensor为例，描述LogSoftMax高阶API内部算法框图，如下图所示。

**图 1**  LogSoftMax算法框图<a name="fig138551349173015"></a>  
<!-- img2text -->
```text
                         ┌────────┐
                         │ x[m,k] │
                         └────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│       ┌──────────────────────────────┐                       │
│       │          reducemax           │                       │
│       │        ([m,k]->[m,1])        │                       │
│       └──────────────────────────────┘                       │
│                      │                                       │
│                      ▼                                       │
│       ┌──────────────────────────────┐         ┌──────────┐  │
│       │          broadcast           │────────→│ max[m,8] │  │
│       │        ([m,1]->[m,8])        │         └──────────┘  │
│       └──────────────────────────────┘                       │
│                      │                                       │
│                      ▼                                       │
│       ┌──────────────────────────────┐                       │
│  ┌───→│            y=sub             │                       │
│  │    │    (x[m,k] - max[m,8])       │                       │
│  │    └──────────────────────────────┘                       │
│  │                   │                                      │
│  │                   ▼                                      │
│  │    ┌──────────────────────────────┐                      │
│  │    │         exp(y[m,k])          │                      │
│  │    └──────────────────────────────┘                      │
│  │                   │                                      │
│  │                   ▼                                      │
│  │    ┌──────────────────────────────┐                      │
│  │    │          reducesum           │                      │
│  │    │        ([m,k]->[m,1])        │                      │
│  │    └──────────────────────────────┘                      │
│  │                   │                                      │
│  │                   ▼                                      │
│  │    ┌──────────────────────────────┐         ┌──────────┐ │
│  │    │          broadcast           │────────→│ sum[m,8] │ │
│  │    │        ([m,1]->[m,8])        │         └──────────┘ │
│  │    └──────────────────────────────┘                      │
│  │                   │                                      │
│  │                   ▼                                      │
│  └───→┌──────────────────────────────┐                      │
│       │            y=div             │                      │
│       │      (y[m,k] / sum[m,8])     │                      │
│       └──────────────────────────────┘                      │
│                      │                                       │
│                      ▼                                       │
│       ┌──────────────────────────────┐                       │
│       │         log10(y[m,k])        │                       │
│       └──────────────────────────────┘                       │
│                      │                                       │
└──────────────────────┼───────────────────────────────────────┘
                       ▼
                  ┌────────┐
                  │ y[m,k] │
                  └────────┘


图示:
输入输出Tensor   ┌──────────┐
                │          │
                └──────────┘

vector计算      ┌──────────────┐
                │              │
                └──────────────┘

数据流向        ─────────────→
```

计算过程分为如下几步，均在Vector上进行：

1.  reducemax步骤：对输入x的每一行数据求最大值得到\[m, 1\]，计算结果会保存到一个临时空间temp中；
2.  broadcast步骤：对temp中的数据\(\[m, 1\]\)做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，同时输出max；
3.  sub步骤：对输入x的所有数据按行减去max；
4.  exp步骤：对sub之后的所有数据求exp；
5.  reducesum步骤：对exp后的结果的每一行数据求和得到\[m, 1\]，计算结果会保存到临时空间temp中；
6.  broadcast步骤：对temp\(\[m, 1\]\)做一个按datablock为单位的填充，比如float类型下，把\[m, 1\]扩展成\[m, 8\]，同时输出sum；
7.  div步骤：对exp结果的所有数据按行除以sum；
8.  log步骤：对div后的所有数据按行做log10计算，输出y。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void LogSoftMax(const LocalTensor<T>& dst, const LocalTensor<T>& sum, const LocalTensor<T>& max, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, const LogSoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者通过sharedTmpBuffer入参传入。临时空间大小BufferSize的获取方式如下：通过[LogSoftMax Tiling](LogSoftMax-Tiling.md)中提供的接口获取空间范围的大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p667851974013"><a name="p667851974013"></a><a name="p667851974013"></a><span id="ph14851141511365"><a name="ph14851141511365"></a><a name="ph14851141511365"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p98212044172612"><a name="p98212044172612"></a><a name="p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p58813365307"><a name="p58813365307"></a><a name="p58813365307"></a>源操作数是否为NZ格式。默认值为false。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.37%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.39%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p10724195415422"><a name="p10724195415422"></a><a name="p10724195415422"></a>last轴长度需要32Byte对齐。</p>
</td>
</tr>
<tr id="row15331141135911"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p7923102215116"><a name="p7923102215116"></a><a name="p7923102215116"></a>sum</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p1698532712599"><a name="p1698532712599"></a><a name="p1698532712599"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p11602145105916"><a name="p11602145105916"></a><a name="p11602145105916"></a>reduceSum操作数。</p>
<p id="p147313188017"><a name="p147313188017"></a><a name="p147313188017"></a>reduceSum操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p126027454597"><a name="p126027454597"></a><a name="p126027454597"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul12308151213513"></a><a name="ul12308151213513"></a><ul id="ul12308151213513"><li>sum的last轴长度固定为32Byte，即一个<span id="ph68171939163014"><a name="ph68171939163014"></a><a name="ph68171939163014"></a>datablock</span>长度。该<span id="ph138891342173012"><a name="ph138891342173012"></a><a name="ph138891342173012"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph194256461302"><a name="ph194256461302"></a><a name="ph194256461302"></a>datablock</span>中的16个数均为相同的reducesum的值。</li><li>非last轴的长度与目的操作数保持一致。</li></ul>
</td>
</tr>
<tr id="row199571919121112"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p8957151961116"><a name="p8957151961116"></a><a name="p8957151961116"></a>max</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p942715271596"><a name="p942715271596"></a><a name="p942715271596"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p9145644165910"><a name="p9145644165910"></a><a name="p9145644165910"></a>reduceMax操作数。</p>
<p id="p59618114014"><a name="p59618114014"></a><a name="p59618114014"></a>reduceMax操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p19145194415916"><a name="p19145194415916"></a><a name="p19145194415916"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul9474401463"></a><a name="ul9474401463"></a><ul id="ul9474401463"><li>max的last轴长度固定为32Byte，即一个<span id="ph1217011525305"><a name="ph1217011525305"></a><a name="ph1217011525305"></a>datablock</span>长度。该<span id="ph166671453193017"><a name="ph166671453193017"></a><a name="ph166671453193017"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph89115620307"><a name="ph89115620307"></a><a name="ph89115620307"></a>datablock</span>中的16个数均为相同的reducemax的值。</li><li>非last轴的长度与目的操作数保持一致。</li></ul>
</td>
</tr>
<tr id="row685794145814"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p14692842195812"><a name="p14692842195812"></a><a name="p14692842195812"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p9692642135811"><a name="p9692642135811"></a><a name="p9692642135811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p6692842105813"><a name="p6692842105813"></a><a name="p6692842105813"></a>源操作数。</p>
<p id="p19692142135811"><a name="p19692142135811"></a><a name="p19692142135811"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p46921042135815"><a name="p46921042135815"></a><a name="p46921042135815"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1786132117583"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p1265172114584"><a name="p1265172114584"></a><a name="p1265172114584"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p8651142118586"><a name="p8651142118586"></a><a name="p8651142118586"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p14651182114583"><a name="p14651182114583"></a><a name="p14651182114583"></a>临时缓存。临时空间大小BufferSize的获取方式请参考<a href="LogSoftMax-Tiling.md">LogSoftMax Tiling</a>。</p>
<p id="p9651021165815"><a name="p9651021165815"></a><a name="p9651021165815"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row41301530181918"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p13630123491515"><a name="p13630123491515"></a><a name="p13630123491515"></a>LogSoftMax计算所需Tiling信息，Tiling信息的获取请参考<a href="LogSoftMax-Tiling.md">LogSoftMax Tiling</a>。</p>
</td>
</tr>
<tr id="row71298302198"><td class="cellrowborder" valign="top" width="17.37%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>src的shape信息。SoftMaxShapeInfo类型，具体定义如下：</p>
<a name="screen1740392817204"></a><a name="screen1740392817204"></a><pre class="screen" codetype="Cpp" id="screen1740392817204">struct SoftMaxShapeInfo {
    uint32_t srcM; // 非尾轴长度的乘积
    uint32_t srcK; // 尾轴长度，必须32Bytes对齐
    uint32_t oriSrcM; // 原始非尾轴长度的乘积
    uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
<p id="p1065853610376"><a name="p1065853610376"></a><a name="p1065853610376"></a>注意，当输入输出的数据格式为NZ（FRACTAL_NZ）格式时，尾轴长度为reduce轴长度，即<a href="SoftMax.md#fig0172155842215">图2</a>中的W0*W1，非尾轴为H0*H1。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   输入源数据需保持值域在\[-2147483647.0, 2147483647.0\]。若输入不在范围内，输出结果无效。
-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

完整的样例请参考[logsoftmax算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/logsoftmax)。

```
//DTYPE_X、DTYPE_A、DTYPE_B、DTYPE_C分别表示源操作数、目的操作数、maxLocal、sumLocal操作数数据类型
pipe.InitBuffer(inQueueX, BUFFER_NUM, totalLength * sizeof(DTYPE_X));
pipe.InitBuffer(outQueueA, BUFFER_NUM, totalLength * sizeof(DTYPE_A));
pipe.InitBuffer(outQueueB, BUFFER_NUM, outsize * sizeof(DTYPE_B));
pipe.InitBuffer(outQueueC, BUFFER_NUM, outsize * sizeof(DTYPE_C));
pipe.InitBuffer(tmpQueue, BUFFER_NUM, tmpsize);
AscendC::LocalTensor<DTYPE_X> srcLocal = inQueueX.DeQue<DTYPE_X>();
AscendC::LocalTensor<DTYPE_A> dstLocal = outQueueA.AllocTensor<DTYPE_A>();
AscendC::LocalTensor<DTYPE_B> maxLocal = outQueueB.AllocTensor<DTYPE_B>();
AscendC::LocalTensor<DTYPE_C> sumLocal = outQueueC.AllocTensor<DTYPE_C>();
AscendC::SoftMaxShapeInfo softmaxInfo = {outter, inner, outter, inner};
AscendC::LocalTensor<uint8_t> tmpLocal = tmpQueue.AllocTensor<uint8_t>();
AscendC::LogSoftMax<DTYPE_X, false>(dstLocal, sumLocal, maxLocal, srcLocal, tmpLocal, softmaxTiling, softmaxInfo);
```

结果示例如下：

```
输入数据(srcLocal): [0.80541134 0.08385705 0.49426016 ...  0.30962205 0.28947052]
输出数据(dstLocal): [-0.6344272 -1.4868407 -1.0538127  ...  -1.2560008 -1.2771227]
```

