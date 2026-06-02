# Broadcast<a name="ZH-CN_TOPIC_0000002523343902"></a>

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

## 功能说明<a name="section785018556590"></a>

将输入按照输出shape进行广播。

比如A的shape为\(2,1\)，广播的目标shape为\(2,16\)，则会将原来的一列扩展为相同的16列。

```
输入数据： 
[[ 1]
 [ 2]]
输出数据： 
[[ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
 [ 2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]]
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，\[m, 1\]广播到\[m, k\]为例，描述Broadcast高阶API内部算法框图，如下图所示。

**图 1**  Broadcast算法框图<a name="fig1957114910209"></a>  
<!-- img2text -->
```
                    ┌────────┐
                    │ x[m,1] │
                    └────────┘
                         │
                         ↓
┌──────────────────────────────────────┐
│                              k对齐场景 │
│            ┌────────────────┐        │
│            │      Brcb      │        │
│            │  ([m,1]->[m,8])│        │
│            └────────────────┘        │
│                     │                │
│                     ↓                │
│            ┌────────────────┐        │
│            │      Copy      │        │
│            │  ([m,8]->[m,k])│        │
│            └────────────────┘        │
└──────────────────────────────────────┘
                         │
                         ↓
                    ┌────────┐
                    │ y[m,k] │
                    └────────┘


                           ┌────────┐
                           │ x[m,1] │
                           └────────┘
                                │
                                ↓
┌────────────────────────────────────────────────┐
│                                        k非对齐场景 │
│                  ┌────────────────┐            │
│                  │      Brcb      │            │
│                  │  ([m,1]->[m,8])│            │
│                  └────────────────┘            │
│                           │                    │
│                           ↓                    │
│                  ┌────────────────┐            │
│                  │      Copy      │            │
│                  │  ([m,8]->[m,k'])│           │
│                  └────────────────┘            │
│                           │                    │
│                           ↓                    │
│                  ┌────────────────┐            │
│                  │   GatherMask   │            │
│                  │  ([m,k']->[m,k])│           │
│                  └────────────────┘            │
└────────────────────────────────────────────────┘
                                │
                                ↓
                           ┌────────┐
                           │ y[m,k] │
                           └────────┘


图示:
输入输出Tensor    ┌────────┐
                │        │
                └────────┘

vector计算       ┌────────────────┐
                │                │
                └────────────────┘

数据流向             ───→
```

计算过程分为如下几步，均在Vector上进行：

1.  brcb步骤：将每个元素广播为一个datablock；
2.  Copy步骤：将每个datablock均复制为多个datablock，k对齐场景下即为结果y；
3.  对于k非对齐的场景，再使用GatherMask截取\[m, k\]个元素， 其中k'表示k向上对齐32B的大小。

## 函数原型<a name="section8850255125911"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
    __aicore__ inline void Broadcast(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t>& sharedTmpBuffer)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
    __aicore__ inline void Broadcast(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dstShape[dim], const uint32_t srcShape[dim])
    ```

-   支持动态shape

    ```
    template<class T, int constRank=-1, uint32_t* constDstShape = nullptr, uint32_t* constSrcShape = nullptr, bool constSrcInnerPad = false>
    __aicore__ inline void Broadcast(const LocalTensor<T>& dst, const LocalTensor<T>& src, const uint32_t* dstShape, const uint32_t* srcShape, BroadcastTiling* tiling)
    ```

该接口需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetBroadCastMaxMinTmpSize](GetBroadCastMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

另外，提供了一个Kernel侧计算Tiling的接口，针对Broadcast的实现计算Tiling，获取Tiling结果。该接口的模板参数功能与支持动态shape的Broadcast接口模板参数相同，其余参数说明请参见[表5](#table5458981523)。

-   **kernel侧tiling计算接口**

    ```
    template<class T, int constRank=-1, uint32_t* constDstShape = nullptr, uint32_t* constSrcShape = nullptr>
    __aicore__ inline void GetBroadcastTilingInfo(uint32_t rank, const uint32_t* dstShape, const uint32_t* srcShape, bool srcInnerPad, BroadcastTiling& tiling)
    ```

## 参数说明<a name="section1085025505914"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>操作数的数据类型。</p>
<p id="p8481936151811"><a name="p8481936151811"></a><a name="p8481936151811"></a><span id="ph4481136201817"><a name="ph4481136201817"></a><a name="ph4481136201817"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：b8、b16、b32、b64位宽对应的数据类型，具体数据类型请参考<a href="内置数据类型.md#section16395539499">不同位宽对应的数据类型</a>。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>dim</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p33819162174"><a name="p33819162174"></a><a name="p33819162174"></a>输入/输出tensor的维度，目前仅支持1维和2维。</p>
</td>
</tr>
<tr id="row6777152811436"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p23791451102416"><a name="p23791451102416"></a><a name="p23791451102416"></a>axis</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p161350818582"><a name="p161350818582"></a><a name="p161350818582"></a>要广播的维度，目前仅支持0和1。</p>
</td>
</tr>
<tr id="row6563634154317"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1838644151511"><a name="p1838644151511"></a><a name="p1838644151511"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p73844410158"><a name="p73844410158"></a><a name="p73844410158"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  支持动态shape接口的模板参数说明

<a name="table15191144396"></a>
<table><thead align="left"><tr id="row1319214441495"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p81928441990"><a name="p81928441990"></a><a name="p81928441990"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1719217441792"><a name="p1719217441792"></a><a name="p1719217441792"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row18193154413914"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p171931544793"><a name="p171931544793"></a><a name="p171931544793"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p919316441397"><a name="p919316441397"></a><a name="p919316441397"></a>操作数的数据类型，目前支持int8_t、uint8_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
<tr id="row21939441394"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p131936448917"><a name="p131936448917"></a><a name="p131936448917"></a>constRank</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p2410710134420"><a name="p2410710134420"></a><a name="p2410710134420"></a>输入/输出tensor的维度数目。</p>
<a name="ul1984451212447"></a><a name="ul1984451212447"></a><ul id="ul1984451212447"><li>默认值-1为动态shape场景，内部按照<span>GetBroadcastTilingInfo</span>接口中的参数rank计算；</li><li>constRank大于0时，必须与<span>GetBroadcastTilingInfo</span>接口中的参数rank取值相同。rank当前支持的范围为[1, 9]。</li></ul>
</td>
</tr>
<tr id="row101939443911"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p619317445910"><a name="p619317445910"></a><a name="p619317445910"></a>constDstShape</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p144451218154512"><a name="p144451218154512"></a><a name="p144451218154512"></a>输出tensor的shape。uint32_t类型的数组。</p>
<a name="ul55111165020"></a><a name="ul55111165020"></a><ul id="ul55111165020"><li>该数组中任一维度取值为0，表示该维度为动态场景，该维度实际取值由参数dstShape对应维度取值决定。</li><li>该数组中任一维度取值大于0，表示该维度为静态场景，该维度取值与参数dstShape中对应维度取值相同。</li></ul>
<p id="p7193104420911"><a name="p7193104420911"></a><a name="p7193104420911"></a>该参数预留，传入默认值nullptr即可。</p>
</td>
</tr>
<tr id="row719311442914"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p18193744295"><a name="p18193744295"></a><a name="p18193744295"></a>constSrcShape</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p1137512315555"><a name="p1137512315555"></a><a name="p1137512315555"></a>输入tensor的shape。uint32_t类型的数组。</p>
<a name="ul937518314557"></a><a name="ul937518314557"></a><ul id="ul937518314557"><li>该数组中任一维度取值为0，表示该维度为动态场景，实际shape由参数srcShape决定。</li><li>该数组中任一维度取值大于0，表示该维度为静态场景，与参数srcShape中对应维度取值相同。</li></ul>
<p id="p1419315443918"><a name="p1419315443918"></a><a name="p1419315443918"></a>该参数预留，传入默认值nullptr即可。</p>
</td>
</tr>
<tr id="row719394413913"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1519312443912"><a name="p1519312443912"></a><a name="p1519312443912"></a><span>constSrcInnerPad</span></p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p11936441299"><a name="p11936441299"></a><a name="p11936441299"></a>表示输入的最后一维srcShape[rank-1]是否32B对齐，其中rank为输入/输出tensor的维度数目。</p>
<p id="p91932441997"><a name="p91932441997"></a><a name="p91932441997"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  接口参数说明

<a name="table1485015517590"></a>
<table><thead align="left"><tr id="row885118552595"><th class="cellrowborder" valign="top" width="19.24%" id="mcps1.2.4.1.1"><p id="p1585195518592"><a name="p1585195518592"></a><a name="p1585195518592"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.23%" id="mcps1.2.4.1.2"><p id="p0851185511597"><a name="p0851185511597"></a><a name="p0851185511597"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.53%" id="mcps1.2.4.1.3"><p id="p1785175516591"><a name="p1785175516591"></a><a name="p1785175516591"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2851125520594"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p9851165515593"><a name="p9851165515593"></a><a name="p9851165515593"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p1785185514591"><a name="p1785185514591"></a><a name="p1785185514591"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>目的操作数。</p>
<p id="p12851115519599"><a name="p12851115519599"></a><a name="p12851115519599"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row6851155510593"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1385135518597"><a name="p1385135518597"></a><a name="p1385135518597"></a>srcLocal</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p585119553596"><a name="p585119553596"></a><a name="p585119553596"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p963863814519"><a name="p963863814519"></a><a name="p963863814519"></a>源操作数。</p>
<p id="p493465115344"><a name="p493465115344"></a><a name="p493465115344"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p15450144034510"><a name="p15450144034510"></a><a name="p15450144034510"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row4852185535916"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1244747105613"><a name="p1244747105613"></a><a name="p1244747105613"></a>dstShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p44478765615"><a name="p44478765615"></a><a name="p44478765615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p5692173212345"><a name="p5692173212345"></a><a name="p5692173212345"></a>输出tensor的shape：uint32_t类型的数组，长度为1或者2， 输入/输出的shape维度数目必须一致。</p>
</td>
</tr>
<tr id="row204461978565"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p14852105575915"><a name="p14852105575915"></a><a name="p14852105575915"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p168521855115913"><a name="p168521855115913"></a><a name="p168521855115913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p1281673324712"><a name="p1281673324712"></a><a name="p1281673324712"></a>输入tensor的shape：uint32_t类型的数组，长度为1或者2， 输入/输出的shape维度数目必须一致。</p>
</td>
</tr>
<tr id="row171991119901"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p22166407018"><a name="p22166407018"></a><a name="p22166407018"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p621617401705"><a name="p621617401705"></a><a name="p621617401705"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。</p>
<p id="p979635010404"><a name="p979635010404"></a><a name="p979635010404"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1853387155411"><a name="p1853387155411"></a><a name="p1853387155411"></a>用于Broadcast内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetBroadCastMaxMinTmpSize.md">GetBroadCastMaxMinTmpSize</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  支持动态shape接口的参数说明

<a name="table9839113891119"></a>
<table><thead align="left"><tr id="row18839113817119"><th class="cellrowborder" valign="top" width="19.24%" id="mcps1.2.4.1.1"><p id="p16839103817114"><a name="p16839103817114"></a><a name="p16839103817114"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.23%" id="mcps1.2.4.1.2"><p id="p1583910385112"><a name="p1583910385112"></a><a name="p1583910385112"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.53%" id="mcps1.2.4.1.3"><p id="p8839183881116"><a name="p8839183881116"></a><a name="p8839183881116"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1684119386113"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1884143817118"><a name="p1884143817118"></a><a name="p1884143817118"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p18841123819111"><a name="p18841123819111"></a><a name="p18841123819111"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p584112389111"><a name="p584112389111"></a><a name="p584112389111"></a>目的操作数。</p>
<p id="p984119385117"><a name="p984119385117"></a><a name="p984119385117"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row884183881119"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p28411838191117"><a name="p28411838191117"></a><a name="p28411838191117"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p084103821116"><a name="p084103821116"></a><a name="p084103821116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p18841193861112"><a name="p18841193861112"></a><a name="p18841193861112"></a>源操作数。</p>
<p id="p38411038101117"><a name="p38411038101117"></a><a name="p38411038101117"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p1984143881117"><a name="p1984143881117"></a><a name="p1984143881117"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1684183812118"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1684117384117"><a name="p1684117384117"></a><a name="p1684117384117"></a>dstShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p5841153810116"><a name="p5841153810116"></a><a name="p5841153810116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p8842338141112"><a name="p8842338141112"></a><a name="p8842338141112"></a>输出tensor的shape：uint32_t类型的数组，长度取值范围为[1, 9]。输入/输出的shape维度数目必须一致，且满足条件dstShape[i] &gt;= srcShape[i]。</p>
</td>
</tr>
<tr id="row17842538121120"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p684215381113"><a name="p684215381113"></a><a name="p684215381113"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p14842143812119"><a name="p14842143812119"></a><a name="p14842143812119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p5842123821115"><a name="p5842123821115"></a><a name="p5842123821115"></a>输入tensor的shape：uint32_t类型的数组，长度取值范围为[1, 9]。输入/输出的shape维度数目必须一致，且满足条件dstShape[i] &gt;= srcShape[i]。</p>
<p id="p1484223891114"><a name="p1484223891114"></a><a name="p1484223891114"></a>当srcShape[i]的值为1，且dstShape[i]不等于srcShape[i]时，表示i轴为广播轴。</p>
</td>
</tr>
<tr id="row884213386116"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p38422038111111"><a name="p38422038111111"></a><a name="p38422038111111"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.23%" headers="mcps1.2.4.1.2 "><p id="p9842638101116"><a name="p9842638101116"></a><a name="p9842638101116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p084293812119"><a name="p084293812119"></a><a name="p084293812119"></a>Broadcast接口所需的Tiling信息。BroadcastTiling*类型，通过调用Kernel侧的tiling计算接口<span>GetBroadcastTilingInfo</span>获取。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  kernel侧tiling计算接口参数说明

<a name="table5458981523"></a>
<table><thead align="left"><tr id="row12458128725"><th class="cellrowborder" valign="top" width="19.15%" id="mcps1.2.4.1.1"><p id="p154586819217"><a name="p154586819217"></a><a name="p154586819217"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.01%" id="mcps1.2.4.1.2"><p id="p10249113413913"><a name="p10249113413913"></a><a name="p10249113413913"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.84%" id="mcps1.2.4.1.3"><p id="p1145948424"><a name="p1145948424"></a><a name="p1145948424"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row15459581220"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="p204591189220"><a name="p204591189220"></a><a name="p204591189220"></a>rank</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p9249634096"><a name="p9249634096"></a><a name="p9249634096"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.84%" headers="mcps1.2.4.1.3 "><p id="p1245978827"><a name="p1245978827"></a><a name="p1245978827"></a>输入/输出tensor的维度数目，目前支持的取值为[1, 9]。</p>
</td>
</tr>
<tr id="row1745913810211"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="p64594811216"><a name="p64594811216"></a><a name="p64594811216"></a>dstShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p32491534693"><a name="p32491534693"></a><a name="p32491534693"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.84%" headers="mcps1.2.4.1.3 "><p id="p1459138227"><a name="p1459138227"></a><a name="p1459138227"></a>输出tensor的shape：uint32_t类型的数组，长度取值范围为[1, 9]。输入/输出的shape维度数目必须一致，且满足条件dstShape[i] &gt;= srcShape[i]。</p>
</td>
</tr>
<tr id="row3459581528"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="p045978929"><a name="p045978929"></a><a name="p045978929"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p122495341296"><a name="p122495341296"></a><a name="p122495341296"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.84%" headers="mcps1.2.4.1.3 "><p id="p59514711295"><a name="p59514711295"></a><a name="p59514711295"></a>输入tensor的shape：uint32_t类型的数组，长度取值范围为[1, 9]。输入/输出的shape维度数目必须一致，且满足条件dstShape[i] &gt;= srcShape[i]。</p>
<p id="p59564792913"><a name="p59564792913"></a><a name="p59564792913"></a>当srcShape[i]的值为1，且dstShape[i]不等于srcShape[i]时，表示i轴为广播轴。</p>
</td>
</tr>
<tr id="row3459128625"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="p124590810217"><a name="p124590810217"></a><a name="p124590810217"></a><span>srcInnerPad</span></p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p13249143418911"><a name="p13249143418911"></a><a name="p13249143418911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.84%" headers="mcps1.2.4.1.3 "><p id="p12330522103013"><a name="p12330522103013"></a><a name="p12330522103013"></a>表示输入的最后一维srcShape[rank-1]是否32B对齐。</p>
<p id="p184601381720"><a name="p184601381720"></a><a name="p184601381720"></a>当前仅支持取值为false。</p>
</td>
</tr>
<tr id="row145813442510"><td class="cellrowborder" valign="top" width="19.15%" headers="mcps1.2.4.1.1 "><p id="p18811749254"><a name="p18811749254"></a><a name="p18811749254"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.01%" headers="mcps1.2.4.1.2 "><p id="p192501349912"><a name="p192501349912"></a><a name="p192501349912"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.84%" headers="mcps1.2.4.1.3 "><p id="p138812491253"><a name="p138812491253"></a><a name="p138812491253"></a>计算返回的Tiling信息。BroadcastTiling&amp;类型。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section11852175575912"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   **不支持源操作数与目的操作数地址重叠。**
-   当前仅支持ND格式的输入，不支持其他格式。
-   dim目前仅支持1或者2， axis目前仅支持0或者1。
-   对于Ascend 950PR/Ascend 950DT，输入/输出tensor支持的维度数目，即rank支持的取值范围为\[1, 9\]。

## 调用示例<a name="section208521655195918"></a>

```
#include "kernel_operator.h"

template <typename T, int32_t dim, int32_t axis>
class KernelBroadcast {
public:
    __aicore__ inline KernelBroadcast()
    {}
    __aicore__ inline void Init(
        GM_ADDR srcGm, GM_ADDR dstGm, const uint32_t dstShape[dim], const uint32_t srcShape[dim])
    {
        for (uint32_t i = 0; i < dim; i++) {
            srcSize *= srcShape[i];
            dstSize *= dstShape[i];
        }
        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(srcGm), srcSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dstGm), dstSize);

        pipe.InitBuffer(inQueueX, 1, srcSize * sizeof(T));
        pipe.InitBuffer(outQueue, 1, dstSize * sizeof(T));
        dstShape_ = dstShape;
        srcShape_ = srcShape;
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, srcSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> srcLocal = inQueueX.DeQue<T>();
        AscendC::Broadcast<T, dim, axis>(dstLocal, srcLocal, dstShape_, srcShape_);

        outQueue.EnQue<T>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, dstSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    const uint32_t *dstShape_{nullptr};
    const uint32_t *srcShape_{nullptr};
    int32_t srcSize{1};
    int32_t dstSize{1};
};

template <typename T, int32_t dim, int32_t axis>
__aicore__ void kernel_broadcast_operator(
    GM_ADDR srcGm, GM_ADDR dstGm, const uint32_t dstShape[dim], const uint32_t srcShape[dim])
{
    KernelBroadcast<T, dim, axis> op;
    op.Init(srcGm, dstGm, dstShape, srcShape);
    op.Process();
}
```

```
AscendC::BroadcastTiling tiling;
AscendC::GetBroadcastTilingInfo<T>(dim, dstShape_, srcShape_, false, tiling);
AscendC::Broadcast<T>(dstLocal, srcLocal, dstShape_, srcShape_, &tiling);
```

结果示例如下：

```
输入数据（srcLocal）: 
[[ 1]
 [ 2]
 [ 3]
 [ 4]
 [ 5]
 [ 6]
 [ 7]
 [ 8]
 [ 9]
 [10]
 [11]
 [12]
 [13]
 [14]
 [15]
 [16]]
dim：2
axis：1
输出数据（dstLocal）: 
[[ 1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1]
 [ 2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2]
 [ 3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3]
 [ 4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4]
 [ 5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5]
 [ 6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6]
 [ 7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7]
 [ 8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8]
 [ 9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9]
 [10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10]
 [11 11 11 11 11 11 11 11 11 11 11 11 11 11 11 11]
 [12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12]
 [13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13]
 [14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14]
 [15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15]
 [16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16]]
```

