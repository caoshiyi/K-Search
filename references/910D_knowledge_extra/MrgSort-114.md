# MrgSort<a name="ZH-CN_TOPIC_0000002554423783"></a>

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

将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序，排布方式如下：

Ascend 950PR/Ascend 950DT采用方式一。

-   排布方式一：

    MrgSort处理的数据一般是经过Sort处理后的数据，也就是Sort接口的输出，队列的结构如下所示：

    -   数据类型为float，每个结构占据8Bytes。

        <!-- img2text -->
```
┌───────────────┬───────────────┐
│   score[0]    │   index[0]    │
├───────────────┼───────────────┤
│   score[1]    │   index[1]    │
├───────────────┼───────────────┤
│   score[2]    │   index[2]    │
├───────────────┼───────────────┤
│   score[3]    │   index[3]    │
├───────────────┼───────────────┤
│   score[4]    │   index[4]    │
└───────────────┴───────────────┘
     4Bytes          4Bytes
```

    -   数据类型为half，每个结构也占据8Bytes，中间有2Bytes保留。

        <!-- img2text -->
```
┌──────────┬──────────┬──────────┐
│ score[0] │ reserved │ index[0] │
├──────────┼──────────┼──────────┤
│ score[1] │ reserved │ index[1] │
├──────────┼──────────┼──────────┤
│ score[2] │ reserved │ index[2] │
├──────────┼──────────┼──────────┤
│ score[3] │ reserved │ index[3] │
├──────────┼──────────┼──────────┤
│ score[4] │ reserved │ index[4] │
└──────────┴──────────┴──────────┘
  2Bytes     2Bytes     4Bytes
```

-   排布方式二：Region Proposal排布

    输入输出数据均为Region Proposal，具体请参见[Sort](Sort.md)中的排布方式二。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, bool isExhaustedSuspension = false>
__aicore__ inline void MrgSort(const LocalTensor<T> &dst, const MrgSortSrcList<T> &sortList, const uint16_t elementCountList[4], uint32_t sortedNum[4], uint16_t validBit, const int32_t repeatTime)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>操作数的数据类型。</p>
<p id="p5315184745513"><a name="p5315184745513"></a><a name="p5315184745513"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row1623812985111"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p152385297515"><a name="p152385297515"></a><a name="p152385297515"></a>isExhaustedSuspension</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p1189955864"><a name="p1189955864"></a><a name="p1189955864"></a>某条队列耗尽（即该队列已经全部排序到目的操作数）后，是否需要停止合并。类型为bool，参数取值如下：</p>
<a name="ul28710018710"></a><a name="ul28710018710"></a><ul id="ul28710018710"><li>false：直到所有队列耗尽完才停止合并。</li><li>true：某条队列耗尽后，停止合并。</li></ul>
<p id="p6598143620537"><a name="p6598143620537"></a><a name="p6598143620537"></a>默认值为false。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table1196021819497"></a>
<table><thead align="left"><tr id="row19606188492"><th class="cellrowborder" valign="top" width="13.651365136513652%" id="mcps1.2.4.1.1"><p id="p5960318164918"><a name="p5960318164918"></a><a name="p5960318164918"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.421242124212421%" id="mcps1.2.4.1.2"><p id="p129609185493"><a name="p129609185493"></a><a name="p129609185493"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.92739273927393%" id="mcps1.2.4.1.3"><p id="p149601218134915"><a name="p149601218134915"></a><a name="p149601218134915"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1496012182493"><td class="cellrowborder" valign="top" width="13.651365136513652%" headers="mcps1.2.4.1.1 "><p id="p39605185493"><a name="p39605185493"></a><a name="p39605185493"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.421242124212421%" headers="mcps1.2.4.1.2 "><p id="p179601218134912"><a name="p179601218134912"></a><a name="p179601218134912"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.92739273927393%" headers="mcps1.2.4.1.3 "><p id="p11806624101513"><a name="p11806624101513"></a><a name="p11806624101513"></a>目的操作数，存储经过排序后的数据。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row796116186493"><td class="cellrowborder" valign="top" width="13.651365136513652%" headers="mcps1.2.4.1.1 "><p id="p29611818154918"><a name="p29611818154918"></a><a name="p29611818154918"></a>sortList</p>
</td>
<td class="cellrowborder" valign="top" width="12.421242124212421%" headers="mcps1.2.4.1.2 "><p id="p396121816492"><a name="p396121816492"></a><a name="p396121816492"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.92739273927393%" headers="mcps1.2.4.1.3 "><p id="p12961918204918"><a name="p12961918204918"></a><a name="p12961918204918"></a>源操作数，支持2-4个队列，并且每个队列都已经排好序，类型为MrgSortSrcList结构体，具体请参考<a href="#table16792555114320">表3</a>。MrgSortSrcList中传入要合并的队列。</p>
<a name="screen92612599281"></a><a name="screen92612599281"></a><pre class="screen" codetype="Cpp" id="screen92612599281">template &lt;typename T&gt;
struct MrgSortSrcList {
    LocalTensor&lt;T&gt; src1;
    LocalTensor&lt;T&gt; src2;
    LocalTensor&lt;T&gt; src3; // 当要合并的队列个数小于3，可以为空tensor
    LocalTensor&lt;T&gt; src4; // 当要合并的队列个数小于4，可以为空tensor
};</pre>
</td>
</tr>
<tr id="row1096110183494"><td class="cellrowborder" valign="top" width="13.651365136513652%" headers="mcps1.2.4.1.1 "><p id="p139612184497"><a name="p139612184497"></a><a name="p139612184497"></a>elementCountList</p>
</td>
<td class="cellrowborder" valign="top" width="12.421242124212421%" headers="mcps1.2.4.1.2 "><p id="p8961141811498"><a name="p8961141811498"></a><a name="p8961141811498"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.92739273927393%" headers="mcps1.2.4.1.3 "><p id="p4961151813492"><a name="p4961151813492"></a><a name="p4961151813492"></a>四个源队列的长度（排序方式一：8Bytes结构的数目，排序方式二：16*sizeof(T)Bytes结构的数目），类型为长度为4的uint16_t数据类型的数组，理论上每个元素取值范围[0, 4095]，但不能超出UB的存储空间。</p>
</td>
</tr>
<tr id="row0990719173911"><td class="cellrowborder" valign="top" width="13.651365136513652%" headers="mcps1.2.4.1.1 "><p id="p14990131903911"><a name="p14990131903911"></a><a name="p14990131903911"></a>sortedNum</p>
</td>
<td class="cellrowborder" valign="top" width="12.421242124212421%" headers="mcps1.2.4.1.2 "><p id="p169911019193915"><a name="p169911019193915"></a><a name="p169911019193915"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.92739273927393%" headers="mcps1.2.4.1.3 "><p id="p7991161915399"><a name="p7991161915399"></a><a name="p7991161915399"></a>耗尽模式下（即isExhaustedSuspension为true时），停止合并时每个队列已排序的元素个数。</p>
</td>
</tr>
<tr id="row4921426163918"><td class="cellrowborder" valign="top" width="13.651365136513652%" headers="mcps1.2.4.1.1 "><p id="p192172683920"><a name="p192172683920"></a><a name="p192172683920"></a>validBit</p>
</td>
<td class="cellrowborder" valign="top" width="12.421242124212421%" headers="mcps1.2.4.1.2 "><p id="p1811084613917"><a name="p1811084613917"></a><a name="p1811084613917"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.92739273927393%" headers="mcps1.2.4.1.3 "><div class="p" id="p127511927174115"><a name="p127511927174115"></a><a name="p127511927174115"></a>有效队列个数，取值如下：<a name="ul175182784117"></a><a name="ul175182784117"></a><ul id="ul175182784117"><li>0b11：前两条队列有效</li><li>0b111：前三条队列有效</li><li>0b1111：四条队列全部有效</li></ul>
</div>
</td>
</tr>
<tr id="row8722424133912"><td class="cellrowborder" valign="top" width="13.651365136513652%" headers="mcps1.2.4.1.1 "><p id="p17231324163915"><a name="p17231324163915"></a><a name="p17231324163915"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.421242124212421%" headers="mcps1.2.4.1.2 "><p id="p19723162420398"><a name="p19723162420398"></a><a name="p19723162420398"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.92739273927393%" headers="mcps1.2.4.1.3 "><p id="p14794152154220"><a name="p14794152154220"></a><a name="p14794152154220"></a>迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围：repeatTime∈[1,255]。</p>
<div class="p" id="p1179419244213"><a name="p1179419244213"></a><a name="p1179419244213"></a>repeatTime参数生效是有条件的，需要同时满足以下四个条件：<a name="ul1879420254213"></a><a name="ul1879420254213"></a><ul id="ul1879420254213"><li>srcLocal包含四条队列并且validBit=15</li><li>四个源队列的长度一致</li><li>四个源队列连续存储</li><li>isExhaustedSuspension为false</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

**表 3**  MrgSortSrcList参数说明

<a name="table16792555114320"></a>
<table><thead align="left"><tr id="row147921255164312"><th class="cellrowborder" valign="top" width="11.881188118811883%" id="mcps1.2.4.1.1"><p id="p10792155584319"><a name="p10792155584319"></a><a name="p10792155584319"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.241224122412241%" id="mcps1.2.4.1.2"><p id="p179235517432"><a name="p179235517432"></a><a name="p179235517432"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.87758775877587%" id="mcps1.2.4.1.3"><p id="p17792155511436"><a name="p17792155511436"></a><a name="p17792155511436"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row779215516437"><td class="cellrowborder" valign="top" width="11.881188118811883%" headers="mcps1.2.4.1.1 "><p id="p176952594455"><a name="p176952594455"></a><a name="p176952594455"></a>src1</p>
</td>
<td class="cellrowborder" valign="top" width="12.241224122412241%" headers="mcps1.2.4.1.2 "><p id="p20553145411469"><a name="p20553145411469"></a><a name="p20553145411469"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.87758775877587%" headers="mcps1.2.4.1.3 "><p id="p12792114310153"><a name="p12792114310153"></a><a name="p12792114310153"></a>源操作数，第一个已经排好序的队列。</p>
<p id="p1812010505151"><a name="p1812010505151"></a><a name="p1812010505151"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p4378240162818"><a name="p4378240162818"></a><a name="p4378240162818"></a>数据类型与目的操作数保持一致。</p>
<p id="p5411161326"><a name="p5411161326"></a><a name="p5411161326"></a><span id="ph1041119613321"><a name="ph1041119613321"></a><a name="ph1041119613321"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row1795413574612"><td class="cellrowborder" valign="top" width="11.881188118811883%" headers="mcps1.2.4.1.1 "><p id="p19954145154612"><a name="p19954145154612"></a><a name="p19954145154612"></a>src2</p>
</td>
<td class="cellrowborder" valign="top" width="12.241224122412241%" headers="mcps1.2.4.1.2 "><p id="p695414524617"><a name="p695414524617"></a><a name="p695414524617"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.87758775877587%" headers="mcps1.2.4.1.3 "><p id="p4706011151616"><a name="p4706011151616"></a><a name="p4706011151616"></a>源操作数，第二个已经排好序的队列。</p>
<p id="p943601517169"><a name="p943601517169"></a><a name="p943601517169"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1695415594619"><a name="p1695415594619"></a><a name="p1695415594619"></a>数据类型与目的操作数保持一致。</p>
<p id="p1240111153211"><a name="p1240111153211"></a><a name="p1240111153211"></a><span id="ph740141114325"><a name="ph740141114325"></a><a name="ph740141114325"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row109816294617"><td class="cellrowborder" valign="top" width="11.881188118811883%" headers="mcps1.2.4.1.1 "><p id="p19812213467"><a name="p19812213467"></a><a name="p19812213467"></a>src3</p>
</td>
<td class="cellrowborder" valign="top" width="12.241224122412241%" headers="mcps1.2.4.1.2 "><p id="p498172164619"><a name="p498172164619"></a><a name="p498172164619"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.87758775877587%" headers="mcps1.2.4.1.3 "><p id="p69171723151613"><a name="p69171723151613"></a><a name="p69171723151613"></a>源操作数，第三个已经排好序的队列。</p>
<p id="p28591626141613"><a name="p28591626141613"></a><a name="p28591626141613"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1498118211462"><a name="p1498118211462"></a><a name="p1498118211462"></a>数据类型与目的操作数保持一致。</p>
<p id="p11182171513216"><a name="p11182171513216"></a><a name="p11182171513216"></a><span id="ph12182715153215"><a name="ph12182715153215"></a><a name="ph12182715153215"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row6412145019474"><td class="cellrowborder" valign="top" width="11.881188118811883%" headers="mcps1.2.4.1.1 "><p id="p114121550154712"><a name="p114121550154712"></a><a name="p114121550154712"></a>src4</p>
</td>
<td class="cellrowborder" valign="top" width="12.241224122412241%" headers="mcps1.2.4.1.2 "><p id="p4412135044712"><a name="p4412135044712"></a><a name="p4412135044712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.87758775877587%" headers="mcps1.2.4.1.3 "><p id="p2933534181610"><a name="p2933534181610"></a><a name="p2933534181610"></a>源操作数，第四个已经排好序的队列。</p>
<p id="p1356673741610"><a name="p1356673741610"></a><a name="p1356673741610"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p54122050124710"><a name="p54122050124710"></a><a name="p54122050124710"></a>数据类型与目的操作数保持一致。</p>
<p id="p934832017323"><a name="p934832017323"></a><a name="p934832017323"></a><span id="ph1834862013325"><a name="ph1834862013325"></a><a name="ph1834862013325"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   当存在score\[i\]与score\[j\]相同时，如果i\>j，则score\[j\]将首先被选出来，排在前面，即index的顺序与输入顺序一致。
-   每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

-   处理128个half类型数据。

    该样例适用于：

    Ascend 950PR/Ascend 950DT

    ```
    #include "kernel_operator.h"
    template <typename T>
    class FullSort
    {
    public:
        __aicore__ inline FullSort() {}
        __aicore__ inline void Init(__gm__ uint8_t *srcValueGm, __gm__ uint8_t *srcIndexGm, __gm__ uint8_t *dstValueGm, __gm__ uint8_t *dstIndexGm)
        {
            concatRepeatTimes = elementCount / 16;
            inBufferSize = elementCount * sizeof(uint32_t);
            outBufferSize = elementCount * sizeof(uint32_t);
            calcBufferSize = elementCount * 8;
            tmpBufferSize = elementCount * 8;
            sortedLocalSize = elementCount * 4;
            sortRepeatTimes = elementCount / 32;
            extractRepeatTimes = elementCount / 32;
            sortTmpLocalSize = elementCount * 4;
            valueGlobal.SetGlobalBuffer((__gm__ T *)srcValueGm);
            indexGlobal.SetGlobalBuffer((__gm__ uint32_t *)srcIndexGm);
            dstValueGlobal.SetGlobalBuffer((__gm__ T *)dstValueGm);
            dstIndexGlobal.SetGlobalBuffer((__gm__ uint32_t *)dstIndexGm);
            pipe.InitBuffer(queIn, 2, inBufferSize);
            pipe.InitBuffer(queOut, 2, outBufferSize);
            pipe.InitBuffer(queCalc, 1, calcBufferSize * sizeof(T));
            pipe.InitBuffer(queTmp, 2, tmpBufferSize * sizeof(T));
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
            AscendC::LocalTensor<T> valueLocal = queIn.AllocTensor<T>();
            AscendC::DataCopy(valueLocal, valueGlobal, elementCount);
            queIn.EnQue(valueLocal);
            AscendC::LocalTensor<uint32_t> indexLocal = queIn.AllocTensor<uint32_t>();
            AscendC::DataCopy(indexLocal, indexGlobal, elementCount);
            queIn.EnQue(indexLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<T> valueLocal = queIn.DeQue<T>();
            AscendC::LocalTensor<uint32_t> indexLocal = queIn.DeQue<uint32_t>();
            AscendC::LocalTensor<T> sortedLocal = queCalc.AllocTensor<T>();
            AscendC::LocalTensor<T> concatTmpLocal = queTmp.AllocTensor<T>();
            AscendC::LocalTensor<T> sortTmpLocal = queTmp.AllocTensor<T>();
            AscendC::LocalTensor<T> dstValueLocal = queOut.AllocTensor<T>();
            AscendC::LocalTensor<uint32_t> dstIndexLocal = queOut.AllocTensor<uint32_t>();
            AscendC::LocalTensor<T> concatLocal;
    
            AscendC::Concat(concatLocal, valueLocal, concatTmpLocal, concatRepeatTimes);
            AscendC::Sort<T, false>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, sortRepeatTimes);
            uint32_t singleMergeTmpElementCount = elementCount / 4;
            uint32_t baseOffset = AscendC::GetSortOffset<T>(singleMergeTmpElementCount);
            AscendC::MrgSortSrcList sortList = AscendC::MrgSortSrcList(sortedLocal[0], sortedLocal[baseOffset], sortedLocal[2 * baseOffset], sortedLocal[3 * baseOffset]);
            uint16_t singleDataSize = elementCount / 4;
            const uint16_t elementCountList[4] = {singleDataSize, singleDataSize, singleDataSize, singleDataSize};
            uint32_t sortedNum[4];
            AscendC::MrgSort<T, false>(sortTmpLocal, sortList, elementCountList, sortedNum, 0b1111, 1);
            AscendC::Extract(dstValueLocal, dstIndexLocal, sortTmpLocal, extractRepeatTimes);
    
            queTmp.FreeTensor(concatTmpLocal);
            queTmp.FreeTensor(sortTmpLocal);
            queIn.FreeTensor(valueLocal);
            queIn.FreeTensor(indexLocal);
            queCalc.FreeTensor(sortedLocal);
            queOut.EnQue(dstValueLocal);
            queOut.EnQue(dstIndexLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<T> dstValueLocal = queOut.DeQue<T>();
            AscendC::LocalTensor<uint32_t> dstIndexLocal = queOut.DeQue<uint32_t>();
            AscendC::DataCopy(dstValueGlobal, dstValueLocal, elementCount);
            AscendC::DataCopy(dstIndexGlobal, dstIndexLocal, elementCount);
            queOut.FreeTensor(dstValueLocal);
            queOut.FreeTensor(dstIndexLocal);
        }
    
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 2> queIn;
        AscendC::TQue<AscendC::TPosition::VECOUT, 2> queOut;
        AscendC::TQue<AscendC::TPosition::VECIN, 2> queTmp;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> queCalc;
        AscendC::GlobalTensor<T> valueGlobal;
        AscendC::GlobalTensor<uint32_t> indexGlobal;
        AscendC::GlobalTensor<T> dstValueGlobal;
        AscendC::GlobalTensor<uint32_t> dstIndexGlobal;
        uint32_t elementCount = 128;
        uint32_t concatRepeatTimes;
        uint32_t inBufferSize;
        uint32_t outBufferSize;
        uint32_t calcBufferSize;
        uint32_t tmpBufferSize;
        uint32_t sortedLocalSize;
        uint32_t sortTmpLocalSize;
        uint32_t sortRepeatTimes;
        uint32_t extractRepeatTimes;
    };
    
    extern "C" __global__ __aicore__ void sort_operator(__gm__ uint8_t *src0Gm, __gm__ uint8_t *src1Gm, __gm__ uint8_t *dst0Gm, __gm__ uint8_t *dst1Gm)
    {
        FullSort<half> op;
        op.Init(src0Gm, src1Gm, dst0Gm, dst1Gm);
        op.Process();
    }
    ```

    ```
    示例结果
    输入数据(srcValueGm): 128个float类型数据
    [31 30 29 ... 2 1 0
     63 62 61 ... 34 33 32
     95 94 93 ... 66 65 64
     127 126 125 ... 98 97 96]
    输入数据(srcIndexGm):
    [31 30 29 ... 2 1 0
     63 62 61 ... 34 33 32
     95 94 93 ... 66 65 64
     127 126 125 ... 98 97 96]
    输出数据(dstValueGm):
    [127 126 125 ... 2 1 0]
    输出数据(dstIndexGm):
    [127 126 125 ... 2 1 0]
    ```

