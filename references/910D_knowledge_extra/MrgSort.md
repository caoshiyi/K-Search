# MrgSort<a name="ZH-CN_TOPIC_0000002523303810"></a>

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

## 功能说明<a name="section618mcpsimp"></a>

将已经排好序的最多4条队列，合并排列成1条队列，结果按照score域由大到小排序。

MrgSort指令处理的数据一般是经过Sort32指令处理后的数据，也就是Sort32指令的输出，队列的结构如下所示：

-   数据类型为float，每个结构占据8Bytes。

    <!-- img2text -->
```
┌────────────────────┬────────────────────┐
│      score[0]      │      index[0]      │
├────────────────────┼────────────────────┤
│      score[1]      │      index[1]      │
├────────────────────┼────────────────────┤
│      score[2]      │      index[2]      │
├────────────────────┼────────────────────┤
│      score[3]      │      index[3]      │
├────────────────────┼────────────────────┤
│      score[4]      │      index[4]      │
└────────────────────┴────────────────────┘
        4Bytes               4Bytes
```

-   数据类型为half，每个结构也占据8Bytes，中间有2Bytes保留。

    <!-- img2text -->
```
┌──────────┬──────────┬──────────────┐
│ score[0] │ reserved │   index[0]   │
├──────────┼──────────┼──────────────┤
│ score[1] │ reserved │   index[1]   │
├──────────┼──────────┼──────────────┤
│ score[2] │ reserved │   index[2]   │
├──────────┼──────────┼──────────────┤
│ score[3] │ reserved │   index[3]   │
├──────────┼──────────┼──────────────┤
│ score[4] │ reserved │   index[4]   │
└──────────┴──────────┴──────────────┘
   2Bytes     2Bytes      4Bytes
```

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void MrgSort(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.780000000000001%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.22%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.780000000000001%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.22%" headers="mcps1.2.3.1.2 "><p id="p16592838204215"><a name="p16592838204215"></a><a name="p16592838204215"></a><span id="ph3592143814216"><a name="ph3592143814216"></a><a name="ph3592143814216"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table1196021819497"></a>
<table><thead align="left"><tr id="row19606188492"><th class="cellrowborder" valign="top" width="13.631363136313631%" id="mcps1.2.4.1.1"><p id="p5960318164918"><a name="p5960318164918"></a><a name="p5960318164918"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.61126112611261%" id="mcps1.2.4.1.2"><p id="p129609185493"><a name="p129609185493"></a><a name="p129609185493"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.75737573757377%" id="mcps1.2.4.1.3"><p id="p149601218134915"><a name="p149601218134915"></a><a name="p149601218134915"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1496012182493"><td class="cellrowborder" valign="top" width="13.631363136313631%" headers="mcps1.2.4.1.1 "><p id="p39605185493"><a name="p39605185493"></a><a name="p39605185493"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.61126112611261%" headers="mcps1.2.4.1.2 "><p id="p179601218134912"><a name="p179601218134912"></a><a name="p179601218134912"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p11806624101513"><a name="p11806624101513"></a><a name="p11806624101513"></a>目的操作数，存储经过排序后的数据。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p194546161418"><a name="p194546161418"></a><a name="p194546161418"></a><span id="ph443416246237"><a name="ph443416246237"></a><a name="ph443416246237"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row796116186493"><td class="cellrowborder" valign="top" width="13.631363136313631%" headers="mcps1.2.4.1.1 "><p id="p29611818154918"><a name="p29611818154918"></a><a name="p29611818154918"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.61126112611261%" headers="mcps1.2.4.1.2 "><p id="p396121816492"><a name="p396121816492"></a><a name="p396121816492"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p571917113912"><a name="p571917113912"></a><a name="p571917113912"></a>源操作数，4个队列，并且每个队列都已经排好序，类型为MrgSortSrcList结构体，定义如下：</p>
<a name="screen1546714181992"></a><a name="screen1546714181992"></a><pre class="screen" codetype="Cpp" id="screen1546714181992">template &lt;typename T&gt; struct MrgSortSrcList {
    __aicore__ MrgSortSrcList() {}
    __aicore__ MrgSortSrcList(const LocalTensor&lt;T&gt;&amp; src1In, const LocalTensor&lt;T&gt;&amp; src2In, const LocalTensor&lt;T&gt;&amp; src3In,
        const LocalTensor&lt;T&gt;&amp; src4In)
    {
        src1 = src1In[0];
        src2 = src2In[0];
        src3 = src3In[0];
        src4 = src4In[0];
    }
    LocalTensor&lt;T&gt; src1; // 第一个已经排好序的队列
    LocalTensor&lt;T&gt; src2; // 第二个已经排好序的队列
    LocalTensor&lt;T&gt; src3; // 第三个已经排好序的队列
    LocalTensor&lt;T&gt; src4; // 第四个已经排好序的队列
};</pre>
<p id="p8114998103"><a name="p8114998103"></a><a name="p8114998103"></a>源操作数的数据类型与目的操作数保持一致。src1、src2、src3、src4类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。<span id="ph441646171420"><a name="ph441646171420"></a><a name="ph441646171420"></a>LocalTensor的起始地址需要8字节对齐。</span></p>
</td>
</tr>
<tr id="row1096110183494"><td class="cellrowborder" valign="top" width="13.631363136313631%" headers="mcps1.2.4.1.1 "><p id="p139612184497"><a name="p139612184497"></a><a name="p139612184497"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="12.61126112611261%" headers="mcps1.2.4.1.2 "><p id="p8961141811498"><a name="p8961141811498"></a><a name="p8961141811498"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p11323173012104"><a name="p11323173012104"></a><a name="p11323173012104"></a>排序所需参数，类型为MrgSort4Info结构体。</p>
<p id="p18834113201011"><a name="p18834113201011"></a><a name="p18834113201011"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_proposal.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p4961151813492"><a name="p4961151813492"></a><a name="p4961151813492"></a>参数说明请参考<a href="#table7515358184615">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  MrgSort4Info参数说明

<a name="table7515358184615"></a>
<table><thead align="left"><tr id="row1951513585460"><th class="cellrowborder" valign="top" width="14.360000000000001%" id="mcps1.2.3.1.1"><p id="p195151858154612"><a name="p195151858154612"></a><a name="p195151858154612"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85.64%" id="mcps1.2.3.1.2"><p id="p1151555820466"><a name="p1151555820466"></a><a name="p1151555820466"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row25157580465"><td class="cellrowborder" valign="top" width="14.360000000000001%" headers="mcps1.2.3.1.1 "><p id="p05156585462"><a name="p05156585462"></a><a name="p05156585462"></a>elementLengths</p>
</td>
<td class="cellrowborder" valign="top" width="85.64%" headers="mcps1.2.3.1.2 "><p id="p1751545874619"><a name="p1751545874619"></a><a name="p1751545874619"></a>四个源队列的长度（8Bytes结构的数目），类型为长度为4的uint16_t数据类型的数组，理论上每个元素取值范围[0, 4095]，但不能超出UB的存储空间。</p>
</td>
</tr>
<tr id="row1515155817465"><td class="cellrowborder" valign="top" width="14.360000000000001%" headers="mcps1.2.3.1.1 "><p id="p55154583465"><a name="p55154583465"></a><a name="p55154583465"></a>ifExhaustedSuspension</p>
</td>
<td class="cellrowborder" valign="top" width="85.64%" headers="mcps1.2.3.1.2 "><p id="p135151558194611"><a name="p135151558194611"></a><a name="p135151558194611"></a>某条队列耗尽后，指令是否需要停止，类型为bool，默认false。</p>
</td>
</tr>
<tr id="row1234542411541"><td class="cellrowborder" valign="top" width="14.360000000000001%" headers="mcps1.2.3.1.1 "><p id="p143461024185414"><a name="p143461024185414"></a><a name="p143461024185414"></a>validBit</p>
</td>
<td class="cellrowborder" valign="top" width="85.64%" headers="mcps1.2.3.1.2 "><div class="p" id="p4633153935413"><a name="p4633153935413"></a><a name="p4633153935413"></a>有效队列个数，取值如下：<a name="ul47411933141413"></a><a name="ul47411933141413"></a><ul id="ul47411933141413"><li>3：前两条队列有效</li><li>7：前三条队列有效</li><li>15：四条队列全部有效</li></ul>
</div>
</td>
</tr>
<tr id="row59828263546"><td class="cellrowborder" valign="top" width="14.360000000000001%" headers="mcps1.2.3.1.1 "><p id="p898232615410"><a name="p898232615410"></a><a name="p898232615410"></a>repeatTimes</p>
</td>
<td class="cellrowborder" valign="top" width="85.64%" headers="mcps1.2.3.1.2 "><p id="p330857175415"><a name="p330857175415"></a><a name="p330857175415"></a>迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围：repeatTimes∈[1,255]。</p>
<div class="p" id="p1212745155714"><a name="p1212745155714"></a><a name="p1212745155714"></a>repeatTimes参数生效是有条件的，需要同时满足以下四个条件：<a name="ul2212945195714"></a><a name="ul2212945195714"></a><ul id="ul2212945195714"><li>src包含四条队列并且validBit=15</li><li>四个源队列的长度一致</li><li>四个源队列连续存储</li><li>ifExhaustedSuspension = False</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   当存在score\[i\]与score\[j\]相同时，如果i\>j，则score\[j\]将首先被选出来，排在前面。
-   每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
-   需要注意此函数排序的队列非region proposal结构。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

-   接口使用样例

    ```
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(float));
    AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    // 对8个已排好序的队列进行合并排序，repeatTimes = 2，数据连续存放
    // 每个队列包含32个(score,index)的8Bytes结构
    // 最后输出对score域的256个数完成排序后的结果
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = 32;
    params.elementLengths[1] = 32;
    params.elementLengths[2] = 32;
    params.elementLengths[3] = 32;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b1111;
    params.repeatTimes = 2;
    
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = workLocal[0];
    srcList.src2 = workLocal[64]; // workLocal为float类型，每个队列占据256Bytes空间
    srcList.src3 = workLocal[128];
    srcList.src4 = workLocal[192];
    
    AscendC::MrgSort<float>(dstLocal, srcList, params);
    outQueueDst.EnQue<float>(dstLocal);
    outQueueDst.FreeTensor(dstLocal);
    ```

