# MrgSort4<a name="ZH-CN_TOPIC_0000002554424617"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将已经排好序的最多4条Region Proposals队列，排列并合并成1条队列，结果按照score域由大到小排序。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void MrgSort4(const LocalTensor<T>& dst, const MrgSortSrcList<T>& src, const MrgSort4Info& params)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="13.52%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.48%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="13.52%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.48%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.58125812581258%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.75737573757377%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19576531173410"><a name="p19576531173410"></a><a name="p19576531173410"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p16576163119347"><a name="p16576163119347"></a><a name="p16576163119347"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p88116136127"><a name="p88116136127"></a><a name="p88116136127"></a>目的操作数，存储经过排序后的Region Proposals。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1871313861718"><a name="p1871313861718"></a><a name="p1871313861718"></a>LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p165761231123417"><a name="p165761231123417"></a><a name="p165761231123417"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p757693163410"><a name="p757693163410"></a><a name="p757693163410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p527920812248"><a name="p527920812248"></a><a name="p527920812248"></a>源操作数，4个Region Proposals队列，并且每个Region Proposal队列都已经排好序，类型为MrgSortSrcList结构体，具体定义如下：</p>
<a name="screen17521147102017"></a><a name="screen17521147102017"></a><pre class="screen" codetype="Cpp" id="screen17521147102017">template &lt;typename T&gt; struct MrgSortSrcList {
    __aicore__ MrgSortSrcList() {}
    __aicore__ MrgSortSrcList(const LocalTensor&lt;T&gt;&amp; src1In, const LocalTensor&lt;T&gt;&amp; src2In, const LocalTensor&lt;T&gt;&amp; src3In,
        const LocalTensor&lt;T&gt;&amp; src4In)
    {
        src1 = src1In[0];
        src2 = src2In[0];
        src3 = src3In[0];
        src4 = src4In[0];
    }
    LocalTensor&lt;T&gt; src1; // 第一个已经排好序的Region Proposals队列
    LocalTensor&lt;T&gt; src2; // 第二个已经排好序的Region Proposals队列
    LocalTensor&lt;T&gt; src3; // 第三个已经排好序的Region Proposals队列
    LocalTensor&lt;T&gt; src4; // 第四个已经排好序的Region Proposals队列
};</pre>
<p id="p8114998103"><a name="p8114998103"></a><a name="p8114998103"></a>Region Proposal队列的数据类型与目的操作数保持一致。src1、src2、src3、src4类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p1938181518223"><a name="p1938181518223"></a><a name="p1938181518223"></a>LocalTensor的起始地址需要保证16字节对齐（针对half数据类型），32字节对齐（针对float数据类型）。</p>
</td>
</tr>
<tr id="row6301859135119"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p229173384114"><a name="p229173384114"></a><a name="p229173384114"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="12.58125812581258%" headers="mcps1.2.4.1.2 "><p id="p32933310418"><a name="p32933310418"></a><a name="p32933310418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.75737573757377%" headers="mcps1.2.4.1.3 "><p id="p1465085251516"><a name="p1465085251516"></a><a name="p1465085251516"></a>排序所需参数，类型为MrgSort4Info结构体。</p>
<p id="p395104375712"><a name="p395104375712"></a><a name="p395104375712"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_proposal.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p102993315413"><a name="p102993315413"></a><a name="p102993315413"></a>参数说明请参考<a href="#table7515358184615">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  MrgSort4Info参数说明

<a name="table7515358184615"></a>
<table><thead align="left"><tr id="row1951513585460"><th class="cellrowborder" valign="top" width="13.611361136113612%" id="mcps1.2.4.1.1"><p id="p195151858154612"><a name="p195151858154612"></a><a name="p195151858154612"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.48124812481248%" id="mcps1.2.4.1.2"><p id="p185151585466"><a name="p185151585466"></a><a name="p185151585466"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.9073907390739%" id="mcps1.2.4.1.3"><p id="p1151555820466"><a name="p1151555820466"></a><a name="p1151555820466"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row25157580465"><td class="cellrowborder" valign="top" width="13.611361136113612%" headers="mcps1.2.4.1.1 "><p id="p05156585462"><a name="p05156585462"></a><a name="p05156585462"></a>elementLengths</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p125156582464"><a name="p125156582464"></a><a name="p125156582464"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.9073907390739%" headers="mcps1.2.4.1.3 "><p id="p1751545874619"><a name="p1751545874619"></a><a name="p1751545874619"></a>四个源Region Proposals队列的长度（Region Proposal数目），类型为长度为4的uint16_t数据类型的数组，理论上每个元素取值范围[0, 4095]，但不能超出UB的存储空间。</p>
</td>
</tr>
<tr id="row1515155817465"><td class="cellrowborder" valign="top" width="13.611361136113612%" headers="mcps1.2.4.1.1 "><p id="p55154583465"><a name="p55154583465"></a><a name="p55154583465"></a>ifExhaustedSuspension</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p6515185814615"><a name="p6515185814615"></a><a name="p6515185814615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.9073907390739%" headers="mcps1.2.4.1.3 "><p id="p135151558194611"><a name="p135151558194611"></a><a name="p135151558194611"></a>某条队列耗尽后，指令是否需要停止，类型为bool，默认false。</p>
</td>
</tr>
<tr id="row1234542411541"><td class="cellrowborder" valign="top" width="13.611361136113612%" headers="mcps1.2.4.1.1 "><p id="p143461024185414"><a name="p143461024185414"></a><a name="p143461024185414"></a>validBit</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p16291844175420"><a name="p16291844175420"></a><a name="p16291844175420"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.9073907390739%" headers="mcps1.2.4.1.3 "><div class="p" id="p4633153935413"><a name="p4633153935413"></a><a name="p4633153935413"></a>有效队列个数，取值如下：<a name="ul1925313221792"></a><a name="ul1925313221792"></a><ul id="ul1925313221792"><li>3：前两条队列有效</li><li>7：前三条队列有效</li><li>15：四条队列全部有效</li></ul>
</div>
</td>
</tr>
<tr id="row59828263546"><td class="cellrowborder" valign="top" width="13.611361136113612%" headers="mcps1.2.4.1.1 "><p id="p898232615410"><a name="p898232615410"></a><a name="p898232615410"></a>repeatTimes</p>
</td>
<td class="cellrowborder" valign="top" width="12.48124812481248%" headers="mcps1.2.4.1.2 "><p id="p169826260544"><a name="p169826260544"></a><a name="p169826260544"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.9073907390739%" headers="mcps1.2.4.1.3 "><p id="p330857175415"><a name="p330857175415"></a><a name="p330857175415"></a>迭代次数，每一次源操作数和目的操作数跳过四个队列总长度。取值范围：repeatTimes∈[1,255]。</p>
<div class="p" id="p1301057145414"><a name="p1301057145414"></a><a name="p1301057145414"></a>repeatTimes参数生效是有条件的，需要同时满足以下四个条件：<a name="ul19311157155420"></a><a name="ul19311157155420"></a><ul id="ul19311157155420"><li>四个源Region Proposals队列的长度一致</li><li>四个源Region Proposals队列连续存储</li><li>ifExhaustedSuspension = False</li><li>validBit=15</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   当存在proposal\[i\]与proposal\[j\]的score值相同时，如果i\>j，则proposal\[j\]将首先被选出来，排在前面。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   不支持源操作数与目的操作数之间存在地址重叠。

## 调用示例<a name="section642mcpsimp"></a>

-   接口使用样例

    ```
    // vconcatWorkLocal为已经创建并且完成排序的4个Region Proposals，每个Region Proposal数目是16个
    struct MrgSortSrcList<half> srcList(vconcatWorkLocal[0], vconcatWorkLocal[1], vconcatWorkLocal[2], vconcatWorkLocal[3]);
    uint16_t elementLengths[4] = {16, 16, 16, 16};
    struct MrgSort4Info srcInfo(elementLengths, false, 15, 1);
    AscendC::MrgSort4(dstLocal, srcList, srcInfo);
    ```

```

```

```
示例结果
输入数据(src_gm):
[-38.1    82.7   -40.75  -54.62   21.67  -58.53   25.94  -79.5   -61.44
  26.7   -27.45   48.78   86.75  -18.1   -58.8    62.38   46.38  -78.94
 -87.7   -13.81  -13.25   46.94  -47.8   -50.44   34.16   20.3    80.1
 -94.1    52.4   -42.75   83.4    80.44  -66.8   -82.7   -91.44  -95.6
  66.2   -30.97  -36.53   61.66   24.92  -45.1    38.97  -34.62  -69.8
  59.1    34.22   11.695 -33.47   52.1    -4.832  46.88   56.78   71.4
  13.29  -35.78   52.44  -46.03   83.8    83.56   71.3    -9.086 -65.06
  46.25 ]
输出数据(dst_gm):
[  0.      0.      0.      0.     86.75    0.      0.      0.      0.
   0.      0.      0.     83.8     0.      0.      0.      0.      0.
   0.      0.     83.56    0.      0.      0.      0.      0.      0.
   0.     83.4     0.      0.      0.      0.      0.      0.      0.
  82.7     0.      0.      0.      0.      0.      0.      0.     80.44
   0.      0.      0.      0.      0.      0.      0.     80.1     0.
   0.      0.      0.      0.      0.      0.     71.4     0.      0.
   0.      0.      0.      0.      0.     71.3     0.      0.      0.
   0.      0.      0.      0.     66.2     0.      0.      0.      0.
   0.      0.      0.     62.38    0.      0.      0.      0.      0.
   0.      0.     61.66    0.      0.      0.      0.      0.      0.
   0.     59.1     0.      0.      0.      0.      0.      0.      0.
  56.78    0.      0.      0.      0.      0.      0.      0.     52.44
   0.      0.      0.      0.      0.      0.      0.     52.4     0.
   0.      0.      0.      0.      0.      0.     52.1     0.      0.
   0.      0.      0.      0.      0.     48.78    0.      0.      0.
   0.      0.      0.      0.     46.94    0.      0.      0.      0.
   0.      0.      0.     46.88    0.      0.      0.      0.      0.
   0.      0.     46.38    0.      0.      0.      0.      0.      0.
   0.     46.25    0.      0.      0.      0.      0.      0.      0.
  38.97    0.      0.      0.      0.      0.      0.      0.     34.22
   0.      0.      0.      0.      0.      0.      0.     34.16    0.
   0.      0.      0.      0.      0.      0.     26.7     0.      0.
   0.      0.      0.      0.      0.     25.94    0.      0.      0.
   0.      0.      0.      0.     24.92    0.      0.      0.      0.
   0.      0.      0.     21.67    0.      0.      0.      0.      0.
   0.      0.     20.3     0.      0.      0.      0.      0.      0.
   0.     13.29    0.      0.      0.      0.      0.      0.      0.
  11.695   0.      0.      0.      0.      0.      0.      0.     -4.832
   0.      0.      0.      0.      0.      0.      0.     -9.086   0.
   0.      0.      0.      0.      0.      0.    -13.25    0.      0.
   0.      0.      0.      0.      0.    -13.81    0.      0.      0.
   0.      0.      0.      0.    -18.1     0.      0.      0.      0.
   0.      0.      0.    -27.45    0.      0.      0.      0.      0.
   0.      0.    -30.97    0.      0.      0.      0.      0.      0.
   0.    -33.47    0.      0.      0.      0.      0.      0.      0.
 -34.62    0.      0.      0.      0.      0.      0.      0.    -35.78
   0.      0.      0.      0.      0.      0.      0.    -36.53    0.
   0.      0.      0.      0.      0.      0.    -38.1     0.      0.
   0.      0.      0.      0.      0.    -40.75    0.      0.      0.
   0.      0.      0.      0.    -42.75    0.      0.      0.      0.
   0.      0.      0.    -45.1     0.      0.      0.      0.      0.
   0.      0.    -46.03    0.      0.      0.      0.      0.      0.
   0.    -47.8     0.      0.      0.      0.      0.      0.      0.
 -50.44    0.      0.      0.      0.      0.      0.      0.    -54.62
   0.      0.      0.      0.      0.      0.      0.    -58.53    0.
   0.      0.      0.      0.      0.      0.    -58.8     0.      0.
   0.      0.      0.      0.      0.    -61.44    0.      0.      0.
   0.      0.      0.      0.    -65.06    0.      0.      0.      0.
   0.      0.      0.    -66.8     0.      0.      0.      0.      0.
   0.      0.    -69.8     0.      0.      0.      0.      0.      0.
   0.    -78.94    0.      0.      0.      0.      0.      0.      0.
 -79.5     0.      0.      0.      0.      0.      0.      0.    -82.7
   0.      0.      0.      0.      0.      0.      0.    -87.7     0.
   0.      0.      0.      0.      0.      0.    -91.44    0.      0.
   0.      0.      0.      0.      0.    -94.1     0.      0.      0.
   0.      0.      0.      0.    -95.6     0.      0.      0.   ]
```

