# Sort32<a name="ZH-CN_TOPIC_0000002523303992"></a>

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

排序函数，一次迭代可以完成32个数的排序，数据需要按如下描述结构进行保存：

score和index分别存储在src0和src1中，按score进行排序（score大的排前面），排序好的score与其对应的index一起以（score, index）的结构存储在dst中。不论score为half还是float类型，dst中的（score, index）结构总是占据8Bytes空间。

如下所示：

-   当score为float，index为uint32\_t类型时，计算结果中index存储在高4Bytes，score存储在低4Bytes。

    <!-- img2text -->
```text
score
┌───────────────────────┐
│       score[0]        │
├───────────────────────┤
│       score[1]        │
├───────────────────────┤
│       score[2]        │
├───────────────────────┤
│          ...          │
├───────────────────────┤
│       score[30]       │
├───────────────────────┤
│       score[31]       │
└───────────────────────┘
<───────────────────────>
          4B

index
┌───────────────────────┐
│       index[0]        │
├───────────────────────┤
│       index[1]        │
├───────────────────────┤
│       index[2]        │
├───────────────────────┤
│          ...          │
├───────────────────────┤
│       index[30]       │
├───────────────────────┤
│       index[31]       │
└───────────────────────┘
<───────────────────────>
          4B

                    ───────────────→

result
┌──────────────────────────┬──────────────────────────┐
│         Score[5]         │         index[5]         │
├──────────────────────────┼──────────────────────────┤
│        score[11]         │        index[11]         │
├──────────────────────────┼──────────────────────────┤
│        score[20]         │        index[20]         │
├──────────────────────────┼──────────────────────────┤
│           ...            │           ...            │
├──────────────────────────┼──────────────────────────┤
│         score[1]         │         index[1]         │
├──────────────────────────┼──────────────────────────┤
│         score[8]         │         index[8]         │
└──────────────────────────┴──────────────────────────┘
<─────────────────────────────────────────────────────>
                          8B
```

-   当score为half，index为uint32\_t类型时，计算结果中index存储在高4Bytes，score存储在低2Bytes， 中间的2Bytes保留。

    <!-- img2text -->
```
┌──────────────┐             ┌──────────────┐                               ┌──────────────┬──────────────┬──────────────┐
│   score[0]   │             │   index[0]   │                               │   Score[5]   │   reserved   │   index[5]   │
├──────────────┤             ├──────────────┤                               ├──────────────┼──────────────┼──────────────┤
│   score[1]   │             │   index[1]   │                               │  score[11]   │   reserved   │  index[11]   │
├──────────────┤             ├──────────────┤                               ├──────────────┼──────────────┼──────────────┤
│   score[2]   │             │   index[2]   │                               │  score[20]   │   reserved   │  index[20]   │
├──────────────┤             ├──────────────┤             ┌──────────┐      ├──────────────┼──────────────┼──────────────┤
│      ...     │             │      ...     │     ─────→ │          │ ───→ │      ...     │      ...     │      ...     │
├──────────────┤             ├──────────────┤             │          │      ├──────────────┼──────────────┼──────────────┤
│  score[30]   │             │  index[30]   │             └──────────┘      │   score[1]   │   reserved   │   index[1]   │
├──────────────┤             ├──────────────┤                               ├──────────────┼──────────────┼──────────────┤
│  score[31]   │             │  index[31]   │                               │   score[8]   │   reserved   │   index[8]   │
└──────────────┘             └──────────────┘                               └──────────────┴──────────────┴──────────────┘
      ↓                              ↓                                                ↗──────────────────────────────↘
                                                                                      ↖──────────────────────────────↙

<──────────────>
      2B

                                   <──────────────>
                                         4B

                                                                                  <──────────────────────────────────────────────>
                                                                                                  8B
```

说明:
- 左侧为 score 数据，每个元素宽度为 2B，标注了 score[0]、score[1]、score[2]、...、score[30]、score[31]
- 中间为 index 数据，每个元素宽度为 4B，标注了 index[0]、index[1]、index[2]、...、index[30]、index[31]
- 右侧为合并后的结果，每个元素宽度为 8B，按列布局为：Score / reserved / index
- 右侧示例中显示的元素顺序为：Score[5] + reserved + index[5]，score[11] + reserved + index[11]，score[20] + reserved + index[20]，...，score[1] + reserved + index[1]，score[8] + reserved + index[8]
- 红色箭头表示 score 写入结果的低 2Bytes，index 写入结果的高 4Bytes，中间 2Bytes 为 reserved
- 结合上下文：当 score 为 half、index 为 uint32_t 类型时，计算结果中 index 存储在高 4Bytes，score 存储在低 2Bytes，中间的 2Bytes 保留。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void Sort32(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src1, const int32_t repeatTime)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.58%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.42%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.58%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.42%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p1618914391266"><a name="p1618914391266"></a><a name="p1618914391266"></a><span id="ph219011391667"><a name="ph219011391667"></a><a name="ph219011391667"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/float</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table62161631132810"></a>
<table><thead align="left"><tr id="row12216103118284"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p1421643114288"><a name="p1421643114288"></a><a name="p1421643114288"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p82165310285"><a name="p82165310285"></a><a name="p82165310285"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row82161131182810"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p62165318282"><a name="p62165318282"></a><a name="p62165318282"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p102161931162814"><a name="p102161931162814"></a><a name="p102161931162814"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p3944122817141"><a name="p3944122817141"></a><a name="p3944122817141"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p16168483301"><a name="p16168483301"></a><a name="p16168483301"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row5216163192815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p13216193192813"><a name="p13216193192813"></a><a name="p13216193192813"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p7217031182818"><a name="p7217031182818"></a><a name="p7217031182818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p185486379149"><a name="p185486379149"></a><a name="p185486379149"></a>源操作数。</p>
<p id="p5449124113142"><a name="p5449124113142"></a><a name="p5449124113142"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1315655114301"><a name="p1315655114301"></a><a name="p1315655114301"></a><span id="ph6621175133017"><a name="ph6621175133017"></a><a name="ph6621175133017"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1521763119281"><a name="p1521763119281"></a><a name="p1521763119281"></a>此源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row88875522820"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p141252118106"><a name="p141252118106"></a><a name="p141252118106"></a>src1</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p105185518102"><a name="p105185518102"></a><a name="p105185518102"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p683812512153"><a name="p683812512153"></a><a name="p683812512153"></a>源操作数。</p>
<p id="p577151261519"><a name="p577151261519"></a><a name="p577151261519"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p653611582304"><a name="p653611582304"></a><a name="p653611582304"></a><span id="ph9865155873012"><a name="ph9865155873012"></a><a name="ph9865155873012"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p178873523815"><a name="p178873523815"></a><a name="p178873523815"></a>此源操作数固定为uint32_t数据类型。</p>
</td>
</tr>
<tr id="row521753120287"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1336715511855"><a name="p1336715511855"></a><a name="p1336715511855"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p63676515516"><a name="p63676515516"></a><a name="p63676515516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p636715110511"><a name="p636715110511"></a><a name="p636715110511"></a>重复迭代次数，int32_t类型，每次迭代完成32个元素的排序，下次迭代src0和src1各跳过32个elements，dst跳过32*8 Byte空间。取值范围：repeatTime∈[0,255]。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   当存在score\[i\]与score\[j\]相同时，如果i\>j，则score\[j\]将首先被选出来，排在前面。
-   每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

-   接口使用样例

    ```
    AscendC::LocalTensor<float> srcLocal0 = inQueueSrc0.DeQue<float>();
    AscendC::LocalTensor<uint32_t> srcLocal1 = inQueueSrc1.DeQue<uint32_t>();
    AscendC::LocalTensor<float> dstLocal = outQueueDst.AllocTensor<float>();
    // repeatTime = 4, 对128个数分成4组进行排序，每次完成1组32个数的排序
    AscendC::Sort32<float>(dstLocal, srcLocal0, srcLocal1, 4);
    outQueueDst.EnQue<float>(dstLocal);
    inQueueSrc0.FreeTensor(srcLocal0);
    inQueueSrc1.FreeTensor(srcLocal1);
    ```

