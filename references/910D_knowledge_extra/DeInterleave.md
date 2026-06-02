# DeInterleave<a name="ZH-CN_TOPIC_0000002523343650"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

给定源操作数src0和src1，将src0和src1中的元素解交织存入结果操作数dst0和dst1中。解交织排列方式如下图所示，其中每个方格代表一个元素。

<!-- img2text -->
```
两个输入
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌───┬───┬───┬───┐      ┌───┐         ┌───┬───┬───┬───┐      ┌───┐          │
│  │ 0 │ 0 │ 1 │ 1 │      │ 3 │         │ 4 │ 4 │ 5 │ 5 │      │ 7 │          │
│  └───┴───┴───┴───┘      └───┘         └───┴───┴───┴───┘      └───┘          │
│                                                                              │
│                          │                                                   │
│                          ↓                                                   │
│                                                                              │
│  ┌───┬───┬───┬───┐      ┌───┐         ┌───┬───┬───┬───┐      ┌───┐          │
│  │ 0 │ 1 │ 2 │ 3 │      │ 7 │         │ 0 │ 1 │ 2 │ 3 │      │ 7 │          │
│  └───┴───┴───┴───┘      └───┘         └───┴───┴───┴───┘      └───┘          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

一个输入
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌───┬───┬───┬───┐      ┌───┐                                               │
│  │ 0 │ 0 │ 1 │ 1 │      │ 3 │                                               │
│  └───┴───┴───┴───┘      └───┘                                               │
│                                                                              │
│                          │                                                   │
│                          ↓                                                   │
│                                                                              │
│  ┌───┬───┬───┬───┐                  ┌───┬───┬───┬───┐                        │
│  │ 0 │ 1 │ 2 │ 3 │                  │ 0 │ 1 │ 2 │ 3 │                        │
│  └───┴───┴───┴───┘                  └───┴───┴───┴───┘                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

-   两个输入

    ```
    template <typename T>
    __aicore__ inline void DeInterleave(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t count)
    ```

-   一个输入

    ```
    template <typename T>
    __aicore__ inline void DeInterleave(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, const LocalTensor<T>& src, const int32_t srcCount)
    ```

## 参数说明<a name="section176711403104"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.509999999999998%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.49%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.509999999999998%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.49%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>dst0/dst1</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1515191511407"><a name="p1515191511407"></a><a name="p1515191511407"></a>目的操作数。</p>
<p id="p65530137137"><a name="p65530137137"></a><a name="p65530137137"></a><span id="ph173308471594"><a name="ph173308471594"></a><a name="ph173308471594"></a><span id="ph9902231466"><a name="ph9902231466"></a><a name="ph9902231466"></a><span id="ph1782115034816"><a name="ph1782115034816"></a><a name="ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p37511234195317"><a name="p37511234195317"></a><a name="p37511234195317"></a><span id="ph19174141065411"><a name="ph19174141065411"></a><a name="ph19174141065411"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p468305719192"><a name="p468305719192"></a><a name="p468305719192"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row6553613191315"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p195531113161311"><a name="p195531113161311"></a><a name="p195531113161311"></a>src/src0/src1</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p155310135134"><a name="p155310135134"></a><a name="p155310135134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p7218122944012"><a name="p7218122944012"></a><a name="p7218122944012"></a>源操作数。</p>
<p id="p15422163732418"><a name="p15422163732418"></a><a name="p15422163732418"></a><span id="ph97971326111115"><a name="ph97971326111115"></a><a name="ph97971326111115"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p2811183544"><a name="p2811183544"></a><a name="p2811183544"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p2012716431610"><a name="p2012716431610"></a><a name="p2012716431610"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p11792551164616"><a name="p11792551164616"></a><a name="p11792551164616"></a><span id="ph11792155174611"><a name="ph11792155174611"></a><a name="ph11792155174611"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float/uint64_t/int64_t</p>
</td>
</tr>
<tr id="row103840207421"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p20104222104514"><a name="p20104222104514"></a><a name="p20104222104514"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p2183122716423"><a name="p2183122716423"></a><a name="p2183122716423"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p62147396317"><a name="p62147396317"></a><a name="p62147396317"></a>输入/输出数据元素个数，dst0/dst1/src0/src1长度大小为count。count必须为偶数。</p>
</td>
</tr>
<tr id="row825185514307"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p825165517307"><a name="p825165517307"></a><a name="p825165517307"></a>srcCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p18894175913307"><a name="p18894175913307"></a><a name="p18894175913307"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p1025125514300"><a name="p1025125514300"></a><a name="p1025125514300"></a>输入数据元素个数，两个输出的大小都为输入的一半。srcCount必须为偶数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section14483414194"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section176061616102911"></a>

本样例中只展示Compute流程中的部分代码。

-   两个输入

    ```
    AscendC::DeInterleave(dst0Local, dst1Local, src0Local, src1Local, 512);
    ```

    结果示例如下：

    ```
    输入数据src0Local: [1 2 3 ... 512]
    输入数据src1Local: [513 514 515 ... 1024]
    输出数据dst0Local: [1 3 5 ... 1023]
    输出数据dst1Local: [2 4 6 ... 1024]
    ```

-   一个输入

    ```
    AscendC::DeInterleave(dst0Local, dst1Local, srcLocal, 512);
    ```

    结果示例如下：

    ```
    输入数据srcLocal: [1 2 3 ... 512]
    输出数据dst0Local: [1 3 5 ... 511]
    输出数据dst1Local: [2 4 6 ... 512]
    ```

