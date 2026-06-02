# BilinearInterpolation\(ISASI\)<a name="ZH-CN_TOPIC_0000002554343875"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p158351640700"><a name="p158351640700"></a><a name="p158351640700"></a><span id="ph141589125420"><a name="ph141589125420"></a><a name="ph141589125420"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

功能分为水平迭代和垂直迭代。每个水平迭代顺序地从src0Offset读取8个偏移值，表示src0的偏移，每个偏移值指向src0的一个DataBlock的起始地址，如果repeatMode=false，从src1中取一个值，与src0中8个DataBlock中每个值进行乘操作；如果repeatMode=true，从src1中取8个值，按顺序与src0中8个DataBlock中的值进行乘操作，最后当前迭代的dst结果与前一个dst结果按DataBlock进行累加，存入目的地址，在同一个水平迭代内dst地址不变。然后进行垂直迭代，垂直迭代的dst起始地址为上一轮垂直迭代的dst起始地址加上vROffset，本轮垂直迭代占用dst空间为dst起始地址之后的8个DataBlock，每轮垂直迭代进行hRepeat次水平迭代。

<!-- img2text -->
```text
src0Offset
                 Iteration(0,0)                    Iteration(0,1)
                      ↘                                 ↙
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 0  │ O1 │ O2 │ O3 │ O4 │ O5 │ O6 │ O7 │ O0 │ O1 │ O2 │ O3 │ O4 │ O5 │ O6 │ O7 │ … │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

src0
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │ b8 │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ … │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
   ↘   ↓   ↓   ↓   ↓   ↓   ↓   ↓
    ↘  ↘   ↘   ↘   ↘   ↘   ↘   ↘
     ┌─────────────────────────────── MUL ────────────────────────────────┐

src1
                                                   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
                                                   │ e0 │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │ e0 │ e1 │ e2 │ e3 │ e4 │ e5 │ e6 │ e7 │ … │
                                                   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                                                     ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
                                                     └────────── Iteration(0,0) ──────────┘
                                                                                 └────────── Iteration(0,1) ──────────┘

                        MUL
┌────┬────┬────┬────┬────┬────┬────┬────┐      ┌────┬────┬────┬────┬────┬────┬────┬────┐      ┌────┬────┬────┬────┬────┬────┬────┬────┐
│ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │  …   │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │  …   │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │
└────┴────┴────┴────┴────┴────┴────┴────┘      └────┴────┴────┴────┴────┴────┴────┴────┘      └────┴────┴────┴────┴────┴────┴────┴────┘

──────────────────────────────────────────────→ Horizontal repeat times = n

┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│ Iteration(0,0) │  +   │ Iteration(0,1) │  +   │ Iteration(0,n) │   =   dst[0]
└────────────────┘      └────────────────┘      └────────────────┘       ┌────┬────┬────┬────┬────┬────┬────┬────┐
                                                                          │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │
                                                                          └────┴────┴────┴────┴────┴────┴────┴────┘

┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│ Iteration(1,0) │  +   │ Iteration(1,1) │  +   │ Iteration(1,n) │   =   dst[v_offset]
└────────────────┘      └────────────────┘      └────────────────┘       ┌────┬────┬────┬────┬────┬────┬────┬────┐
                                                                          │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │
                                                                          └────┴────┴────┴────┴────┴────┴────┴────┘
                                   …

┌────────────────┐      ┌────────────────┐      ┌────────────────┐
│ Iteration(m,0) │  +   │ Iteration(m,1) │  +   │ Iteration(m,n) │   =   dst[v_offset*(m-1)]
└────────────────┘      └────────────────┘      └────────────────┘       ┌────┬────┬────┬────┬────┬────┬────┬────┐
                                                                          │ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 │
                                                                          └────┴────┴────┴────┴────┴────┴────┴────┘

↓
Vertical repeat times = m


Legend
┌────┐
│ 0x │  4B
└────┘

┌────┐
│ b8 │  one datablock
└────┘

┌────┐
│ e7 │  one element
└────┘
```

说明:
- 图中 src0Offset 的每个偏移值表示 src0 的一个 DataBlock 起始地址；每个水平迭代顺序读取 8 个偏移值。
- 每个水平迭代中，src0 的 8 个 DataBlock 与 src1 对应数据做 MUL；图中用 Iteration(0,0)、Iteration(0,1) 标示相邻水平迭代。
- Horizontal repeat times = n：同一垂直层内共有 n 次水平迭代，相加后写入同一个 dst。
- Vertical repeat times = m：共有 m 次垂直迭代；各轮结果分别写入 `dst[0]`、`dst[v_offset]`、`dst[v_offset*(m-1)]` 等位置。
- 图中斜向映射箭头较多，未逐一完整展开；其含义是 src0Offset 指向 src0 中被选中的 DataBlock，并与 src1 中对应元素/元素组进行乘法。

## 函数原型<a name="section620mcpsimp"></a>

-   mask逐bit模式：

    ```
    template <typename T>
    __aicore__ inline void BilinearInterpolation(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src0Offset, const LocalTensor<T>& src1, uint64_t mask[], uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
    ```

-   mask连续模式：

    ```
    template <typename T>
    __aicore__ inline void BilinearInterpolation(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& src0Offset, const LocalTensor<T>& src1, uint64_t mask, uint8_t hRepeat, bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat, const LocalTensor<uint8_t> &sharedTmpBuffer)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="16.5%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.5%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="16.5%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.5%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数数据类型。</p>
<p id="p8228091239"><a name="p8228091239"></a><a name="p8228091239"></a><span id="ph13658558725"><a name="ph13658558725"></a><a name="ph13658558725"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.92%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.59%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p3142455111215"><a name="p3142455111215"></a><a name="p3142455111215"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p75263387255"><a name="p75263387255"></a><a name="p75263387255"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>src0、src1</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p168861511132"><a name="p168861511132"></a><a name="p168861511132"></a>源操作数。</p>
<p id="p159019410132"><a name="p159019410132"></a><a name="p159019410132"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p0585164052519"><a name="p0585164052519"></a><a name="p0585164052519"></a><span id="ph4891174012259"><a name="ph4891174012259"></a><a name="ph4891174012259"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1484485824312"><a name="p1484485824312"></a><a name="p1484485824312"></a>两个源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row6371728192310"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p23702852318"><a name="p23702852318"></a><a name="p23702852318"></a>src0Offset</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p424383212237"><a name="p424383212237"></a><a name="p424383212237"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p15371728162310"><a name="p15371728162310"></a><a name="p15371728162310"></a>源操作数。</p>
<p id="p981854462314"><a name="p981854462314"></a><a name="p981854462314"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p13772161619292"><a name="p13772161619292"></a><a name="p13772161619292"></a><span id="ph1471131712914"><a name="ph1471131712914"></a><a name="ph1471131712914"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask[]/mask</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row1344832017251"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8448920182512"><a name="p8448920182512"></a><a name="p8448920182512"></a>hRepeat</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p1244815201254"><a name="p1244815201254"></a><a name="p1244815201254"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p20448132019251"><a name="p20448132019251"></a><a name="p20448132019251"></a>水平方向迭代次数，取值范围为[1, 255]。</p>
</td>
</tr>
<tr id="row1184538182512"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p1884123822518"><a name="p1884123822518"></a><a name="p1884123822518"></a>repeatMode</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p158414389252"><a name="p158414389252"></a><a name="p158414389252"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p78283504416"><a name="p78283504416"></a><a name="p78283504416"></a>迭代模式：</p>
<a name="ul77853487528"></a><a name="ul77853487528"></a><ul id="ul77853487528"><li>false：每次迭代src0读取的8个datablock中每个值均与src1的单个数值相乘。</li><li>true：每次迭代src0的每个datablock分别与src1的1个数值相乘，共消耗8个block和8个elements。</li></ul>
</td>
</tr>
<tr id="row1329210352257"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p229218353258"><a name="p229218353258"></a><a name="p229218353258"></a>dstBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p1729283511254"><a name="p1729283511254"></a><a name="p1729283511254"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p7292143516253"><a name="p7292143516253"></a><a name="p7292143516253"></a>单次迭代内，目的操作数不同DataBlock间地址步长，以32B为单位。</p>
</td>
</tr>
<tr id="row136783102511"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p26843114256"><a name="p26843114256"></a><a name="p26843114256"></a>vROffset</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p1868931152517"><a name="p1868931152517"></a><a name="p1868931152517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p186818318258"><a name="p186818318258"></a><a name="p186818318258"></a>垂直迭代间，目的操作数地址偏移量，以元素为单位，取值范围为[128, 65535)，vROffset * sizeof(T)需要保证32字节对齐 。</p>
</td>
</tr>
<tr id="row649962862517"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p175006289257"><a name="p175006289257"></a><a name="p175006289257"></a>vRepeat</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p115001828202518"><a name="p115001828202518"></a><a name="p115001828202518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p1850012812258"><a name="p1850012812258"></a><a name="p1850012812258"></a>垂直方向迭代次数，取值范围为[1, 255]。</p>
</td>
</tr>
<tr id="row06443252253"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p7644625192513"><a name="p7644625192513"></a><a name="p7644625192513"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p7645182532515"><a name="p7645182532515"></a><a name="p7645182532515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p19161759918"><a name="p19161759918"></a><a name="p19161759918"></a>临时空间。</p>
<p id="p149612283138"><a name="p149612283138"></a><a name="p149612283138"></a><span id="ph696111288137"><a name="ph696111288137"></a><a name="ph696111288137"></a>Ascend 950PR/Ascend 950DT</span>，不需要分配临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   src0、src1、src0Offset之间不允许地址重叠，且两个垂直repeat的目的地址之间不允许地址重叠。

## 调用示例<a name="section642mcpsimp"></a>

-   接口样例-mask连续模式

    ```
    AscendC::LocalTensor<half> dstLocal, src0Local, src1Local;
    AscendC::LocalTensor<uint32_t> src0OffsetLocal;
    AscendC::LocalTensor<uint8_t> tmpLocal;
    uint64_t mask = 128;        // mask连续模式
    uint8_t hRepeat = 2;        // 水平迭代2次
    bool repeatMode = false;    // 迭代模式
    uint16_t dstBlkStride = 1;  // 单次迭代内数据连续写入
    uint16_t vROffset = 128;    // 相邻迭代间数据连续写入
    uint8_t vRepeat = 2;        // 垂直迭代2次
    
    AscendC::BilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat, repeatMode,
                dstBlkStride, vROffset, vRepeat, tmpLocal);
    ```

-   接口样例-mask逐bit模式

    ```
    AscendC::LocalTensor<half> dstLocal, src0Local, src1Local;
    AscendC::LocalTensor<uint32_t> src0OffsetLocal;
    AscendC::LocalTensor<uint8_t> tmpLocal;
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX }; // mask逐bit模式
    uint8_t hRepeat = 2;        // 水平迭代2次
    bool repeatMode = false;    // 迭代模式
    uint16_t dstBlkStride = 1;  // 单次迭代内数据连续写入
    uint16_t vROffset = 128;    // 相邻迭代间数据连续写入
    uint8_t vRepeat = 2;        // 垂直迭代2次
    
    AscendC::BilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat, repeatMode,
                dstBlkStride, vROffset, vRepeat, tmpLocal);
    ```

结果示例如下：

```
输入数据(src0Local,half): [1,2,3,...,512]
输入数据(src1Local,half): [2,3,4,...,17]
输入数据(src0OffsetLocal,uint32_t): [0,32,64,...,992]
输出数据(dstLocal,half): [389, 394, 399, 404, ...,4096]
```

