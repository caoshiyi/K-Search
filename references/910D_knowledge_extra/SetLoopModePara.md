# SetLoopModePara<a name="ZH-CN_TOPIC_0000002554423453"></a>

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

DataCopy、DataCopyPad过程中通过该接口使能loop mode并且设置loop mode的参数，需要和[ResetLoopModePara](ResetLoopModePara.md)搭配使用，在数据搬运结束后通过ResetLoopModePara重置loop mode的参数。支持的通路如下：

-   GM-\>VECIN
-   VECOUT-\>GM

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetLoopModePara(const LoopModeParams& loopParams, DataCopyMVType type)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>loopParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p964217483714"><a name="p964217483714"></a><a name="p964217483714"></a>循环模式参数 LoopModeParams类型，定义如下，具体参数说明请参考<a href="#table1940815635619">表2</a>。</p>
<pre class="screen" id="screen1929612197911"><a name="screen1929612197911"></a><a name="screen1929612197911"></a>struct LoopModeParams {
        loop1Size = 0;
        loop2Size = 0;
        loop1SrcStride = 0;
        loop1DstStride = 0;
        loop2SrcStride = 0;
        loop2DstStride = 0 ;
};</pre>
</td>
</tr>
<tr id="row6553613191315"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p1030265017216"><a name="p1030265017216"></a><a name="p1030265017216"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p155310135134"><a name="p155310135134"></a><a name="p155310135134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p202731227454"><a name="p202731227454"></a><a name="p202731227454"></a>数据搬运模式。DataCopyMVType为枚举类型，定义如下，具体参数说明请参考<a href="#table1166074612214">表3</a>。</p>
<pre class="screen" id="screen1354412162247"><a name="screen1354412162247"></a><a name="screen1354412162247"></a>enum class DataCopyMVType : uint8_t {
    UB_TO_OUT = 0,
    OUT_TO_UB = 1,
};</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  LoopModeParams结构体参数说明

<a name="table1940815635619"></a>
<table><thead align="left"><tr id="row1940813563564"><th class="cellrowborder" valign="top" width="16.689999999999998%" id="mcps1.2.3.1.1"><p id="p1408155635620"><a name="p1408155635620"></a><a name="p1408155635620"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.31%" id="mcps1.2.3.1.2"><p id="p0409115655616"><a name="p0409115655616"></a><a name="p0409115655616"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row195147113498"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p1615719212254"><a name="p1615719212254"></a><a name="p1615719212254"></a>loop1Size</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p14375151815499"><a name="p14375151815499"></a><a name="p14375151815499"></a>用于设置内层循环的循环次数，数据类型为uint32_t，取值范围为[0, 2^21)。</p>
</td>
</tr>
<tr id="row340910561569"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p1356141941913"><a name="p1356141941913"></a><a name="p1356141941913"></a>loop2Size</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p7648132417192"><a name="p7648132417192"></a><a name="p7648132417192"></a>用于设置外层循环的循环次数，数据类型为uint32_t，取值范围为[0, 2^21)。</p>
</td>
</tr>
<tr id="row15692203815118"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p102538173580"><a name="p102538173580"></a><a name="p102538173580"></a>loop1SrcStride</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p13625155717375"><a name="p13625155717375"></a><a name="p13625155717375"></a>用于设置内层循环中相邻迭代源操作数的数据块间的间隔，单位为Byte，数据类型为uint64_t。</p>
<a name="ul2030853083810"></a><a name="ul2030853083810"></a><ul id="ul2030853083810"><li>当数据搬运模式是UB_TO_OUT的时候取值范围为[0, 2^21)，并且loop1SrcStride必须32B对齐。</li><li>当数据搬运模式是OUT_TO_UB的时候取值范围为[0, 2^40)。</li></ul>
</td>
</tr>
<tr id="row069218382119"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p225351713586"><a name="p225351713586"></a><a name="p225351713586"></a>loop1DstStride</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p17217125332617"><a name="p17217125332617"></a><a name="p17217125332617"></a>用于设置内层循环中相邻迭代目的操作数的数据块间的间隔，单位为Byte，数据类型为uint64_t。</p>
<a name="ul156331432163920"></a><a name="ul156331432163920"></a><ul id="ul156331432163920"><li>当数据搬运模式是UB_TO_OUT的时候取值范围为[0, 2^40)。</li><li>当数据搬运模式是OUT_TO_UB的时候取值范围为[0, 2^21)，并且loop1DstStride必须32B对齐。</li></ul>
</td>
</tr>
<tr id="row26921538101118"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p1421013317366"><a name="p1421013317366"></a><a name="p1421013317366"></a>loop2SrcStride</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p182102353617"><a name="p182102353617"></a><a name="p182102353617"></a>用于设置外层循环中相邻迭代源操作数的数据块间的间隔，单位为Byte，数据类型为uint64_t。</p>
<a name="ul2892174743912"></a><a name="ul2892174743912"></a><ul id="ul2892174743912"><li>当数据搬运模式是UB_TO_OUT的时候取值范围为[0, 2^21)，并且loop2SrcStride必须32B对齐。</li><li>当数据搬运模式是OUT_TO_UB的时候取值范围为[0, 2^40)。</li></ul>
</td>
</tr>
<tr id="row1569215387118"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p198182214369"><a name="p198182214369"></a><a name="p198182214369"></a>loop2DstStride</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p148182212361"><a name="p148182212361"></a><a name="p148182212361"></a>用于设置外层循环中相邻迭代目的操作数的数据块间的间隔，单位为Byte，数据类型为uint64_t。</p>
<a name="ul1413116306413"></a><a name="ul1413116306413"></a><ul id="ul1413116306413"><li>当数据搬运模式是UB_TO_OUT的时候取值范围为[0, 2^40)。</li><li>当数据搬运模式是OUT_TO_UB的时候取值范围为[0, 2^21)，并且loop2DstStride必须32B对齐。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 3**  DataCopyMVType结构体参数说明

<a name="table1166074612214"></a>
<table><thead align="left"><tr id="row2066014461221"><th class="cellrowborder" valign="top" width="16.689999999999998%" id="mcps1.2.3.1.1"><p id="p11660174612227"><a name="p11660174612227"></a><a name="p11660174612227"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.31%" id="mcps1.2.3.1.2"><p id="p866115468223"><a name="p866115468223"></a><a name="p866115468223"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row116611646112213"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p6262039194513"><a name="p6262039194513"></a><a name="p6262039194513"></a>UB_TO_OUT</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p115783414467"><a name="p115783414467"></a><a name="p115783414467"></a>从UB搬运到GM的通路。</p>
</td>
</tr>
<tr id="row11661124616223"><td class="cellrowborder" valign="top" width="16.689999999999998%" headers="mcps1.2.3.1.1 "><p id="p1223162134716"><a name="p1223162134716"></a><a name="p1223162134716"></a>OUT_TO_UB</p>
</td>
<td class="cellrowborder" valign="top" width="83.31%" headers="mcps1.2.3.1.2 "><p id="p88693119479"><a name="p88693119479"></a><a name="p88693119479"></a>从GM搬运到UB的通路。</p>
</td>
</tr>
</tbody>
</table>

下面的样例呈现了SetLoopModePara的使用方法。

-   样例中在数据类型为int8\_t的场景下，数据块大小为384，配置DataCopyPad的数据搬运模式为Compact模式，blockLen = 48，blockCount = 2，表明每个连续传输数据块包含48Bytes，且连续传输数据块有两个，srcStride = 0, dstStride = 0，isPad = false，表明源操作数相邻数据块之间没有间隔且不需要填充用户自定义的数据；
-   再设置SetLoopModePara中LoopModeParams的参数：loop1Size = 2，loop2Size = 2，loop1SrcStride = 96，loop2SrcStride =192，loop1DstStride =  128，loop2DstStride = 288，DataCopyMVType为OUT\_TO\_UB，表明内层循环和外层循坏的次数分别为2次，内层循环和外层循环中相邻迭代源操作数的数据块间隔分别为96Bytes和192Bytes，内层循环和外层循环中相邻迭代目的操作数的数据块间隔分别为128Bytes和288Bytes，通路是从GM搬运到UB；
-   使用以上配置，调用SetLoopModePara再调用DataCopyPad就可以开启DataCopyPad的loop模式完成数据类型为int8\_t的数据块大小为384的数据搬运。详细图解如下：

**图 1**  源操作数搬运场景示例<a name="fig6671114911311"></a>  
<!-- img2text -->
```
loop2SrcStride = 192                                                   loop2Size = 2

            loop1SrcStride = 96                 loop1Size = 2                              src.gm len438*8=384B

src(GM)
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ 48Bytes  │ 48Bytes  │ 48Bytes  │ 48Bytes  │ 48Bytes  │ 48Bytes  │ 48Bytes  │ 48Bytes  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
<─────────────────────>
     blockLen = 48
<───────────────────────────────────────────>
                blockCount = 2
<───────────────────────────────────────────────────────────────────────────────────────────>
                                   loop1SrcStride = 96
<───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────>
                                                           loop2SrcStride = 192
```

- blockLen=48: 覆盖第1-1块(48Bytes)
- blockCount=2: 覆盖第1-2块(48Bytes+48Bytes)
- loop1SrcStride=96: 第1组与第2组之间的源操作数相邻迭代数据块间隔，覆盖按内层循环分组的跨度
- loop2SrcStride=192: 外层循环中相邻迭代源操作数的数据块间隔，覆盖按外层循环分组的跨度

**图 2**  目的操作数搬运场景示例<a name="fig226181311513"></a>  
<!-- img2text -->
```text
dst ( VECIN )

                                        loop1DstStride = 128                                                     ub len48*8 + 32*2 + 64*1 = 512B

┌────────────┬────────────┬──────────┬────────────┬────────────┬──────────┬────────────┬────────────┬──────────┬────────────┬────────────┐
│  48Bytes   │  48Bytes   │ 32Bytes  │  48Bytes   │  48Bytes   │ 64Bytes  │  48Bytes   │  48Bytes   │ 32Bytes  │  48Bytes   │  48Bytes   │
└────────────┴────────────┴──────────┴────────────┴────────────┴──────────┴────────────┴────────────┴──────────┴────────────┴────────────┘

<────────────────────────>
      blockLen = 48

<────────────>
burstDstStride = 0

                                     <────────────────────────>
                                           blockLen = 48

                                <──────────────────────────────────────────────────────────────────────────────>
                                                      loop2DstStride = 288
```

- blockLen=48: 覆盖第1块(48Bytes)
- burstDstStride=0: 第1块与第2块之间无间隔，连续搬运
- blockLen=48: 覆盖第7块(48Bytes)
- loop1DstStride=128: 前一组起始位置到后一组起始位置跨度为128Bytes，即第1组(第1-3块，48Bytes+48Bytes+32Bytes)到第2组(第4-6块，48Bytes+48Bytes+64Bytes)的组间步长
- loop2DstStride=288: 前一大组起始位置到后一大组起始位置跨度为288Bytes，即第1大组(第1-6块)到第2大组(第7-11块)的组间步长

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   源操作数和目的操作数的起始地址需要保证32字节对齐。
-   目的操作数的数据不能重叠，如果有重叠，硬件层面不会报错或者告警，同时也不能保证重叠数据的正确性；但是不同迭代可以交织，例如内层循环中相邻迭代目的操作数的数据块间的间隔可以小于相邻连续目的操作数的数据块的间隔。
-   需要在每次使能loop mode并且设置loop mode的参数后通过ResetLoopModePara进行寄存器的复位，否则会影响到下一次对应通路的搬运的使用，引发异常。

## 调用示例<a name="section1227835243314"></a>

本示例中操作数数据类型为int8\_t。

```
AscendC::LocalTensor<int8_t> srcLocal = inQueueSrc.AllocTensor<int8_t>();
AscendC::DataCopyExtParams copyParams{2, 48 * sizeof(int8_t), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
AscendC::LoopModeParams loopParam2Ub {2, 2, 96, 128, 192, 288};
AscendC::SetLoopModePara(loopParam2Ub, DataCopyMVType::OUT_TO_UB);
AscendC::DataCopyPad<int8_t, PaddingMode::Compact(srcLocal, srcGlobal, copyParams, padParams); // 从GM->VECIN搬运 48 * 2 * 2 * 2 = 384Bytes
AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
AscendC::LoopModeParams loopParam2Gm {2, 2, 128, 96, 288, 192};
AscendC::SetLoopModePara(loopParams2Gm, DataCopyMVType::UB_TO_OUT);
DataCopyPad<T, PaddingMode::Compact>(dstGlobal, srcLocal, copyParams);
AscendC::ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
```

结果示例如下：

```
输入数据src0Global: [1 2 3 ... 384]
输出数据dstGlobal:[1 2 3 ... 384]
```

