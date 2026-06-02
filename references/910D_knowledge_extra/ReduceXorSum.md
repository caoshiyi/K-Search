# ReduceXorSum<a name="ZH-CN_TOPIC_0000002523304210"></a>

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

按照元素执行Xor（按位异或）运算，并将计算结果ReduceSum求和。

> **说明：** 
>**注意：当最终计算结果超出int16范围\[-32768，32767\]后，将输出-32768 或者 32767。**

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ReduceXorSum(LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ReduceXorSum(LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const uint32_t calCount);
    ```

由于该接口的内部实现中需要保存异或结果以及进行其他运算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetReduceXorSumMaxMinTmpSize](GetReduceXorSumMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

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
<p id="p382544110205"><a name="p382544110205"></a><a name="p382544110205"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int16_t。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p2322650204211"><a name="p2322650204211"></a><a name="p2322650204211"></a>是否允许修改源操作数，默认值为false。如果开发者允许源操作数被改写，可以使能该参数，使能后能够节省部分内存空间。</p>
<p id="p183225504424"><a name="p183225504424"></a><a name="p183225504424"></a>设置为<strong id="b732214507424"><a name="b732214507424"></a><a name="b732214507424"></a>true</strong>，则本接口内部计算时<strong id="b5322165017429"><a name="b5322165017429"></a><a name="b5322165017429"></a>复用</strong>src0Tensor和src1Tensor的内存空间，节省内存空间；设置为<strong id="b2322350134218"><a name="b2322350134218"></a><a name="b2322350134218"></a>false</strong>，则本接口内部计算时<strong id="b6322175013421"><a name="b6322175013421"></a><a name="b6322175013421"></a>不复用</strong>src0Tensor和src1Tensor的内存空间。</p>
<p id="p62891018544"><a name="p62891018544"></a><a name="p62891018544"></a>isReuseSource的使用样例请参考<a href="更多样例-104.md#section639165323915">更多样例</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.38%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p7421248132119"><a name="p7421248132119"></a><a name="p7421248132119"></a>输出值需要sizeof(T)大小的空间进行保存。开发者要根据该大小和框架的对齐要求来为dstTensor分配实际内存空间。</p>
<div class="note" id="note12620157122515"><a name="note12620157122515"></a><a name="note12620157122515"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1962012573256"><a name="p1962012573256"></a><a name="p1962012573256"></a><strong id="b26211557122513"><a name="b26211557122513"></a><a name="b26211557122513"></a>注意：遵循框架对内存开辟的要求（开辟内存的大小满足32Byte对齐），即sizeof(T)不是32Byte对齐时，需要向上进行32Byte对齐。为了对齐而多开辟的内存空间不填值，为一些随机值。</strong></p>
</div></div>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>src0Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数0。</p>
<p id="p261714443395"><a name="p261714443395"></a><a name="p261714443395"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row15497954143513"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p154021015364"><a name="p154021015364"></a><a name="p154021015364"></a>src1Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p1440319114363"><a name="p1440319114363"></a><a name="p1440319114363"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p940311113616"><a name="p940311113616"></a><a name="p940311113616"></a>源操作数1。</p>
<p id="p340313120367"><a name="p340313120367"></a><a name="p340313120367"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p0403518364"><a name="p0403518364"></a><a name="p0403518364"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row4848123011192"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1148552183919"><a name="p1148552183919"></a><a name="p1148552183919"></a>临时缓存。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于ReduceXorSum计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetReduceXorSumMaxMinTmpSize.md">GetReduceXorSumMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row206666339251"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1949611581317"><a name="p1949611581317"></a><a name="p1949611581317"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p174961758436"><a name="p174961758436"></a><a name="p174961758436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p184961858133"><a name="p184961858133"></a><a name="p184961858133"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   calCount需要保证小于或等于src0Tensor和src1Tensor的元素范围。
-   当最终计算结果超出int16范围\[-32768，32767\]后，将输出-32768 或者 32767。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;

pipe.InitBuffer(inQueueX, 1, 32 * sizeof(int16_t));
pipe.InitBuffer(inQueueY, 1, 32 * sizeof(int16_t));
pipe.InitBuffer(outQueue, 1, 32);
pipe.InitBuffer(tmpQue, 1, bufferSize);  // bufferSize 通过Host侧tiling参数获取

AscendC::LocalTensor<int16_t> dstLocal = outQueue.AllocTensor<int16_t>();
AscendC::LocalTensor<int16_t> src0Local = inQueueX.AllocTensor<int16_t>();
AscendC::LocalTensor<int16_t> src1Local = inQueueY.AllocTensor<int16_t>();
AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();

// 不使用输入内存，输入shape信息为32, 算子输入的数据类型为int16_t, 实际计算个数为前32
AscendC::ReduceXorSum<int16_t, false>(dstLocal, src0Local, src1Local, sharedTmpBuffer, 32);
```

结果示例如下：

```
输入输出的数据类型为int16_t
输入数据(src0Local): [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
输入数据(src1Local): [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

输出数据(dstLocal): [32 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] // 仅32为有效值
```

