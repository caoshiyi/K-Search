# ClampMin<a name="ZH-CN_TOPIC_0000002523304774"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1110612208496"></a>

将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。

<!-- img2text -->
$$
dstTensor_i =
\begin{cases}
srcTensor_i, & \text{if } srcTensor_i \geq scalar \\
scalar, & \text{otherwise}
\end{cases}
$$

<!-- img2text -->
$$
dstTensor_i =
\begin{cases}
srcTensor_i, & srcTensor_i \geq scalar \\
scalar, & srcTensor_i < scalar
\end{cases}
$$

## 函数原型<a name="section11872143332919"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ClampMin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void ClampMin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const T scalar, const uint32_t calCount)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetClampMaxMinTmpSize](GetClampMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明<a name="section198731433172915"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001538537601_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001538537601_p675519193268"><a name="zh-cn_topic_0000001538537601_p675519193268"></a><a name="zh-cn_topic_0000001538537601_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001538537601_p375511918267"><a name="zh-cn_topic_0000001538537601_p375511918267"></a><a name="zh-cn_topic_0000001538537601_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001538537601_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>操作数的数据类型。</p>
<p id="p20761101616589"><a name="p20761101616589"></a><a name="p20761101616589"></a><span id="ph126252025205"><a name="ph126252025205"></a><a name="ph126252025205"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table15873113310294"></a>
<table><thead align="left"><tr id="row387313342913"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p108731336299"><a name="p108731336299"></a><a name="p108731336299"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.38%" id="mcps1.2.4.1.2"><p id="p9873183342912"><a name="p9873183342912"></a><a name="p9873183342912"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10873153311295"><a name="p10873153311295"></a><a name="p10873153311295"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row887323316296"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p168731433162919"><a name="p168731433162919"></a><a name="p168731433162919"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p087323318299"><a name="p087323318299"></a><a name="p087323318299"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p7873103319298"><a name="p7873103319298"></a><a name="p7873103319298"></a>目的操作数。</p>
<p id="p1687363382916"><a name="p1687363382916"></a><a name="p1687363382916"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1887373302919"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1487316336299"><a name="p1487316336299"></a><a name="p1487316336299"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p17873163362918"><a name="p17873163362918"></a><a name="p17873163362918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p11873193312912"><a name="p11873193312912"></a><a name="p11873193312912"></a>源操作数。</p>
<p id="p387414337297"><a name="p387414337297"></a><a name="p387414337297"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p178744331292"><a name="p178744331292"></a><a name="p178744331292"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row587423382912"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p18874933132917"><a name="p18874933132917"></a><a name="p18874933132917"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p78741233132911"><a name="p78741233132911"></a><a name="p78741233132911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p10874633152918"><a name="p10874633152918"></a><a name="p10874633152918"></a>临时缓存。</p>
<p id="p6874113314291"><a name="p6874113314291"></a><a name="p6874113314291"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p13874833112919"><a name="p13874833112919"></a><a name="p13874833112919"></a>用于ClampMin内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p1787453312912"><a name="p1787453312912"></a><a name="p1787453312912"></a>临时空间大小BufferSize的获取方式请参考<a href="GetClampMaxMinTmpSize.md">GetClampMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row2582153962218"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p458212399227"><a name="p458212399227"></a><a name="p458212399227"></a>scalar</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p193571850192218"><a name="p193571850192218"></a><a name="p193571850192218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1258213915222"><a name="p1258213915222"></a><a name="p1258213915222"></a>scalar数据，数据类型与srcTensor一致，支持数据类型为：half、float。</p>
</td>
</tr>
<tr id="row18874113352915"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p2087433320296"><a name="p2087433320296"></a><a name="p2087433320296"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p11874733192916"><a name="p11874733192916"></a><a name="p11874733192916"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p4874153311290"><a name="p4874153311290"></a><a name="p4874153311290"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section8875533172917"></a>

无

## 约束说明<a name="section10875233142913"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section387573318293"></a>

完整的调用样例请参考[更多样例](更多样例-104.md)。

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
pipe.InitBuffer(tmpQue, 1, bufferSize);
AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
// 输入shape信息为128
AscendC::ClampMin<half>(dstLocal, srcLocal, sharedTmpBuffer, static_cast<half>(2), 128);
```

结果示例如下：

```
输入数据(srcLocal): [0 0 0 ...  3 3]
输出数据(dstLocal): [2 2 2  ...  3 3]
```

