# SoftmaxFlash<a name="ZH-CN_TOPIC_0000002554343465"></a>

## 产品支持情况<a name="section1733961153710"></a>

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

**注意：该接口后续即将废弃，请使用精度和性能更好的SoftmaxFlashV2接口**。

Softmax增强版本，除了可以对输入tensor做SoftmaxFlash计算，还可以根据上一次Softmax计算的sum和max来更新本次的Softmax计算结果。last轴切轴的情况，每次计算的reduce结果并非是全轴的，需要根据上一次Softmax计算的sum和max来更新本次的Softmax计算结果，可以使用该增强接口。不支持NZ格式。

当前仅支持传入shape为ND格式，内部的reduce过程都是按last轴进行。不使能update时，该接口等同于[SoftMax](SoftMax.md)。

为方便理解，通过Python脚本实现的方式，表达其计算公式如下，其中src、inmax、 insum、update为输入，dst、x\_sum、x\_max、exp\_max为输出。

```
def softmax_flash(src, inmax=None, insum=None, update=None):
    if update == None:
        #基于last轴进行rowmax(按行取最大值)处理
        x_max = np.max(src, axis=-1, keepdims=True)
        x_sub = src - x_max
        x_exp = np.exp(x_sub)
        #基于last轴进行rowsum(按行求和)处理
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        dst = x_exp / x_sum
        exp_max = None
        return dst, x_max, x_sum, exp_max
    else:
        #将inmax和src拼接后求rowmax
        x_max = np.max(np.concatenate((inmax, src), axis=-1), axis=-1, keepdims=True)
        x_exp = np.exp(src - x_max)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        exp_max = np.exp(inmax - x_max)
        x_sum = exp_max * insum +  x_sum
        exp_max = exp_max * insum / x_sum
        dst = x_exp / x_sum
        return dst, x_max, x_sum, exp_max
```

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间

    ```
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlash(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor, const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor, const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const SoftMaxTiling &tiling, bool isUpdate = false, const SoftMaxShapeInfo &softmaxShapeInfo = {})
    ```

    ```
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlash(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlash(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

    ```
    template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void SoftmaxFlash(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
    ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftmaxFlash Tiling接口](SoftmaxFlash-Tiling接口.md)中提供的GetSoftMaxFlashMaxTmpSize/GetSoftMaxFlashMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

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
<p id="p15140111103613"><a name="p15140111103613"></a><a name="p15140111103613"></a><span id="ph14851141511365"><a name="ph14851141511365"></a><a name="ph14851141511365"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p6210735174418"><a name="p6210735174418"></a><a name="p6210735174418"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p5203192942015"><a name="p5203192942015"></a><a name="p5203192942015"></a>srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。是否满足基本块的要求，可以采用如下两种方式之一判断：</p>
<a name="ul353295811167"></a><a name="ul353295811167"></a><ul id="ul353295811167"><li>srcTensor和dstTensor的shape信息[m,n]需要满足如下条件：<a name="ul09181366549"></a><a name="ul09181366549"></a><ul id="ul09181366549"><li>尾轴长度n小于2048并且大于等于256/sizeof(T)（即half场景下n最小为128，float场景下n最小为64），同时n是64的倍数；</li><li>非尾轴长度的乘积m为8的倍数。</li></ul>
</li></ul>
<a name="ul14203192932019"></a><a name="ul14203192932019"></a><ul id="ul14203192932019"><li>在Tiling实现中，通过调用<a href="IsBasicBlockInSoftMax.md">IsBasicBlockInSoftMax</a>判断Tiling切分策略是否满足基本块的切分要求。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.08%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.15%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1480238142815"><a name="p1480238142815"></a><a name="p1480238142815"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p948033819282"><a name="p948033819282"></a><a name="p948033819282"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p15716105922110"><a name="p15716105922110"></a><a name="p15716105922110"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1327102402720"><a name="p1327102402720"></a><a name="p1327102402720"></a>dstTensor的shape和源操作数srcTensor一致。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p20480153802811"><a name="p20480153802811"></a><a name="p20480153802811"></a>sumTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p154801238192812"><a name="p154801238192812"></a><a name="p154801238192812"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p1659223718226"><a name="p1659223718226"></a><a name="p1659223718226"></a>目的操作数。</p>
<p id="p1477114221222"><a name="p1477114221222"></a><a name="p1477114221222"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p19237153711415"><a name="p19237153711415"></a><a name="p19237153711415"></a>用于保存softmax计算过程中reducesum的结果。</p>
<a name="ul12308151213513"></a><a name="ul12308151213513"></a><ul id="ul12308151213513"><li>sumTensor的last轴长度固定为32Byte，即一个<span id="ph885316116307"><a name="ph885316116307"></a><a name="ph885316116307"></a>datablock</span>长度。该<span id="ph145417593316"><a name="ph145417593316"></a><a name="ph145417593316"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph0156111103211"><a name="ph0156111103211"></a><a name="ph0156111103211"></a>datablock</span>中的16个数均为相同的reducesum的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1648015388281"><a name="p1648015388281"></a><a name="p1648015388281"></a>maxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p14480738162817"><a name="p14480738162817"></a><a name="p14480738162817"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p033515495269"><a name="p033515495269"></a><a name="p033515495269"></a>目的操作数。</p>
<p id="p718410319222"><a name="p718410319222"></a><a name="p718410319222"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p296320351261"><a name="p296320351261"></a><a name="p296320351261"></a>用于保存softmax计算过程中reducemax的结果。</p>
<a name="ul9474401463"></a><a name="ul9474401463"></a><ul id="ul9474401463"><li>maxTensor的last轴长度固定为32Byte，即一个<span id="ph714417517322"><a name="ph714417517322"></a><a name="ph714417517322"></a>datablock</span>长度。该<span id="ph1028819613324"><a name="ph1028819613324"></a><a name="ph1028819613324"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph176121863219"><a name="ph176121863219"></a><a name="ph176121863219"></a>datablock</span>中的16个数均为相同的reducemax的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p204801738152820"><a name="p204801738152820"></a><a name="p204801738152820"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1248073811284"><a name="p1248073811284"></a><a name="p1248073811284"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p1159016208298"><a name="p1159016208298"></a><a name="p1159016208298"></a>源操作数。</p>
<p id="p17539184112218"><a name="p17539184112218"></a><a name="p17539184112218"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p10724195415422"><a name="p10724195415422"></a><a name="p10724195415422"></a>last轴长度需要32Byte对齐。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3480143892814"><a name="p3480143892814"></a><a name="p3480143892814"></a>expMaxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p144802038122815"><a name="p144802038122815"></a><a name="p144802038122815"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p920126333"><a name="p920126333"></a><a name="p920126333"></a>目的操作数。</p>
<p id="p10556846192216"><a name="p10556846192216"></a><a name="p10556846192216"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul18871125781517"></a><a name="ul18871125781517"></a><ul id="ul18871125781517"><li>expMaxTensor的last轴长度固定为32Byte，即一个<span id="ph184394168326"><a name="ph184394168326"></a><a name="ph184394168326"></a>datablock</span>长度。该<span id="ph3817111723219"><a name="ph3817111723219"></a><a name="ph3817111723219"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph14885112012328"><a name="ph14885112012328"></a><a name="ph14885112012328"></a>datablock</span>中的16个数均为相同的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p548011382287"><a name="p548011382287"></a><a name="p548011382287"></a>inSumTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p134801438182813"><a name="p134801438182813"></a><a name="p134801438182813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p14729613163018"><a name="p14729613163018"></a><a name="p14729613163018"></a>源操作数。</p>
<p id="p2748175012218"><a name="p2748175012218"></a><a name="p2748175012218"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1729191333011"><a name="p1729191333011"></a><a name="p1729191333011"></a>softmax计算所需要的sum值。</p>
<a name="ul6386131510362"></a><a name="ul6386131510362"></a><ul id="ul6386131510362"><li>inSumTensor的last轴长度固定为32Byte，即一个<span id="ph147306258326"><a name="ph147306258326"></a><a name="ph147306258326"></a>datablock</span>长度。该<span id="ph1519572712323"><a name="ph1519572712323"></a><a name="ph1519572712323"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph46781629143214"><a name="ph46781629143214"></a><a name="ph46781629143214"></a>datablock</span>中的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row1811919157286"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p184801438162811"><a name="p184801438162811"></a><a name="p184801438162811"></a>inMaxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1748023810286"><a name="p1748023810286"></a><a name="p1748023810286"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p5531616153011"><a name="p5531616153011"></a><a name="p5531616153011"></a>源操作数。</p>
<p id="p1836195582212"><a name="p1836195582212"></a><a name="p1836195582212"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p247013293372"><a name="p247013293372"></a><a name="p247013293372"></a>softmax计算所需要的max值。</p>
<a name="ul2353125216375"></a><a name="ul2353125216375"></a><ul id="ul2353125216375"><li>inMaxTensor的last轴长度固定为32Byte，即一个<span id="ph187663313325"><a name="ph187663313325"></a><a name="ph187663313325"></a>datablock</span>长度。该<span id="ph1914715358323"><a name="ph1914715358323"></a><a name="ph1914715358323"></a>datablock</span>中的所有数据为同一个值，比如half数据类型下，该<span id="ph26211237133212"><a name="ph26211237133212"></a><a name="ph26211237133212"></a>datablock</span>里的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row12348141616337"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftmaxFlash-Tiling接口.md">SoftmaxFlash Tiling接口</a>。</p>
</td>
</tr>
<tr id="row1765118177284"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p54804387283"><a name="p54804387283"></a><a name="p54804387283"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1048083882812"><a name="p1048083882812"></a><a name="p1048083882812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p16481838112813"><a name="p16481838112813"></a><a name="p16481838112813"></a>接口计算所需tiling信息，Tiling信息的获取请参考<a href="SoftmaxFlash-Tiling接口.md">SoftmaxFlash Tiling接口</a>。</p>
</td>
</tr>
<tr id="row135941736162817"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p64811038182817"><a name="p64811038182817"></a><a name="p64811038182817"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p748114382285"><a name="p748114382285"></a><a name="p748114382285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p1048119384289"><a name="p1048119384289"></a><a name="p1048119384289"></a>是否使能update算法。</p>
</td>
</tr>
<tr id="row869173524410"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>softmaxShapeInfo</p>
</td>
<td class="cellrowborder" valign="top" width="8.08%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.15%" headers="mcps1.2.4.1.3 "><p id="p1681113319405"><a name="p1681113319405"></a><a name="p1681113319405"></a>srcTensor的shape信息。SoftMaxShapeInfo类型，具体定义如下：</p>
<a name="screen1740392817204"></a><a name="screen1740392817204"></a><pre class="screen" codetype="Cpp" id="screen1740392817204">struct SoftMaxShapeInfo {
    uint32_t srcM; // 非尾轴长度的乘积
    uint32_t srcK; // 尾轴长度，必须32Byte对齐
    uint32_t oriSrcM; // 原始非尾轴长度的乘积
    uint32_t oriSrcK;  // 原始尾轴长度
};</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   srcTensor和dstTensor的空间可以复用，maxTensor和inMaxTensor的空间可以复用，sumTensor和inSumTensor的空间可以复用。
-   sumTensor、maxTensor、expMaxTensor、inSumTensor、inMaxTensor的Tensor空间，last轴长度必须固定32Byte。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

本样例输入src的Shape大小为\[80,144\]，输出Shape大小dst=\[80,144\]，输入inExpSumTensor=\[80,16\]，输入inMaxTensor=\[80,16\]，输出expMaxTensor=\[80,16\]，数据类型均为half，update为false。

```
#include "kernel_operator.h"

template <typename T>
class KernelSoftmaxFlash {
public:
    __aicore__ inline KernelSoftmaxFlash()
    {}
    __aicore__ inline void Init(
        GM_ADDR srcGm, GM_ADDR inMaxGm, GM_ADDR inSumGm, GM_ADDR dstGm, const SoftMaxTiling &tilingData)
    {
        elementNumPerBlk = 32 / sizeof(T);
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        maxGlobal.SetGlobalBuffer((__gm__ T *)inMaxGm);
        sumGlobal.SetGlobalBuffer((__gm__ T *)inSumGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrc, 1, height * width * sizeof(T));
        pipe.InitBuffer(outQueueDst, 1, height * width * sizeof(T));
        pipe.InitBuffer(inMaxQueue, 1, height * elementNumPerBlk * sizeof(T));
        pipe.InitBuffer(inSumQueue, 1, height * elementNumPerBlk * sizeof(T));
        pipe.InitBuffer(expMaxQueue, 1, height * elementNumPerBlk * sizeof(T));
        tiling = tilingData;
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
        AscendC::LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        AscendC::LocalTensor<T> inSumLocal = inSumQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> inMaxLocal = inMaxQueue.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, height * width);
        AscendC::DataCopy(inSumLocal, sumGlobal, height * elementNumPerBlk);
        AscendC::DataCopy(inMaxLocal, maxGlobal, height * elementNumPerBlk);
        inQueueSrc.EnQue(srcLocal);
        inSumQueue.EnQue(inSumLocal);
        inMaxQueue.EnQue(inMaxLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

        AscendC::LocalTensor<T> inMaxLocal = inMaxQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> inSumLocal = inSumQueue.AllocTensor<T>();
        AscendC::LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();
        AscendC::SoftMaxShapeInfo srcShape = {height, width, height, width};
        AscendC::SoftmaxFlash<T, false>(srcLocal,
            inSumLocal,
            inMaxLocal,
            srcLocal,
            expMaxTensor,
            inSumLocal,
            inMaxLocal,
            tiling,
            false,
            srcShape);

        AscendC::DataCopy(dstLocal, srcLocal, height * width);

        outQueueDst.EnQue<T>(dstLocal);
        inMaxQueue.FreeTensor(inMaxLocal);
        inSumQueue.FreeTensor(inSumLocal);
        inQueueSrc.FreeTensor(srcLocal);
 
        expMaxQueue.FreeTensor(expMaxTensor);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        AscendC::DataCopy(dstGlobal, dstLocal, height * width);
        outQueueDst.FreeTensor(dstLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inMaxQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inSumQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> expMaxQueue;

    AscendC::GlobalTensor<T> srcGlobal, dstGlobal;
    AscendC::GlobalTensor<T> maxGlobal, sumGlobal;
    uint32_t elementNumPerBlk = 0;
    uint32_t width = 144;
    uint32_t height = 80;
    SoftMaxTiling tiling;
};

extern "C" __global__ __aicore__ void softmax_flash_kernel_half(GM_ADDR srcGm, GM_ADDR inMaxGm, GM_ADDR inSumGm, GM_ADDR dstGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelSoftmaxFlash<half> op;
    op.Init(srcGm, inMaxGm, inSumGm, dstGm, tilingData.softmaxTilingData);
    op.Process();
}
```

