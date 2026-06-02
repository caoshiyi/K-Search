# SoftmaxFlashV3<a name="ZH-CN_TOPIC_0000002554344405"></a>

## 产品支持情况<a name="section717414537392"></a>

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

SoftmaxFlash增强版本，对应Softmax PASA算法。将输入tensor\[m<sub>0</sub>, m<sub>1</sub>, ..., m<sub>t</sub>, n\]（t大于或等于0）的非尾轴长度m<sub>0</sub>, m<sub>1</sub>, ..., m<sub>t</sub>相乘的结果看作m，则输入tensor的shape看作\[m, n\]。对输入tensor x的尾轴进行切分，分块个数为splitMeanCnt，切分后的tensor为x\_cnt<sub>i</sub>。按如下公式进行计算，其中x、inmax、insum、inmean为输入，M、S、E、A均为输出。

-   update为false：

    <!-- img2text -->
$$
M_i = \max(x\_{{\rm cnt}_i})
$$

$$
S_i = \sum \exp(x\_{{\rm cnt}_i} - M_i)
$$

$$
E_i = \exp(x\_{{\rm cnt}_i} - M_i)
$$

$$
A_i = \frac{E_i}{S_i}
$$

-   update为true：

    ![](figures/zh-cn_formulaimage_0000002523306760.png)

本接口当前只支持ND格式的输入，内部的reduce过程按last轴处理。

为方便理解，通过Python伪代码实现的方式，表达其计算公式如下。其中，repeatSize为64，elementNumPerBlk/BlkcntPerRepeat为8，splitMeanCnt为8，src、inmean、inmax、 insum、update为输入，dst、x\_mean、x\_sum、x\_max、exp\_max为输出。

```
def softmax_flash_3(src, height, width, loopCnt, alpha, baseK, inmax=None, insum=None, inmean=None, update=False):
    scalar = alpha / (1 - alpha)
    #(m,n)->(m,64)
    tmpbuffer0 = BlockReduceSum(repeatSize, repeatSize, elementNumPerBlk)
    remain = int(width / repeatSize - BlkcntPerRepeat)
    tmpbuffer0 = Add(tmpbuffer0, src, remain, repeatSize * elementNumPerBlk, width)
    #(m,64)->(m,8)
    tmpbuffer0 = BlockReduceSum(1, elementNumPerBlk, elementNumPerBlk)
    #width = baseK * splitMeanCnt
    rowMeanLocal = tmpbuffer0 / baseK
    rowMeanGlobal = np.mean(src, axis=(-1), keepdims=True)
    rowMeanGlobalTmp = (rowMeanGlobal - rowMeanLocal) * scalar
    src = src - rowMeanGlobalTmp 

    if update == False:
        x_mean = rowMeanGlobal
        maxTmp = np.max(src, axis=-1, keepdims=True)
        shiftCurr = (rowMeanGlobal - x_mean) * scalar
        x_max = shiftCurr + maxTmp
        maxTmp = x_max - shiftCurr
        x_sub = src - maxTmp   
        dst = np.exp(x_sub) 
        x_sum = np.sum(dst, axis=-1, keepdims=True)
        exp_max = None
        return dst, x_max, x_sum, x_mean, exp_max
    else:
        x_mean = (rowMeanGlobal + inmean * (loopCnt - 1)) / loopCnt
        maxTmp = np.max(src, axis=-1, keepdims=True)
        shiftCurr = (rowMeanGlobal - x_mean) * scalar
        shiftPrev = (inmean - x_mean) * scalar
	x_max = shiftCurr + maxTmp
        maxTmp = shiftPrev + inmax
        x_max = np.max(np.concatenate((x_max, maxTmp), axis=(-1)), axis=(-1), keepdims=True)
        maxTmp = x_max - shiftCurr
        x_sub = src - maxTmp   
        dst = np.exp(x_sub)
        exp_max = np.exp(inmax - x_max + shiftPrev)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        x_sum = exp_max * insum +  x_sum
        return x_exp, x_max, x_sum, x_mean, exp_max
```

## 函数原型<a name="section620mcpsimp"></a>

-   接口框架申请临时空间

    ```
    template <typename T, typename U, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftmaxFlashV3(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor, const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor, const LocalTensor<U>& inMaxTensor, const SoftMaxTiling& tiling, const SoftMaxParams& params)
    ```

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, typename U, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
    __aicore__ inline void SoftmaxFlashV3(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor, const LocalTensor<U>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxParams& params)
    ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[SoftmaxFlashV3 Tiling接口](SoftmaxFlashV3-Tiling接口.md)中提供的GetSoftMaxFlashV3MaxMinTmpSize接口获取所需最小和最大临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.37%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.63%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>输入srcTensor及输出dstTensor、expMaxTensor操作数的数据类型。</p>
<p id="p21077427455"><a name="p21077427455"></a><a name="p21077427455"></a><span id="ph11107542164518"><a name="ph11107542164518"></a><a name="ph11107542164518"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half。</p>
</td>
</tr>
<tr id="row1956194674517"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p65611746104520"><a name="p65611746104520"></a><a name="p65611746104520"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p15611346104511"><a name="p15611346104511"></a><a name="p15611346104511"></a>输入inMeanTensor、inExpSumTensor、inMaxTensor及输出meanTensor、expSumTensor、maxTensor操作数的数据类型。</p>
<p id="p195681030123113"><a name="p195681030123113"></a><a name="p195681030123113"></a><span id="ph17568630133118"><a name="ph17568630133118"></a><a name="ph17568630133118"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：float。</p>
</td>
</tr>
<tr id="row1783875017236"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p1838850172311"><a name="p1838850172311"></a><a name="p1838850172311"></a>isUpdate</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p1983816503234"><a name="p1983816503234"></a><a name="p1983816503234"></a>是否使能update为true的计算。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p2967318476"><a name="p2967318476"></a><a name="p2967318476"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row9184124919159"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p11692440141619"><a name="p11692440141619"></a><a name="p11692440141619"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p1715789476"><a name="p1715789476"></a><a name="p1715789476"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row11813131962412"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p16813119192415"><a name="p16813119192415"></a><a name="p16813119192415"></a>isDataFormatNZ</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p102836117472"><a name="p102836117472"></a><a name="p102836117472"></a>该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row996414112248"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p69641412246"><a name="p69641412246"></a><a name="p69641412246"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p973519408519"><a name="p973519408519"></a><a name="p973519408519"></a>该参数预留，传入默认值SOFTMAX_DEFAULT_CFG即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.55%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.68%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1480238142815"><a name="p1480238142815"></a><a name="p1480238142815"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p948033819282"><a name="p948033819282"></a><a name="p948033819282"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1327102402720"><a name="p1327102402720"></a><a name="p1327102402720"></a>dstTensor的shape和源操作数srcTensor一致。</p>
</td>
</tr>
<tr id="row178801001268"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3346391166"><a name="p3346391166"></a><a name="p3346391166"></a>meanTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p43466912612"><a name="p43466912612"></a><a name="p43466912612"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p534618910611"><a name="p534618910611"></a><a name="p534618910611"></a>目的操作数。</p>
<p id="p534659564"><a name="p534659564"></a><a name="p534659564"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p63476910610"><a name="p63476910610"></a><a name="p63476910610"></a>用于保存softmax计算过程中平均值的结果。</p>
<a name="ul1134709860"></a><a name="ul1134709860"></a><ul id="ul1134709860"><li>meanTensor的last轴长度固定为32Byte，即一个<span id="ph634711917611"><a name="ph634711917611"></a><a name="ph634711917611"></a>datablock</span>长度。该<span id="ph183471591366"><a name="ph183471591366"></a><a name="ph183471591366"></a>datablock</span>中的所有数据为同一个值。比如float数据类型下，该<span id="ph13347109262"><a name="ph13347109262"></a><a name="ph13347109262"></a>datablock</span>中的8个数均为相同的reducesum求平均后的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p20480153802811"><a name="p20480153802811"></a><a name="p20480153802811"></a>expSumTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p154801238192812"><a name="p154801238192812"></a><a name="p154801238192812"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p1659223718226"><a name="p1659223718226"></a><a name="p1659223718226"></a>目的操作数。</p>
<p id="p22921529162311"><a name="p22921529162311"></a><a name="p22921529162311"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p19237153711415"><a name="p19237153711415"></a><a name="p19237153711415"></a>用于保存softmax计算过程中reducesum的结果。</p>
<a name="ul12308151213513"></a><a name="ul12308151213513"></a><ul id="ul12308151213513"><li>expSumTensor的last轴长度固定为32Byte，即一个<span id="ph16212125511335"><a name="ph16212125511335"></a><a name="ph16212125511335"></a>datablock</span>长度。该<span id="ph1235145803313"><a name="ph1235145803313"></a><a name="ph1235145803313"></a>datablock</span>中的所有数据为同一个值。比如float数据类型下，该<span id="ph1056215610362"><a name="ph1056215610362"></a><a name="ph1056215610362"></a>datablock</span>中的8个数均为相同的reducesum的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1648015388281"><a name="p1648015388281"></a><a name="p1648015388281"></a>maxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p14480738162817"><a name="p14480738162817"></a><a name="p14480738162817"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p033515495269"><a name="p033515495269"></a><a name="p033515495269"></a>目的操作数。</p>
<p id="p1097223511233"><a name="p1097223511233"></a><a name="p1097223511233"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p296320351261"><a name="p296320351261"></a><a name="p296320351261"></a>用于保存softmax计算过程中reducemax的结果。</p>
<a name="ul9474401463"></a><a name="ul9474401463"></a><ul id="ul9474401463"><li>maxTensor的last轴长度固定为32Byte，即一个<span id="ph1620974113410"><a name="ph1620974113410"></a><a name="ph1620974113410"></a>datablock</span>长度。该<span id="ph473675173413"><a name="ph473675173413"></a><a name="ph473675173413"></a>datablock</span>中的所有数据为同一个值。比如float数据类型下，该<span id="ph17613118113419"><a name="ph17613118113419"></a><a name="ph17613118113419"></a>datablock</span>中的8个数均为相同的reducemax的值。</li><li>非last轴的长度与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p204801738152820"><a name="p204801738152820"></a><a name="p204801738152820"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p1248073811284"><a name="p1248073811284"></a><a name="p1248073811284"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p1159016208298"><a name="p1159016208298"></a><a name="p1159016208298"></a>源操作数。</p>
<p id="p495119425238"><a name="p495119425238"></a><a name="p495119425238"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p15480438132811"><a name="p15480438132811"></a><a name="p15480438132811"></a>last轴长度需要32Byte对齐。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p3480143892814"><a name="p3480143892814"></a><a name="p3480143892814"></a>expMaxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p144802038122815"><a name="p144802038122815"></a><a name="p144802038122815"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p920126333"><a name="p920126333"></a><a name="p920126333"></a>目的操作数。</p>
<p id="p19731148142315"><a name="p19731148142315"></a><a name="p19731148142315"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<a name="ul6386131510362"></a><a name="ul6386131510362"></a><ul id="ul6386131510362"><li>expMaxTensor的last轴长度固定为32Byte，即一个<span id="ph862151363410"><a name="ph862151363410"></a><a name="ph862151363410"></a>datablock</span>长度。该<span id="ph662931423419"><a name="ph662931423419"></a><a name="ph662931423419"></a>datablock</span>中的所有数据为同一个值。比如half数据类型下，该<span id="ph161371317203414"><a name="ph161371317203414"></a><a name="ph161371317203414"></a>datablock</span>中的16个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row20284152719415"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1287661943519"><a name="p1287661943519"></a><a name="p1287661943519"></a>inMeanTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p9876171923511"><a name="p9876171923511"></a><a name="p9876171923511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p88763196357"><a name="p88763196357"></a><a name="p88763196357"></a>源操作数。</p>
<p id="p1387691953517"><a name="p1387691953517"></a><a name="p1387691953517"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p08765197355"><a name="p08765197355"></a><a name="p08765197355"></a>softmax计算所需要的mean值。</p>
<a name="ul087611983516"></a><a name="ul087611983516"></a><ul id="ul087611983516"><li>inMeanTensor的last轴长度固定为32Byte，即一个<span id="ph17876131918355"><a name="ph17876131918355"></a><a name="ph17876131918355"></a>datablock</span>长度。该<span id="ph1687615194359"><a name="ph1687615194359"></a><a name="ph1687615194359"></a>datablock</span>中的所有数据为同一个值。比如float数据类型下，该<span id="ph687691993512"><a name="ph687691993512"></a><a name="ph687691993512"></a>datablock</span>中的8个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row17371444131520"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p548011382287"><a name="p548011382287"></a><a name="p548011382287"></a>inExpSumTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p134801438182813"><a name="p134801438182813"></a><a name="p134801438182813"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p14729613163018"><a name="p14729613163018"></a><a name="p14729613163018"></a>源操作数。</p>
<p id="p167015292315"><a name="p167015292315"></a><a name="p167015292315"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p76602013123618"><a name="p76602013123618"></a><a name="p76602013123618"></a>softmax计算所需要的sum值。</p>
<a name="ul7387193685112"></a><a name="ul7387193685112"></a><ul id="ul7387193685112"><li>inExpSumTensor的last轴长度固定为32Byte，即一个<span id="ph11711321183412"><a name="ph11711321183412"></a><a name="ph11711321183412"></a>datablock</span>长度。该<span id="ph20484112218343"><a name="ph20484112218343"></a><a name="ph20484112218343"></a>datablock</span>中的所有数据为同一个值。比如float数据类型下，该<span id="ph17413225113410"><a name="ph17413225113410"></a><a name="ph17413225113410"></a>datablock</span>中的8个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row1811919157286"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p184801438162811"><a name="p184801438162811"></a><a name="p184801438162811"></a>inMaxTensor</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p1748023810286"><a name="p1748023810286"></a><a name="p1748023810286"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p5531616153011"><a name="p5531616153011"></a><a name="p5531616153011"></a>源操作数。</p>
<p id="p3150558152311"><a name="p3150558152311"></a><a name="p3150558152311"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_8"><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_8"><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_8"><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p247013293372"><a name="p247013293372"></a><a name="p247013293372"></a>softmax计算所需要的max值。</p>
<a name="ul2353125216375"></a><a name="ul2353125216375"></a><ul id="ul2353125216375"><li>inMaxTensor的last轴长度固定为32Byte，即一个<span id="ph0342129153413"><a name="ph0342129153413"></a><a name="ph0342129153413"></a>datablock</span>长度。该<span id="ph172770329347"><a name="ph172770329347"></a><a name="ph172770329347"></a>datablock</span>中的所有数据为同一个值。比如float数据类型下，该<span id="ph4963234153413"><a name="ph4963234153413"></a><a name="ph4963234153413"></a>datablock</span>中的8个数均为相同的值。</li><li>非last轴的长度需要与dstTensor保持一致。</li></ul>
</td>
</tr>
<tr id="row635216316480"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1358462510237"><a name="p1358462510237"></a><a name="p1358462510237"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p15584152512313"><a name="p15584152512313"></a><a name="p15584152512313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p155841325182310"><a name="p155841325182310"></a><a name="p155841325182310"></a>临时空间。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_9"><a name="zh-cn_topic_0000002523303824_ph173308471594_9"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_9"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_9"><a name="zh-cn_topic_0000002523303824_ph9902231466_9"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_9"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_9"><a name="zh-cn_topic_0000002523303824_ph1782115034816_9"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_9"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8928165616507"><a name="p8928165616507"></a><a name="p8928165616507"></a>该操作数的数据类型固定uint8_t。</p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="SoftmaxFlashV3-Tiling接口.md">SoftmaxFlashV3 Tiling接口</a>。</p>
</td>
</tr>
<tr id="row1765118177284"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p54804387283"><a name="p54804387283"></a><a name="p54804387283"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p1048083882812"><a name="p1048083882812"></a><a name="p1048083882812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p16481838112813"><a name="p16481838112813"></a><a name="p16481838112813"></a>SoftmaxFlashV3接口计算所需Tiling信息，Tiling信息的获取请参考<a href="SoftmaxFlashV3-Tiling接口.md">SoftmaxFlashV3 Tiling接口</a>。</p>
</td>
</tr>
<tr id="row6303151181211"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p0734515621"><a name="p0734515621"></a><a name="p0734515621"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="7.55%" headers="mcps1.2.4.1.2 "><p id="p1973441513219"><a name="p1973441513219"></a><a name="p1973441513219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.68%" headers="mcps1.2.4.1.3 "><p id="p1573415151214"><a name="p1573415151214"></a><a name="p1573415151214"></a>srcTensor的shape信息和计算相关参数。SoftMaxParams类型，具体定义如下：</p>
<a name="screen1740392817204"></a><a name="screen1740392817204"></a><pre class="screen" codetype="Cpp" id="screen1740392817204">struct SoftMaxParams {
    uint32_t srcM; // 非尾轴长度的乘积
    uint32_t srcK; // 尾轴长度，必须32Byte对齐
    uint32_t oriSrcM; // 原始非尾轴长度的乘积
    uint32_t oriSrcK;  // 原始尾轴长度
    uint32_t loopCnt; // update为true时，公式中的循环次数loopCnt，该参数大于等于1
    uint32_t splitMeanCnt; // 公式中计算每一行平均值时的分块个数，当前该参数仅支持取值为8
    float alpha; // 公式中的计算参数，推荐取值0.9375、0.96889、0.984497
};</pre>
<p id="p1065853610376"><a name="p1065853610376"></a><a name="p1065853610376"></a>注意，当前本接口不支持非对齐场景，因此参数srcM与oriSrcM相等，参数srcK与oriSrcK相等。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   对于输入srcTensor需要满足：尾轴长度n大于等于512，同时n是64的倍数；非尾轴长度的乘积m为8的倍数。
-   srcTensor和dstTensor的Tensor的空间可以复用，meanTensor和inMeanTensor的空间可以复用，maxTensor和inMaxTensor的空间可以复用，expSumTensor和inExpSumTensor的空间可以复用。
-   meanTensor、expSumTensor、maxTensor、expMaxTensor、inMeanTensor、inExpSumTensor、inMaxTensor的Tensor空间，last轴长度必须是32字节。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section94691236101419"></a>

本样例中输入srcTensor和输出dstTensor的shape大小为\[8, 1024\]，输入inMeanTensor、inExpSumTensor、inMaxTensor的shape大小为\[8, 8\]，数据类型为float；输出expMaxTensor的shape大小为\[8, 16\]，数据类型为half；输入和输出的数据排布格式为ND，srcTensor和dstTensor空间不复用，模板参数isUpdate为true。算子样例请参考[softmaxflashv3算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/softmaxflashv3)。

```
#include "kernel_operator.h"

AscendC::LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
AscendC::LocalTensor<U> insumLocal = sumQueue.DeQue<U>();
AscendC::LocalTensor<U> inmaxLocal = maxQueue.DeQue<U>();
AscendC::LocalTensor<U> inmeanLocal = meanQueue.DeQue<U>();
AscendC::LocalTensor<T> expMaxTensor = expMaxQueue.AllocTensor<T>();
AscendC::LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
AscendC::SoftMaxParams params = {height, width, height, width, loopCnt, splitMeanCnt, alpha};
AscendC::SoftmaxFlashV3<T, U, true>(dstLocal, inmeanLocal, insumLocal, inmaxLocal, srcLocal, expMaxTensor, inmeanLocal, insumLocal, inmaxLocal, tiling, params);

outQueueDst.EnQue<T>(dstLocal);
maxQueue.FreeTensor(inmaxLocal);
sumQueue.FreeTensor(insumLocal);
meanQueue.FreeTensor(inmeanLocal);
inQueueSrc.FreeTensor(srcLocal);
```

