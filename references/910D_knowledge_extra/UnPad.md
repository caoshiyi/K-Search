# UnPad<a name="ZH-CN_TOPIC_0000002554423743"></a>

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

## 功能说明<a name="section1010511378106"></a>

对height \* width的二维Tensor在width方向上进行unpad，如果Tensor的width非32B对齐，则不支持调用本接口unpad。本接口具体功能场景如下：Tensor的width已32B对齐，以half为例，如16\*16，进行UnPad，变成16\*15。

## 函数原型<a name="section1834111321944"></a>

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间大小BufferSize的获取方法：通过[UnPad Tiling](UnPad-Tiling.md)中提供的**GetUnPadMaxMinTmpSize**接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式，因此UnPad接口的函数原型有两种：

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T>
    __aicore__ inline void UnPad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, UnPadParams& unPadParams, LocalTensor<uint8_t>& sharedTmpBuffer, UnPadTiling& tiling)
    ```

    该方式下开发者需自行申请并管理临时内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

-   接口框架申请临时空间

    ```
    template <typename T>
    __aicore__ inline void UnPad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, UnPadParams& unPadParams, UnPadTiling& tiling)
    ```

    该方式下开发者无需申请，但是需要预留临时空间的大小。

## 参数说明<a name="section66026315198"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001692058420_row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001692058420_p1029955044218"><a name="zh-cn_topic_0000001692058420_p1029955044218"></a><a name="zh-cn_topic_0000001692058420_p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001692058420_p1629911506421"><a name="zh-cn_topic_0000001692058420_p1629911506421"></a><a name="zh-cn_topic_0000001692058420_p1629911506421"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001692058420_row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001692058420_p1329915004219"><a name="zh-cn_topic_0000001692058420_p1329915004219"></a><a name="zh-cn_topic_0000001692058420_p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001692058420_p8299155010420"><a name="zh-cn_topic_0000001692058420_p8299155010420"></a><a name="zh-cn_topic_0000001692058420_p8299155010420"></a>操作数的数据类型。</p>
<p id="p761944013232"><a name="p761944013232"></a><a name="p761944013232"></a><span id="ph4481136201817"><a name="ph4481136201817"></a><a name="ph4481136201817"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int16_t、uint16_t、half、int32_t、uint32_t、float。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table17652725101920"></a>
<table><thead align="left"><tr id="row12652102518194"><th class="cellrowborder" valign="top" width="21.19211921192119%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.31123112311231%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="66.4966496649665%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1665215254199"><td class="cellrowborder" valign="top" width="21.19211921192119%" headers="mcps1.2.4.1.1 "><p id="p13652192518196"><a name="p13652192518196"></a><a name="p13652192518196"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.31123112311231%" headers="mcps1.2.4.1.2 "><p id="p2428856174212"><a name="p2428856174212"></a><a name="p2428856174212"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="66.4966496649665%" headers="mcps1.2.4.1.3 "><p id="p166524255194"><a name="p166524255194"></a><a name="p166524255194"></a>目的操作数，shape为二维，LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。</p>
<p id="p618437618"><a name="p618437618"></a><a name="p618437618"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row56521525201917"><td class="cellrowborder" valign="top" width="21.19211921192119%" headers="mcps1.2.4.1.1 "><p id="p10652112521911"><a name="p10652112521911"></a><a name="p10652112521911"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.31123112311231%" headers="mcps1.2.4.1.2 "><p id="p126521325141919"><a name="p126521325141919"></a><a name="p126521325141919"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.4966496649665%" headers="mcps1.2.4.1.3 "><p id="p0646410194516"><a name="p0646410194516"></a><a name="p0646410194516"></a>源操作数，shape为二维，LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。</p>
<p id="p104849817"><a name="p104849817"></a><a name="p104849817"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row1652525171911"><td class="cellrowborder" valign="top" width="21.19211921192119%" headers="mcps1.2.4.1.1 "><p id="p1652132541913"><a name="p1652132541913"></a><a name="p1652132541913"></a>UnPadParams</p>
</td>
<td class="cellrowborder" valign="top" width="12.31123112311231%" headers="mcps1.2.4.1.2 "><p id="p10652182519195"><a name="p10652182519195"></a><a name="p10652182519195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.4966496649665%" headers="mcps1.2.4.1.3 "><p id="p10653142515197"><a name="p10653142515197"></a><a name="p10653142515197"></a>UnPad详细参数，UnPadParams数据类型，具体结构体参数说明如下：</p>
<a name="ul18510424132612"></a><a name="ul18510424132612"></a><ul id="ul18510424132612"><li>leftPad，左边unpad的数据量。leftPad要求小于32B。单位：列。当前暂不生效。</li><li>rightPad，右边unpad的数据量。rightPad要求小于32B，大于0。单位：列。当前只支持在右边进行unpad。</li></ul>
<p id="p593016131291"><a name="p593016131291"></a><a name="p593016131291"></a>UnPadParams结构体的定义如下：</p>
<a name="screen192271245293"></a><a name="screen192271245293"></a><pre class="screen" codetype="Cpp" id="screen192271245293">struct UnPadParams {
    uint16_t leftPad = 0;
    uint16_t rightPad = 0;
};</pre>
</td>
</tr>
<tr id="row36528243474"><td class="cellrowborder" valign="top" width="21.19211921192119%" headers="mcps1.2.4.1.1 "><p id="p215910345451"><a name="p215910345451"></a><a name="p215910345451"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.31123112311231%" headers="mcps1.2.4.1.2 "><p id="p20159183474511"><a name="p20159183474511"></a><a name="p20159183474511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.4966496649665%" headers="mcps1.2.4.1.3 "><p id="p815913419455"><a name="p815913419455"></a><a name="p815913419455"></a>共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。共享缓冲区大小的获取方式请参考<a href="UnPad-Tiling.md">UnPad Tiling</a>。</p>
<p id="p14203184218188"><a name="p14203184218188"></a><a name="p14203184218188"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11653172514193"><td class="cellrowborder" valign="top" width="21.19211921192119%" headers="mcps1.2.4.1.1 "><p id="p5653192521919"><a name="p5653192521919"></a><a name="p5653192521919"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.31123112311231%" headers="mcps1.2.4.1.2 "><p id="p1165317256199"><a name="p1165317256199"></a><a name="p1165317256199"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="66.4966496649665%" headers="mcps1.2.4.1.3 "><p id="p1865352511191"><a name="p1865352511191"></a><a name="p1865352511191"></a>计算所需tiling信息，Tiling信息的获取请参考<a href="UnPad-Tiling.md">UnPad Tiling</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section94691236101419"></a>

本样例：Tensor的width已32B对齐，以half为例，如16\*16，进行UnPad，变成16\*15。输入数据类型均为half。

```
#include "kernel_operator.h"

template <typename T>
class KernelUnPad {
public:
    __aicore__ inline KernelUnPad()
    {}
    __aicore__ inline void Init(GM_ADDR dstGm, GM_ADDR srcGm, uint16_t heightIn, uint16_t widthIn, uint16_t oriWidthIn,
        AscendC::UnPadParams &unPadParamsIn, const UnPadTiling &tilingData)
    {
        height = heightIn;
        width = widthIn;
        oriWidth = oriWidthIn;
        unPadParams = unPadParamsIn;
        srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
        dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
        pipe.InitBuffer(inQueueSrcVecIn, 1, height * width * sizeof(T));
        pipe.InitBuffer(inQueueSrcVecOut, 1, height * (width - unPadParams.leftPad - unPadParams.rightPad) * sizeof(T));
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
        AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
        AscendC::DataCopy(srcLocal, srcGlobal, height * width);
        inQueueSrcVecIn.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> dstLocal = inQueueSrcVecIn.DeQue<T>();
        AscendC::LocalTensor<T> srcOutLocal = inQueueSrcVecOut.AllocTensor<T>();
        AscendC::UnPad(srcOutLocal, dstLocal, unPadParams, tiling);
        inQueueSrcVecOut.EnQue(srcOutLocal);
        inQueueSrcVecIn.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> srcOutLocalDe = inQueueSrcVecOut.DeQue<T>();
        AscendC::DataCopy(dstGlobal, srcOutLocalDe, height * (width - unPadParams.leftPad - unPadParams.rightPad));
        inQueueSrcVecOut.FreeTensor(srcOutLocalDe);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
    uint16_t height;
    uint16_t width;
    uint16_t oriWidth;
    AscendC::UnPadParams unPadParams;
    UnPadTiling tiling;
};

extern "C" __global__ __aicore__ void
    kernel_unpad_half_16_16_16(GM_ADDR src_gm, GM_ADDR dst_gm, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelUnPad<half> op;
    AscendC::UnPadParams unPadParams{0, 1};
    op.Init(dst_gm, src_gm, 16, 16, 16, unPadParams, tilingData.unpadTilingData);
    op.Process();
}
```

