# RepeatReduceSum<a name="ZH-CN_TOPIC_0000002554424239"></a>

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

每个repeat内所有数据求和。和[WholeReduceSum](WholeReduceSum.md)接口相比，不支持mask逐bit模式。建议使用功能更全面的[WholeReduceSum](WholeReduceSum.md)接口。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, bool isSetMask = true, typename U = T>
__aicore__ inline void RepeatReduceSum(const LocalTensor<U>& dst, const LocalTensor<T>& src, const int32_t repeatTime, const int32_t mask, const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride, const int32_t srcRepStride);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.51%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.49%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.51%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.49%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>源操作数数据类型。</p>
</td>
</tr>
<tr id="row1068213344417"><td class="cellrowborder" valign="top" width="13.51%" headers="mcps1.2.3.1.1 "><p id="p568211318441"><a name="p568211318441"></a><a name="p568211318441"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="86.49%" headers="mcps1.2.3.1.2 "><p id="p196822311447"><a name="p196822311447"></a><a name="p196822311447"></a>目的操作数数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001429830437_row18835145716587"><td class="cellrowborder" valign="top" width="13.51%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p1383515717581"><a name="zh-cn_topic_0000001429830437_p1383515717581"></a><a name="zh-cn_topic_0000001429830437_p1383515717581"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="86.49%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="9.100910091009101%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.23772377237724%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p4428175618426"><a name="p4428175618426"></a><a name="p4428175618426"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p2428856174212"><a name="p2428856174212"></a><a name="p2428856174212"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p15798114920385"><a name="p15798114920385"></a><a name="p15798114920385"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p166113131185"><a name="p166113131185"></a><a name="p166113131185"></a>LocalTensor的起始地址需要保证2字节对齐（针对half数据类型），4字节对齐（针对float数据类型）。</p>
<p id="p13545202711445"><a name="p13545202711445"></a><a name="p13545202711445"></a><span id="ph1054562704411"><a name="ph1054562704411"></a><a name="ph1054562704411"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint32_t/int32_t/half/float</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10429155616425"><a name="p10429155616425"></a><a name="p10429155616425"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p164291756114215"><a name="p164291756114215"></a><a name="p164291756114215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p48001610153912"><a name="p48001610153912"></a><a name="p48001610153912"></a>源操作数。</p>
<p id="p434181318395"><a name="p434181318395"></a><a name="p434181318395"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p173701626882"><a name="p173701626882"></a><a name="p173701626882"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p146062610199"><a name="p146062610199"></a><a name="p146062610199"></a>针对<span id="ph137353553017"><a name="ph137353553017"></a><a name="ph137353553017"></a>Ascend 950PR/Ascend 950DT</span>：src数据类型uint16_t和int16_t时，dst数据类型分别为uint32_t和int32_t，其他情况下，dst数据类型均与src数据类型相同。</p>
<p id="p118141336182515"><a name="p118141336182515"></a><a name="p118141336182515"></a><span id="ph6654142184717"><a name="ph6654142184717"></a><a name="ph6654142184717"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t/int16_t/uint32_t/int32_t/half/float</p>
</td>
</tr>
<tr id="row103306116356"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1756245258"><a name="p1756245258"></a><a name="p1756245258"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p127561346255"><a name="p127561346255"></a><a name="p127561346255"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p353564621520"><a name="p353564621520"></a><a name="p353564621520"></a>重复迭代次数。取值范围为[0, 255]。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row167719363129"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p42671442191214"><a name="p42671442191214"></a><a name="p42671442191214"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p182672042171211"><a name="p182672042171211"></a><a name="p182672042171211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p2267164251213"><a name="p2267164251213"></a><a name="p2267164251213"></a>用于控制每次迭代内连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]。</p>
</td>
</tr>
<tr id="row242517561320"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p10544141112130"><a name="p10544141112130"></a><a name="p10544141112130"></a>dstBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p4544141113139"><a name="p4544141113139"></a><a name="p4544141113139"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p1854411161315"><a name="p1854411161315"></a><a name="p1854411161315"></a>此参数无效，可以配置任意值。</p>
</td>
</tr>
<tr id="row6301859135119"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p53802038141311"><a name="p53802038141311"></a><a name="p53802038141311"></a>srcBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p3380103813134"><a name="p3380103813134"></a><a name="p3380103813134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p14215346174119"><a name="p14215346174119"></a><a name="p14215346174119"></a>单次迭代内datablock的地址步长。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row1261863091320"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p7911733161316"><a name="p7911733161316"></a><a name="p7911733161316"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p17911433171310"><a name="p17911433171310"></a><a name="p17911433171310"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p102993315413"><a name="p102993315413"></a><a name="p102993315413"></a>目的操作数相邻迭代间的地址步长。以一个repeat归约后的长度为单位。</p>
<p id="p182051291943"><a name="p182051291943"></a><a name="p182051291943"></a>单位为dst数据类型所占字节长度。比如当dst为half时，单位为2Bytes。</p>
</td>
</tr>
<tr id="row5250192917342"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p197977615560"><a name="p197977615560"></a><a name="p197977615560"></a>srcRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.100910091009101%" headers="mcps1.2.4.1.2 "><p id="p1479756155613"><a name="p1479756155613"></a><a name="p1479756155613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.23772377237724%" headers="mcps1.2.4.1.3 "><p id="p1624214011488"><a name="p1624214011488"></a><a name="p1624214011488"></a>源操作数相邻迭代间的地址步长，即源操作数每次迭代跳过的datablock数目。详细说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section5468191312484"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

-   对于RepeatReduceSum，其内部的相加方式采用二叉树方式，两两相加：

    假设源操作数为128个half类型的数据\[data0,data1,data2...data127\]，一个repeat可以计算完，计算过程如下。

    1.  data0和data1相加得到data00，data2和data3相加得到data01...data124和data125相加得到data62，data126和data127相加得到data63；
    2.  data00和data01相加得到data000，data02和data03相加得到data001...data62和data63相加得到data031；
    3.  以此类推，得到目的操作数为1个half类型的数据\[data\]。

    需要注意的是两两相加的计算过程中，计算结果大于65504时结果保存为65504。例如源操作数为\[60000,60000,-30000,100\]，首先60000+60000溢出，结果为65504，第二步计算-30000+100=-29900，第四步计算65504-29900=35604。

## 调用示例<a name="section642mcpsimp"></a>

本样例中只展示Compute流程中的部分代码。完整样例可参考[repeat\_reduce\_sum样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/repeat_reduce_sum)

-   样例片段

    ```
    // dstLocal，srcLocal均为half类型，srcLocal的计算数据量为512，连续排布，计算结果也需要连续排布，使用tensor高维切分计算接口，设定mask为最多的128个全部元素参与计算
    // 根据以上信息，推断出repeat为4，dstRepStride为1，srcBlkStride为1，srcRepStride为8，dstBlkStride无效，此处配置0，因此接口示例为：
    int mask = 128; // 参与计算的有效数字
    int repeat = 4; // repeat = 512 / mask
    AscendC::RepeatReduceSum<half>(dstLocal, srcLocal, repeat, mask, 0, 1, 1, 8);
    ```

-   完整样例

    ```
    #include "kernel_operator.h"
    class KernelReduce {
    public:
        __aicore__ inline KernelReduce() {}
        __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)src);
            dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    
            repeat = srcDataSize / mask;
            pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
            pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
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
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
    
            AscendC::RepeatReduceSum<half>(dstLocal, srcLocal, repeat, mask, 0, 1, 1, 8);
    
            outQueueDst.EnQue<half>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
            AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
            outQueueDst.FreeTensor(dstLocal);
        }
    
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
        AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
        int srcDataSize = 2048;
        int dstDataSize = 16;
        int mask = 128;
        int repeat = 0;
    };
    extern "C" __global__ __aicore__ void reduce_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelReduce op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

    结果示例如下：

    ```
    输入数据(src_gm):
    [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
     ...
     3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
     3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
    ]
    输出数据(dst_gm):
    [128 256 ... 384]
    ```

    ```
    输入数据(src_gm):
    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1.]
    输出数据(dst_gm):
    [128. 128. 128. 128. 128. 128. 128. 128. 128. 128. 128. 128. 128. 128.
     128. 128.]
    ```

