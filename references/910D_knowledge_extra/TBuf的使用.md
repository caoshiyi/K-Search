# TBuf的使用<a name="ZH-CN_TOPIC_0000002523129108"></a>

在大多数算子开发时，核函数计算过程需要使用临时内存来存储运算的中间结果，这些中间结果以临时变量表示，临时变量占用的内存可以使用TBuf数据结构来管理，具体介绍请参考[TBuf](TBuf.md)。下文将以输入的数据类型为bfloat16\_t、在单核上运行的Add算子为例，介绍TBuf的使用方式。本样例中介绍的算子完整代码请参见[使用临时内存的Add算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/21_vectoradd_kernellaunch/VectorAddSingleCoreWithTmpbuf)。

在Atlas A2 训练系列产品/Atlas 800I A2 推理产品上，[Add](Add.md)接口不支持对数据类型bfloat16\_t的源操作数进行求和计算。因此，需要先将算子输入的数据类型转换成Add接口支持的数据类型，再进行计算。为保证计算精度，调用[Cast](Cast.md)接口将输入bfloat16\_t类型转换为float类型，再进行Add计算，并在计算结束后将float类型转换回bfloat16\_t类型。

通过以上分析，得到Ascend C  Add算子的设计规格如下：

-   算子类型（OpType）：Add
-   算子输入输出：

    **表 1**  Add算子输入输出规格

    <a name="table4934296305"></a>
    <table><thead align="left"><tr id="row59358913304"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.1"><p id="p5503181819300"><a name="p5503181819300"></a><a name="p5503181819300"></a><strong id="b1850331853010"><a name="b1850331853010"></a><a name="b1850331853010"></a>name</strong></p>
    </th>
    <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.2"><p id="p1550381833017"><a name="p1550381833017"></a><a name="p1550381833017"></a><strong id="b7503171811309"><a name="b7503171811309"></a><a name="b7503171811309"></a>shape</strong></p>
    </th>
    <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.3"><p id="p1950391883014"><a name="p1950391883014"></a><a name="p1950391883014"></a><strong id="b2503111803020"><a name="b2503111803020"></a><a name="b2503111803020"></a>data type</strong></p>
    </th>
    <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.4"><p id="p14503218133015"><a name="p14503218133015"></a><a name="p14503218133015"></a><strong id="b8503141818301"><a name="b8503141818301"></a><a name="b8503141818301"></a>format</strong></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row393589203016"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p1950331810308"><a name="p1950331810308"></a><a name="p1950331810308"></a>x（输入）</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p1950321863013"><a name="p1950321863013"></a><a name="p1950321863013"></a>(1, 2048)</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p1050613178322"><a name="p1050613178322"></a><a name="p1050613178322"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p19503131815305"><a name="p19503131815305"></a><a name="p19503131815305"></a>ND</p>
    </td>
    </tr>
    <tr id="row6935119173013"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p75031182305"><a name="p75031182305"></a><a name="p75031182305"></a>y（输入）</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p16503111873010"><a name="p16503111873010"></a><a name="p16503111873010"></a>(1, 2048)</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p1350691713217"><a name="p1350691713217"></a><a name="p1350691713217"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p1503918103012"><a name="p1503918103012"></a><a name="p1503918103012"></a>ND</p>
    </td>
    </tr>
    <tr id="row59354943016"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p1450316186305"><a name="p1450316186305"></a><a name="p1450316186305"></a>z（输出）</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p1150361853017"><a name="p1150361853017"></a><a name="p1150361853017"></a>(1, 2048)</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p15506017153218"><a name="p15506017153218"></a><a name="p15506017153218"></a>bfloat16_t</p>
    </td>
    <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p1503101813012"><a name="p1503101813012"></a><a name="p1503101813012"></a>ND</p>
    </td>
    </tr>
    </tbody>
    </table>

-   核函数名称：add\_custom
-   使用的主要接口：
    -   DataCopy：数据搬移接口
    -   Cast：矢量精度转换接口
    -   Add：矢量基础算术接口
    -   EnQue、DeQue等接口：Queue队列管理接口

-   算子实现文件名称：add\_custom.cpp

## 算子类实现<a name="zh-cn_topic_0000002201317266_section824984034911"></a>

该样例的CopyIn，CopyOut任务与[基础矢量算子](基础矢量算子.md)相同，Compute任务的具体流程如下图所示。

**图 1**  输入为bfloat16\_t类型的Add计算流程<a name="zh-cn_topic_0000002201317266_fig816618211471"></a>  
<!-- img2text -->
```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ CopyIn                                     Compute                                                          CopyOut          │
│                                                                                                                              │
│  ┆────────────────────────────────────────── Local Memory ───────────────────────────────────────────────────────────────┆   │
│  ┆                                              pipe                                                                    ┆   │
│  ┆                                   Alloc Memory for Queue                                                             ┆   │
│  ┆                                   Alloc Memory for TBuf                                                              ┆   │
│  ┆                                                                                                                      ┆   │
│  ┆    ┌──────────┐   DeQue   ┌─────────┐   Cast   ┌────────────┐                                                        ┆   │
│  ┆    │ inQueueY │ ───────→  │ yLocal  │ ─────→   │ tmpTensor1 │ ───┐                                                    ┆   │
│  ┆    └──────────┘           └─────────┘          └────────────┘   │                                                    ┆   │
│  ┆       bfloat16_t            bfloat16_t              float        │                                                    ┆   │
│  ┆                                                                   ├─ Add ─→ ┌────────────┐ ─→ Cast ─→ ┌─────────┐   ┆   │
│  ┆    ┌──────────┐   DeQue   ┌─────────┐   Cast   ┌────────────┐   │         │ tmpTensor2 │            │ zLocal  │   ┆   │
│  ┆    │ inQueueX │ ───────→  │ xLocal  │ ─────→   │ tmpTensor0 │ ───┘         └────────────┘            └─────────┘   ┆   │
│  ┆    └──────────┘           └─────────┘          └────────────┘                float                   bfloat16_t     ┆   │
│  ┆       bfloat16_t            bfloat16_t              float                                                │ EnQue     ┆   │
│  ┆                                                                                                           ▼           ┆   │
│  ┆                                                                                                      ┌──────────┐     ┆   │
│  ┆                                                                                                      │ outQueueZ│     ┆   │
│  ┆                                                                                                      └──────────┘     ┆   │
│  ┆                                                                                                                      ┆   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

在Compute任务中，表示Cast转换结果、Add计算结果的临时变量均需要使用临时内存存储。与[基础矢量算子实现](基础矢量算子.md#zh-cn_topic_0000002201157438_section10423482111)的KernelAdd算子类相比，本样例新增三个[TBuf](TBuf.md)类型的成员变量tmpBuf0、tmpBuf1、tmpBuf2，用于管理计算过程中使用的临时内存，代码如下。

```
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z){}
    __aicore__ inline void Process(){}
private:
    __aicore__ inline void CopyIn(){}
    __aicore__ inline void Compute(){}
    __aicore__ inline void CopyOut(){}
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;     
    AscendC::GlobalTensor<bfloat16_t> xGm; 
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;
};
```

初始化函数阶段除原有步骤外，需要调用[InitBuffer](InitBuffer.md)接口为TBuf变量分配内存，具体的初始化函数代码如下：

```
 __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    xGm.SetGlobalBuffer((__gm__ half *)x, TOTAL_LENGTH);
    yGm.SetGlobalBuffer((__gm__ half *)y, TOTAL_LENGTH);
    zGm.SetGlobalBuffer((__gm__ half *)z, TOTAL_LENGTH);

    pipe.InitBuffer(inQueueX, 1, TOTAL_LENGTH * sizeof(bfloat16_t));
    pipe.InitBuffer(inQueueY, 1, TOTAL_LENGTH * sizeof(bfloat16_t));
    pipe.InitBuffer(outQueueZ, 1, TOTAL_LENGTH * sizeof(bfloat16_t));
 
    pipe.InitBuffer(tmpBuf0, TOTAL_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpBuf1, TOTAL_LENGTH * sizeof(float));
    pipe.InitBuffer(tmpBuf2, TOTAL_LENGTH * sizeof(float));
}
```

基于矢量编程范式，核函数需要实现3个基本任务：CopyIn，Compute，CopyOut。与[基础矢量算子实现](基础矢量算子.md#zh-cn_topic_0000002201157438_section10423482111)相同，Process函数按顺序调用CopyIn函数，Compute函数，CopyOut函数。其中，CopyIn函数，CopyOut函数与[基础矢量算子的CopyIn函数](基础矢量算子.md#zh-cn_topic_0000002201157438_zh-cn_topic_0000001514531781_li10182173751518)、[基础矢量算子的CopyOut函数](基础矢量算子.md#zh-cn_topic_0000002201157438_zh-cn_topic_0000001514531781_li1134112320247)的实现没有差异，此处不过多赘述。Compute函数的实现步骤如下：

1.  使用[DeQue](DeQue.md)从VECIN的Queue中取出LocalTensor。
2.  使用TBuf.[Get](Get.md)从TBuf上获取全部长度的Tensor作为临时内存。
3.  使用[Cast](Cast.md)接口将LocalTensor转换为float类型，并存入临时内存。
4.  使用[Add](Add.md)接口完成矢量计算，将计算结果存入临时内存。
5.  使用[Cast](Cast.md)接口将临时内存中的计算结果转换为bfloat16\_t类型。
6.  使用[EnQue](EnQue.md)将bfloat16\_t类型的结果LocalTensor放入VECOUT的Queue中。
7.  使用[FreeTensor](FreeTensor.md)释放不再使用的LocalTensor。

```
__aicore__ inline void Compute()
{
    AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t> ();
    AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t> ();
    AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t> ();
 
    AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
    AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> tmpTensor2 = tmpBuf2.Get<float>();
    AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
    AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
 
    AscendC::Add(tmpTensor2, tmpTensor0, tmpTensor1, TOTAL_LENGTH);
    AscendC::Cast(zLocal, tmpTensor2, AscendC::RoundMode::CAST_RINT, TOTAL_LENGTH);
 
    outQueueZ.EnQue<bfloat16_t>(zLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
}
```

