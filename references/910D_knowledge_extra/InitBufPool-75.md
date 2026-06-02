# InitBufPool<a name="ZH-CN_TOPIC_0000002554423513"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p8360101617570"><a name="p8360101617570"></a><a name="p8360101617570"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

通过Tpipe::InitBufPool接口可划分出整块资源，整块TbufPool资源可以继续通过TBufPool::InitBufPool接口划分成小块资源。

## 函数原型<a name="section620mcpsimp"></a>

-   非共享模式

    ```
    template <class T>
    __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len)
    ```

-   共享模式

    ```
    template <class T, class U>
    __aicore__ inline bool InitBufPool(T& bufPool, uint32_t len, U& shareBuf)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1550165916920"></a>
<table><thead align="left"><tr id="row115015591391"><th class="cellrowborder" valign="top" width="12.139999999999999%" id="mcps1.2.3.1.1"><p id="p12501159099"><a name="p12501159099"></a><a name="p12501159099"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="87.86%" id="mcps1.2.3.1.2"><p id="p85019592918"><a name="p85019592918"></a><a name="p85019592918"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row1550117591914"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p185019592913"><a name="p185019592913"></a><a name="p185019592913"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p12101541625"><a name="p12101541625"></a><a name="p12101541625"></a>bufPool<span>参数的类型。</span></p>
</td>
</tr>
<tr id="row758993611212"><td class="cellrowborder" valign="top" width="12.139999999999999%" headers="mcps1.2.3.1.1 "><p id="p115897367213"><a name="p115897367213"></a><a name="p115897367213"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="87.86%" headers="mcps1.2.3.1.2 "><p id="p12589153615212"><a name="p12589153615212"></a><a name="p12589153615212"></a>shareBuf参数的类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  InitBufPool\(T& bufPool, uint32\_t len\) 原型定义参数说明

<a name="table1960411494236"></a>
<table><thead align="left"><tr id="row17605204922320"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p6605649122315"><a name="p6605649122315"></a><a name="p6605649122315"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.2"><p id="p17605949182313"><a name="p17605949182313"></a><a name="p17605949182313"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p17605184915230"><a name="p17605184915230"></a><a name="p17605184915230"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row260544916231"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p104214518241"><a name="p104214518241"></a><a name="p104214518241"></a>bufPool</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p1842145172415"><a name="p1842145172415"></a><a name="p1842145172415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p164211556249"><a name="p164211556249"></a><a name="p164211556249"></a>新划分的资源池，类型为TBufPool。</p>
</td>
</tr>
<tr id="row03336319398"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1042110572411"><a name="p1042110572411"></a><a name="p1042110572411"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p1342115582416"><a name="p1342115582416"></a><a name="p1342115582416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p154210516247"><a name="p154210516247"></a><a name="p154210516247"></a>新划分资源池长度，单位为字节，非32字节对齐会自动向上补齐至32字节对齐。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  InitBufPool\(T& bufPool, uint32\_t len, U& shareBuf\) 原型定义参数说明

<a name="table5376122715308"></a>
<table><thead align="left"><tr id="row1337716275309"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p1537762711305"><a name="p1537762711305"></a><a name="p1537762711305"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.370000000000001%" id="mcps1.2.4.1.2"><p id="p153771127123013"><a name="p153771127123013"></a><a name="p153771127123013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.27000000000001%" id="mcps1.2.4.1.3"><p id="p17377162715303"><a name="p17377162715303"></a><a name="p17377162715303"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row19377627133012"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p55733283248"><a name="p55733283248"></a><a name="p55733283248"></a>bufPool</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p1357372832410"><a name="p1357372832410"></a><a name="p1357372832410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p1457322822419"><a name="p1457322822419"></a><a name="p1457322822419"></a>新划分的资源池，类型为TBufPool。</p>
</td>
</tr>
<tr id="row13377162793019"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1357312818241"><a name="p1357312818241"></a><a name="p1357312818241"></a>len</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p857319288247"><a name="p857319288247"></a><a name="p857319288247"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p17573122810249"><a name="p17573122810249"></a><a name="p17573122810249"></a>新划分资源池长度，单位为字节，非32字节对齐会自动向上补齐至32字节对齐。</p>
</td>
</tr>
<tr id="row1371133216245"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1417710361241"><a name="p1417710361241"></a><a name="p1417710361241"></a>shareBuf</p>
</td>
<td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.2.4.1.2 "><p id="p15177113615246"><a name="p15177113615246"></a><a name="p15177113615246"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.27000000000001%" headers="mcps1.2.4.1.3 "><p id="p11771236152417"><a name="p11771236152417"></a><a name="p11771236152417"></a>被复用资源池，类型为TBufPool，新划分资源池与被复用资源池共享起始地址及长度。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

1.  新划分的资源池与被复用资源池的物理内存需要一致，两者共享起始地址及长度；
2.  输入长度需要小于等于被复用资源池长度；
3.  其他泛用约束参考[TBufPool](TBufPool.md)；

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

数据量较大且内存有限时，无法一次完成所有数据搬运，需要拆分成多个阶段计算，每次计算使用其中的一部分数据，可以通过TBufPool资源池进行内存地址复用。本例中，从Tpipe划分出资源池tbufPool0，tbufPool0为src0Gm分配空间后，继续分配了资源池tbufPool1，指定tbufPool1与tbufPool2复用并分别运用于第一、二轮计算，此时tbufPool1及tbufPool2共享起始地址及长度。

```
class ResetApi {
public:
    __aicore__ inline ResetApi() {}
    __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
    {
        src0Global.SetGlobalBuffer((__gm__ half*)src0Gm);
        src1Global.SetGlobalBuffer((__gm__ half*)src1Gm);
        dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
        pipe.InitBufPool(tbufPool0, 131072);
        tbufPool0.InitBuffer(srcQue0, 1, 65536); // Total src0
        tbufPool0.InitBufPool(tbufPool1, 65536);
        tbufPool0.InitBufPool(tbufPool2, 65536, tbufPool1);
    }
    __aicore__ inline void Process()
    {
        tbufPool1.InitBuffer(srcQue1, 1, 32768);
        tbufPool1.InitBuffer(dstQue0, 1, 32768);
        CopyIn();
        Compute();
        CopyOut();
        tbufPool1.Reset();
        tbufPool2.InitBuffer(srcQue2, 1, 32768);
        tbufPool2.InitBuffer(dstQue1, 1, 32768);
        CopyIn1();
        Compute1();
        CopyOut1();
        tbufPool2.Reset();
        tbufPool0.Reset();
        pipe.Reset();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> src0Local = srcQue0.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = srcQue1.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global, 16384);
        AscendC::DataCopy(src1Local, src1Global, 16384);
        srcQue0.EnQue(src0Local);
        srcQue1.EnQue(src1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> src0Local = srcQue0.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = srcQue1.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = dstQue0.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 16384);
        dstQue0.EnQue<half>(dstLocal);
        srcQue0.FreeTensor(src0Local);
        srcQue1.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> dstLocal = dstQue0.DeQue<half>();
        AscendC::DataCopy(dstGlobal, dstLocal, 16384);
        dstQue0.FreeTensor(dstLocal);
    }
    __aicore__ inline void CopyIn1()
    {
        AscendC::LocalTensor<half> src0Local = srcQue0.AllocTensor<half>();
        AscendC::LocalTensor<half> src1Local = srcQue2.AllocTensor<half>();
        AscendC::DataCopy(src0Local, src0Global[16384], 16384);
        AscendC::DataCopy(src1Local, src1Global[16384], 16384);
        srcQue0.EnQue(src0Local);
        srcQue2.EnQue(src1Local);
    }
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<half> src0Local = srcQue0.DeQue<half>();
        AscendC::LocalTensor<half> src1Local = srcQue2.DeQue<half>();
        AscendC::LocalTensor<half> dstLocal = dstQue1.AllocTensor<half>();
        AscendC::Add(dstLocal, src0Local, src1Local, 16384);
        dstQue1.EnQue<half>(dstLocal);
        srcQue0.FreeTensor(src0Local);
        srcQue2.FreeTensor(src1Local);
    }
    __aicore__ inline void CopyOut1()
    {
        AscendC::LocalTensor<half> dstLocal = dstQue1.DeQue<half>();
        AscendC::DataCopy(dstGlobal[16384], dstLocal, 16384);
        dstQue1.FreeTensor(dstLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBufPool<AscendC::TPosition::VECCALC> tbufPool0, tbufPool1, tbufPool2;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> srcQue0, srcQue1, srcQue2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> dstQue0, dstQue1;
    AscendC::GlobalTensor<half> src0Global, src1Global, dstGlobal;
};
extern "C" __global__ __aicore__ void tbufpool_kernel(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm)
{
    ResetApi op;
    op.Init(src0Gm, src1Gm, dstGm);
    op.Process();
}
```

