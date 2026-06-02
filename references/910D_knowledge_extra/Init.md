# Init<a name="ZH-CN_TOPIC_0000002523304438"></a>

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

用于内存和同步流水事件EventID的初始化。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void Init()
```

## 约束说明<a name="section633mcpsimp"></a>

重复申请释放tpipe，要与[Destroy](Destroy.md)接口成对使用，tpipe如果要重复申请需要先Destroy释放后再Init。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template <typename srcType>
class KernelAsin {
public:
    __aicore__ inline KernelAsin()
    {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t srcSize, TPipe *pipe)
    {
        src_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(src_gm), srcSize);
        dst_global.SetGlobalBuffer(reinterpret_cast<__gm__ srcType *>(dst_gm), srcSize);
        pipe->InitBuffer(inQueueX, 1, srcSize * sizeof(srcType));
        pipe->InitBuffer(outQueue, 1, srcSize * sizeof(srcType));
        bufferSize = srcSize;
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
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal, src_global, bufferSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        int16_t scalar_value = 3; 
        AscendC::Muls(dstLocal, srcLocal, (srcType)scalar_value, bufferSize);
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset)
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dst_global, dstLocal, bufferSize);
        outQueue.FreeTensor(dstLocal);
    }
private:
    AscendC::GlobalTensor<srcType> src_global;
    AscendC::GlobalTensor<srcType> dst_global;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint32_t bufferSize = 0;
};
template <typename dataType>
__aicore__ void kernel_Test_operator(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t srcSize)
{
    KernelAsin<dataType> op;
    AscendC::TPipe pipeIn;   
    pipeIn.Init();
        
    op.Init(src_gm, dst_gm, srcSize, &pipeIn);
    op.Process();
    pipeIn.Destroy();
    AscendC::TPipe pipeCast;
    op.Init(src_gm, dst_gm, srcSize, &pipeCast);
    op.Process();
    pipeCast.Destroy();
}
```

