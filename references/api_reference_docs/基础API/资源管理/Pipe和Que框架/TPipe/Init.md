# Init

**页面ID:** atlasascendc_api_07_0111  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0111.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

用于内存和同步流水事件EventID的初始化。

#### 函数原型

```
__aicore__ inline void Init()
```

#### 约束说明

重复申请释放tpipe，要与Destroy接口成对使用，tpipe如果要重复申请需要先Destroy释放后再Init。

#### 调用示例

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
