# DoubleBuffer场景<a name="ZH-CN_TOPIC_0000002554408997"></a>

因存在算子中多次搬入搬出数据的场景，为充分利用硬件资源，实现多流水并行，引入[DoubleBuffer](DoubleBuffer.md)机制。[DoubleBuffer](DoubleBuffer.md)是通过将输入数据分成大小相等的两块，充分利用AI Core的硬件资源，实现数据搬入、计算、数据搬出的并行执行方式。下面以“核间不均分，核内不均分”的样例为例，介绍算子中DoubleBuffer的实现，完整样例代码请参见[使用DoubleBuffer的Add算子样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/21_vectoradd_kernellaunch/VectorAddMultiCoreWithTiling)。

**图 1**  DoubleBuffer数据切分示意图<a name="zh-cn_topic_0000002236197681_fig68713182104"></a>  
<!-- img2text -->
```
                                      TOTAL_LENGTH
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                              │
├────────────────┬────────────────┬────────────────┬────────────────┬────────────────┬────────────────┬────────────────┬────────────────┤
│                │                │                │                │                │                │                │                │
└────────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘
└──────────────┘
  BLOCK_LENGTH

       │
       ↓
     Tiling

┌──┬──┬──┬──┬──┬──┬──┬──┐
│  │  │  │  │  │  │  │  │
└──┴──┴──┴──┴──┴──┴──┴──┘

       │
       ↓
  DoubleBuffer

┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
```

## Tiling实现<a name="zh-cn_topic_0000002236197681_section1967484164119"></a>

使能DoubleBuffer后，每一个数据块会分成大小相等的两块，因此，若要使能DoubleBuffer，要求数据总量应该能够均分。为了简化处理，将可用的Unified Buffer空间以32字节为粒度，分成n块dataBlock，如果n不是偶数，则减1，这样就可以保证一套代码兼容开启或不开启DoubleBuffer功能。对应步骤如下：

1.  判断数据总长度totalLength是否满足32字节对齐，如不满足，则计算totalLength向上32字节对齐后的长度totalLengthAligned。

    ```
    constexpr uint32_t BLOCK_SIZE = 32;
    // 为方便计算，这里根据数据类型定义变量alignNum作为对齐数
    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
    // totalLength为数据总量
    uint32_t totalLengthAligned = (totalLength % alignNum == 0)?
            totalLength : ((totalLength + alignNum - 1) / alignNum) * alignNum;
    ```

2.  根据totalLengthAligned，计算每个核的计算数据长度blockLength，分核策略可参照[尾核Tiling](尾核Tiling.md)。
3.  计算其余Tiling参数。

    对当前Unified Buffer可用空间以32字节为粒度，进行切分，计算出数据块个数UB\_BLOCK\_NUM。根据是否开启DoubleBuffer计算出当前可用的最大数据块个数，记作MAX\_AVAILABLE\_UB\_BLOCK\_NUM。最后，以MAX\_AVAILABLE\_UB\_BLOCK\_NUM为粒度，对blockLength进行切分。为方便演示，如下代码直接给出UB\_BLOCK\_NUM，作为当前Unified Buffer可用空间包含的block（32字节）数。

    ```
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t UB_BLOCK_NUM = 21;  // UB最大可以使用的block数量
    constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;
    
    tileNum = blockLength / (alignNum * MAX_AVAILABLE_UB_BLOCK_NUM);
    if (tileNum == 0) {
        // 单核需要计算的长度小于UB可用空间，按照仅有尾块处理
        tileLength = 0;
        lastTileLength = (blockLength + alignNum - 1) / alignNum * alignNum;
    } else if ((blockLength / alignNum) % MAX_AVAILABLE_UB_BLOCK_NUM == 0) {
        // 单核的计算量能被当前可用UB空间均分，仅有主块，无尾块
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = 0;
    } else {
        // 同时有主块和尾块
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = blockLength - tileNum * tileLength;
    }
    ```

## 算子类实现<a name="zh-cn_topic_0000002236197681_section09641704120"></a>

不开启DoubleBuffer时，只需要对每个核上最后一个分块的起始地址做处理；开启DoubleBuffer后，需要处理的数据块长度变成原来的一半，所以需要对最后两个数据块的起始地址做处理。

开启DoubleBuffer，参考[InitBuffer接口函数原型](InitBuffer.md#li1365924755416)，将num参数配置成2，即BUFFER\_NUM。

```
this->initBufferLength = AscendC::Std::max(this->tileLength, this->lastTileLength);
pipe.InitBuffer(inQueueX, BUFFER_NUM, this->initBufferLength * sizeof(dataType));
pipe.InitBuffer(inQueueY, BUFFER_NUM, this->initBufferLength * sizeof(dataType));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->initBufferLength * sizeof(dataType));
```

同时在计算核内每个数据块的长度时，考虑DoubleBuffer场景，需要将Buffer数量，即BUFFER\_NUM=2带入计算。

```
this->tileLength = tiling.tileLength / BUFFER_NUM;
```

由于无法保证尾块满足DoubleBuffer的条件，因此不对尾块进行切分。

```
this->lastTileLength = tiling.lastTileLength;
```

Init函数实现代码如下：

```
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    if (tiling.isEvenCore) {
        this->blockLength = tiling.blockLength;
        this->tileNum = tiling.tileNum;
        this->tileLength = tiling.tileLength / BUFFER_NUM;
        this->lastTileLength = tiling.lastTileLength;

        xGm.SetGlobalBuffer((__gm__ dataType *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dataType *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dataType *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
    } else {
        if (AscendC::GetBlockIdx() < tiling.formerNum) {
            this->tileNum = tiling.formerTileNum;
            this->tileLength = tiling.formerTileLength / BUFFER_NUM;
            this->lastTileLength = tiling.formerLastTileLength;

            xGm.SetGlobalBuffer((__gm__ dataType *)x + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)y + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
        } else {
            this->tileNum = tiling.tailTileNum;
            this->tileLength = tiling.tailTileLength / BUFFER_NUM;
            this->lastTileLength = tiling.tailLastTileLength;

            xGm.SetGlobalBuffer((__gm__ dataType *)x + tiling.formerLength * tiling.formerNum +
                tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)y + tiling.formerLength * tiling.formerNum +
                tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * tiling.formerNum +
                tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
        }
    }

    uint32_t initBufferLength = AscendC::Std::max(this->tileLength, this->lastTileLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, initBufferLength * sizeof(dataType));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, initBufferLength * sizeof(dataType));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, initBufferLength * sizeof(dataType));
}
```

由于开启DoubleBuffer后，切分后的主块数据块个数翻倍，在Process函数中，需要将BUFFER\_NUM带入计算循环次数；尾块独立计算，不开启DoubleBuffer。后续主尾块在CopyIn、Compute、CopyOut函数中的处理，与[尾块tiling处理](尾块Tiling.md)相同。

```
__aicore__ inline void Process()
{
    // 主块进行DoubleBuffer计算，所以loopCount得乘以2
    uint32_t loopCount = this->tileNum * BUFFER_NUM;
    for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn(i, this->tileLength);
        Compute(i, this->tileLength);
        CopyOut(i, this->tileLength);
    }
    // 尾块进行计算, 不做DoubleBuffer操作
    if (this->lastTileLength > 0U) {
        CopyIn(loopCount, this->lastTileLength);
        Compute(loopCount, this->lastTileLength);
        CopyOut(loopCount, this->lastTileLength);
    }
}
```

