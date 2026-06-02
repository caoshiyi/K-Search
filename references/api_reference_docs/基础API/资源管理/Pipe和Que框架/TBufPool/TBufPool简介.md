# TBufPool简介

**页面ID:** atlasascendc_api_07_0122  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0122.html

---

TPipe可以管理全局内存资源，而TBufPool可以手动管理或复用Unified Buffer/L1 Buffer物理内存，主要用于多个stage计算中Unified Buffer/L1 Buffer物理内存不足的场景。

完整样例链接可以参考[TBufPool样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/2_features/2_tbufpool)。

#### 功能图示

下图展示了资源池划分的过程：

1. 通过TPipe::InitBuffer接口可以申请Buffer内存并使用队列进行管理；
2. 通过TPipe::InitBufPool可以划分出资源池BufPool1；
3. 通过TPipe::InitBufPool可以指定BufPool1与BufPool3地址和长度复用;
4. 通过TBufPool::InitBuffer及TBufPool::InitBufPool接口继续将BufPool1及BufPool3划分成Buffer或TBufPool资源池。

**图1 **BufPool资源池划分
<!-- img2text -->
```text
                               ┌──────────────────────┐  ┌──────────────────────┐
                               │         que4         │  │         que5         │
                               └──────────────────────┘  └──────────────────────┘

                               ┌──────────────────────────────────────────────────┐
                               │           BufPool3，指定BufPool1 同地址           │
                               └──────────────────────────────────────────────────┘

                                                       ┌──────────────┐  ┌──────────────┐
                                                       │     que2     │  │     que3     │
                                                       └──────────────┘  └──────────────┘

                               ┌──────────┐  ┌──────────┐  ┌──────────────────────┐
                               │   que1   │  │   TBuf   │  │      BufPool_2       │
                               └──────────┘  └──────────┘  └──────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────────────────────────────────────────────────┐
│ queVecin │  │ queVecOut│  │                      BufPool_1                       │
└──────────┘  └──────────┘  └──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                           Default Buffer Pool(TPipe)                            │
└──────────────────────────────────────────────────────────────────────────────────┘
```

如图示的嵌套关系，最外层TBufPool(BufPool1与BufPool3)需要通过TPipe::InitBufPool申请并初始化，内层TBufPool(BufPool2)可以通过TBufPool::InitBufPool申请并初始化。

#### 约束说明

1. TBufPool必须通过TPipe::InitBufPool或TBufPool::InitBufPool接口进行划分和初始化；资源池只能整体划分成部分，无法部分拼接为整体；
2. 不同TBufPool资源池切换进行计算时，需要调用TBufPool::Reset()接口清空已完成计算的TBufPool，清空后的TBufPool资源池及分配的Buffer和数据默认无效；
3. 不同资源池间分配的Buffer无法混用避免数据踩踏；
4. AllocTensor/FreeTensor、EnQue/DeQue在切分TBufPool资源池时必须成对匹配使用，自动确保同步；
5. 切换资源池的时候，若手写同步，Ascend C不保证地址读写复用同步，因此不推荐手写同步。
