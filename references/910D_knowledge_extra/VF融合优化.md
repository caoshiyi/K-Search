# VF融合优化<a name="ZH-CN_TOPIC_0000002554329051"></a>

【优先级】高

【描述】VF融合是将代码中多个VF函数融合成一个VF函数，有效提升性能。VF融合特性是优化特性，VF自动融合会借助Loop Fuse算法，将VF转换成Loop形态，然后将控制流等价（Control-Flow-Equivalent）的VF进行融合，最后将VF进行还原。编译器首先会做融合前的合法性检查，判断两个VF是否等价，Main侧中间代码是否能在VF内执行以及融合后是否可产生正收益（不会引起传参寄存器溢出、VF代码不会过大）等，如果满足VF融合条件，编译器会自动执行VF融合优化，为保证融合后的VF执行逻辑与语义与融合前一致，会在原来两个VF之间保守地插入同步指令，编译器还会尝试外提、合并融合后的VF中的指令，对VF代码进行优化。融合策略是能融尽融，用户按照符合融合的合法性检查的模式进行编码，可以增加VF融合的机会。

## VF融合原理介绍<a name="section46641606486"></a>

VF融合优化可分为三个阶段：VF浅度融合、VF深度融合和VF内自动同步：

**VF浅度融合**：编译器首先会分析两个VF的控制流是否等价，构建Cost Model分析是否有正向收益，如果满足VF融合条件，将VF外部的控制流融入到VF内，将VF外的Software Loop硬化成VF内的Hardware Loop，然后使能VF自动融合的基础能力，将两个VF融合成一个VF，为后续的VF深度融合提供基础。

<!-- img2text -->
```text
┌───────────────────────┐      ┌────────────────────────────┐      ┌────────────────────────────┐      ┌────────────────────────────┐
│ for ( ) {             │      │ VF外控制流的融             │      │ 构造VF启动融合             │      │ VF1_2 {                    │
│   VF1 {               │ ───→ │ 入到VF内                   │ ───→ │ 的前置特性                 │ ───→ │   //May be optimized to    │
│     ....              │      └────────────────────────────┘      └────────────────────────────┘      │   Hardware Loop            │
│   }                   │                    ↘                                                    ↘     │   for ( ) {                │
│ }                     │                     ↘                                                     ↘    │   }                        │
│                       │                      ↘                                                      │  │   .... // Insert Membar   │
│ .... // Main Scalar Code│                      ↘                                                     │  │   .... // Vec Scalar Code │
│ for ( ) {             │                       ▼                                                     │  │   for ( ) {               │
│   VF2 {               │      ┌────────────────────────────┐                                         │  │     ....                  │
│     ....              │      │ VF1 {                     │                                         │  │   }                       │
│   }                   │      │   //May be optimized to   │                                         │  │ }                         │
│ }                     │      │   Hardware Loop           │                                         │  └────────────────────────────┘
└───────────────────────┘      │   for ( ) {              │                                         │
                               │     ....                 │                                         │
                               │   }                      │                                         │
                               │                          │                                         │
                               │   .... // Main Scalar Code│                                         │
                               │   VF2 {                  │                                         │
                               │   for ( ) {              │                                         │
                               │     ....                 │                                         │
                               │   }                      │                                         │
                               │   }                      │                                         │
                               └────────────────────────────┘                                         │
                                                                                                     │
                               ┌────────────────────────────┐                                         │
                               │ 融合条件：                 │─────────────────────────────────────────┘
                               │ 1. 两个VF控制流等         │
                               │ 价2. VF中嵌套循环         │
                               │ 以软件&vector指令         │
                               │ 3. Cost Model判断         │
                               │ 融合后产生正收益          │
                               └────────────────────────────┘
```

**VF深度融合**：VF深度融合会继续对VF内的Hardware Loop进行融合，从而减少Hardware Loop的启动开销，并且极大地减少冗余的Load/Store操作，充分复用寄存器。

<!-- img2text -->
```
┌──────────────────────────────┐         融合Hardware Loop         ┌──────────────────────────────┐         优化Ld/St指令         ┌──────────────────────────────┐
│ VF1_2 {                      │ ───────────────────────────────→ │ VF1_2 {                      │ ─────────────────────────→ │ VF1_2 {                      │
│   for (i) {                  │                                  │   for (i) {                  │                           │   for (i) {                  │
│     vecA = load(AddrA[i])    │                                  │     vecA = load(AddrA[i])    │                           │     vecA = load(AddrA[i])    │
│     vecB = load(AddrB[i])    │                                  │     vecB = load(AddrB[i])    │                           │     vecB = load(AddrB[i])    │
│     vecC = vecA + vecB       │                                  │     vecC = vecA + vecB       │                           │     vecC = vecA + vecB       │
│     store(AddrDst[i], vecC)  │                                  │     store(AddrDst[i], vecC)  │                           │     store(AddrDst[i], vecC)  │
│   }                          │                                  │     //...Loop Adjacent Code  │                           │     //...Loop Adjacent Code  │
│   //...Loop Adjacent Code    │                                  │     vecC = load(AddrDst[i])  │                           │     vecC = load(AddrDst[i])  │
│   for (i) {                  │                                  │     vecD = load(AddrD[i])    │                           │     vecD = load(AddrD[i])    │
│     vecC = load(AddrDst[i])  │                                  │     vecE = vecC * vecD       │                           │     vecE = vecC * vecD       │
│     vecD = load(AddrD[i])    │                                  │     store(AddrDst[i], VecE)  │                           │     store(AddrDst[i], VecE)  │
│     vecE = vecC * vecD       │                                  │   }                          │                           │   }                          │
│     store(AddrDst[i], VecE)  │                                  │ }                            │                           │ }                            │
│   }                          │                                  └──────────────────────────────┘                           └──────────────────────────────┘
│ }                            │
└──────────────────────────────┘


右图中被删除/优化的指令：
  store(AddrDst[i], vecC)
  vecC = load(AddrDst[i])
```

**VF内自动同步**：编译器会精准地插入必要的同步指令，删除冗余的同步指令，极大地释放了硬件OOO（Out of Order）能力。用户无需手动插入同步指令，极大地降低了用户的编码难度。

## VF融合编写指导<a name="section1631753714477"></a>

1.  多个VF函数自动融合：如果多个VF函数的控制流等价，且满足均为[Hardware Loop循环](VF循环优化.md#section11326136133217)，编译器会执行VF融合优化特性。

    【正例】VF函数DivVF和AddVF会被编译器融合成一个VF函数，并且能优化多余的Load/Store指令。

    ```
    template<typename T>
    __simd_vf__ inline void DivVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t repeatTime, uint32_t oneRepNum){
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::RegTensor<T> reg0, reg1, reg2;
        constexpr float num = 1.0f;
        for(uint16_t j = 0; j < repeatTime; ++j){
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(reg0, srcAddr + j * oneRepNum);
            AscendC::MicroAPI::Duplicate(reg1, num, mask);
            AscendC::MicroAPI::Div(reg2, reg1, reg0, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr + j * oneRepNum, reg2, mask);
        }
    }
    template<typename T>
    __simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t repeatTime, uint32_t oneRepNum){
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::RegTensor<T> srcReg;
        AscendC::MicroAPI::RegTensor<T> dstReg;
        constexpr float num = 1.0f;
        for(uint16_t j = 0; j < repeatTime; ++j){
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + j * oneRepNum);
            AscendC::MicroAPI::Adds(dstReg, srcReg, num, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr + j * oneRepNum, dstReg, mask);
        }
    }
    template<typename T>
    class Kernel {
        public:
        __aicore__ inline Kernel() = default;
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t count, AscendC::TPipe* pipeIn){
            // ... 
     
        }
        __aicore__ inline void CopyIn(){
            // ... 
        }
        __aicore__ inline void Compute(){
            AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            AscendC::DataCopy(yLocal, xLocal, count);
            __ubuf__ T* srcAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
            __ubuf__ T* dstAddr = reinterpret_cast<__ubuf__ T*>(yLocal.GetPhyAddr());
            constexpr uint32_t oneRepNum = 256 / sizeof(T);
            uint32_t repeatTime =  count / oneRepNum;
            DivVF(dstAddr, srcAddr, count, repeatTime, oneRepNum);
            AddVF(dstAddr, dstAddr, count, repeatTime, oneRepNum);
            outQueueY.EnQue<T>(yLocal);
        }
        __aicore__ inline void CopyOut(){
            // ... 
        }
        __aicore__ inline void Process(){
            CopyIn();
            Compute();
            CopyOut();
        }
        private:
        AscendC::TPipe* pipe = nullptr;
        uint32_t count;
        AscendC::GlobalTensor<T> xGm;
        AscendC::GlobalTensor<T> yGm;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    };
    ```

2.  使用基础API连续计算模式：基础API实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。基础API根据对数据操作方法的不同，可以分为两大类：

    -   连续计算API：支持Tensor前n个数据计算。针对源操作数的连续n个数据进行计算并连续写入目的操作数，解决一维tensor的连续计算问题。
    -   高维切分API：支持Repeat和Stride。功能灵活的计算API，提供与Builtin API完全对等的编程能力，充分发挥硬件优势，支持对每个操作数的DataBlock Stride，Repeat Stride，Mask等参数的操作。

    在VF融合优化中，推荐使用基础API的连续计算模式编写算子，可以充分发挥出VF融合优化的能力，与高维切分API相比，连续计算API使得编译器能更好地分析VF融合优化，更加容易满足VF融合优化的条件，使用基础API的连续计算模式能写出性能更优的算子。

    【反例】使用基础API的高维切分模式编写算子，编译器在分析VF融合时受复杂的计算逻辑影响，无法对Add和Mul接口进行VF融合优化。

    ```
    template<typename T>
    class Kernel {
        public:
        // ...
        __aicore__ inline void Compute(){
            AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            AscendC::DataCopy(yLocal, xLocal, inner * outter);
            uint64_t mask = 128;
            AscendC::Add(yLocal, xLocal, xLocal, mask, 4, { 1, 1, 1, 8, 8, 8 });
            AscendC::Mul(yLocal, yLocal, xLocal, mask, 4, { 1, 1, 1, 8, 8, 8 });
            outQueueY.EnQue<T>(yLocal);
        }
        // ...
    };
    ```

    【正例】使用基础API的连续计算模式，编译器分析Add和Mul函数后符合VF融合要求，将Add和Mul融合成一个VF函数。

    ```
    template<typename T>
    class Kernel {
        public:
        // ...
        __aicore__ inline void Compute(){
            AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
            AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
            AscendC::DataCopy(yLocal, xLocal, inner * outter);
            AscendC::Add(yLocal, xLocal, xLocal, count);
            AscendC::Mul(yLocal, yLocal, xLocal, count);
            outQueueY.EnQue<T>(yLocal);
        }
        // ...
    };
    ```

