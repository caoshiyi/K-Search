# Reg矢量计算编程<a name="ZH-CN_TOPIC_0000002554351501"></a>

## 简介<a name="section1825793122916"></a>

Reg矢量计算API是面向RegBase架构开发的API，用户可以通过该API直接对芯片中涉及Vector计算的寄存器进行操作，实现更大的灵活性和更好的性能。Reg矢量计算API与基础API功能相似，但与基础API输入和输出数据必须为LocalTensor不同，Reg矢量计算API的输入或输出数据均为Reg矢量计算寄存器。对于计算类API，其功能是从给定的[寄存器](寄存器数据类型.md)获取数据，进行计算，并将结果保存在给定的寄存器。对于搬运类API，其功能是实现UB和寄存器的数据搬运。由此可见，Reg矢量计算API相较于基础API，将数据搬运和Reg计算过程交给用户自主控制，从而实现更大的开发自由度。

## Regbase编程模型<a name="section1977234218287"></a>

基于**寄存器（Regbase）**的编程模型支持用户编写和调用Vector Funtion（向量函数）。这些函数使用\_\_simd\_vf\_\_标记，并被发送到硬件中的向量运算单元执行。在simd vf函数内部，通过Reg矢量计算API实现计算操作，其内存层级与编程架构如[图1](#fig597318712265)所示。

在SIMD Vector的内存架构中，最靠近Vector计算单元的是VF Reg，它是SIMD的私有内存，包含多种类型的Reg矢量计算寄存器，用于存放并行处理的多个数据元素。单核内所有的VF Reg寄存器共享一个本地内存资源UB。SIMD架构不支持从全局内存（Global Memory）加载数据到Reg矢量计算寄存器，先将数据从全局内存GM搬运至Unified Buffer，再通过显式的Load/Store指令，由Unified Buffer加载到Reg矢量计算寄存器中。

**图 1**  SIMD Reg矢量计算内存层级<a name="fig597318712265"></a>  
<!-- img2text -->
```text
                         SIMD Cube内容            SIMD Vector内容                SIMD内容
                                              ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                            │
│  ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐        │
│  │    L0A/L0B/          │     │     VF Reg(256B)     │     │     通用寄存器（私有） │        │
│  │     L0C Buffer       │     │       （私有）        │     │                      │        │
│  └──────────────────────┘     └──────────────────────┘     └──────────────────────┘        │
│                                                                                            │
│  ┌──────────────────────┐     ┌──────────────────────────────────────────────────────┐      │
│  │      L1 Buffer       │     │                    UB（共享）                         │      │
│  └──────────────────────┘     └──────────────────────────────────────────────────────┘      │
│                                                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐      │
│  │                                Global Memory                                      │      │
│  └────────────────────────────────────────────────────────────────────────────────────┘      │
│                                                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                              └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

SIMD Reg矢量计算编程架构中，通过发出指令到Reg矢量计算执行单元，执行单元从Registers读取数据，进行计算，计算结果写回Registers。DMA搬运单元负责在Registers和Local Memory之间搬运数据。

**图 2**  SIMD Reg矢量计算编程架构<a name="fig116711840123520"></a>  
<!-- img2text -->
```text
┌──────────────────────────────────────────────────────────────────────────────────────┐     ┌──────────────────────────────────────┐
│ AI Core                                                                              │     │            指令发射单元                │
│                                                                                      │     │             AUX Scalar               │
│  指令流 →                                                                             │     └──────────────────────────────────────┘
│  数据流 →                                                                             │                        │
│  同步信号 →                                                                           │                        ↓
│                                                                                      │              ┌────────────────────┐
│                  ┌──────────────┐                                                    │              │                    │
│                  │ 算子指令序列 │                                                    │              │                    │
│                  │              │                                                    │              │                    │
│                  │              │                                                    │              │                    │
│                  │              │                                                    │              └────────────────────┘
│                  └──────┬───────┘                                                    │                        │
│                         │                                                            │                        ↓
│                  ┌──────▼──────┐                                                    │     ┌──────────────────────────────────────┐
│                  │ Scalar计算单元│──────────────────────────────────────────────┐     │     │                                      │
│                  └──────┬──────┘                                              │     │     │  ┌──────────────┐  ┌──────────┐    │
│                         │                                                     │     │     │  │ Reg矢量      │  │   DMA    │    │
│                         │                                                     │     │     │  │ 计算执行     │  │  单元    │    │
│                         │                                                     │     │     │  │ 单元         │  │          │    │
│      ┌────────┐   ┌─────▼─────┐   ┌────────┐   ┌───────────┐   ┌────────┐     │     │     │  └──────┬───────┘  └────┬─────┘    │
│      │指令队列│   │ Vector计算单元│   │指令队列│   │ Cube计算单元 │   │指令队列│     │     │     │         │                 │          │
│      └────────┘   └───────────┘   └────────┘   └───────────┘   └────────┘     │     │     │         ↓                 ↓          │
│                                                                 ┌───────────┐   │     │     │   ┌────────────────────────────┐   │
│                                                                 │ DMA搬运单元 │   │     │     │   │         Registers          │   │
│                                                                 └───────────┘   │     │     │   └────────────────────────────┘   │
│                                                                                  │     │     │                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │     │     └──────────────────────────────────────┘
│  │                                 Local Memory                                │  │     │                        │
│  └──────────────────────────────────────────────────────────────────────────────┘  │     │                        ↓
└──────────────────────────────────────────────────────────────────────────────────────┘     │            ┌────────────────────┐
           │                                                         │                        │            │    Local Memory    │
           ↓                                                         ↓                        │            └────────────────────┘
┌──────────────────────────────────────────────────────────────────────────────────────┐     └──────────────────────────────────────┘
│                                    Global Memory                                    │
└──────────────────────────────────────────────────────────────────────────────────────┘


AI Core 内部关系:
- Scalar计算单元 ← 算子指令序列
- Scalar计算单元 → 指令队列 → Vector计算单元
- Scalar计算单元 → 指令队列 → Cube计算单元
- Scalar计算单元 → 指令队列 → DMA搬运单元
- 各计算/搬运单元位于 Local Memory 上方
- Scalar计算单元 与各单元之间同时存在 指令流 / 数据流 / 同步信号
- Scalar计算单元 与 Global Memory 相连
- DMA搬运单元 与 Global Memory 相连

右侧 SIMD-Reg矢量计算内存层级:
- 指令发射单元 AUX Scalar → Reg矢量计算执行单元
- 指令发射单元 AUX Scalar → DMA单元
- Reg矢量计算执行单元 → Registers
- DMA单元 → Registers
- DMA单元 → Local Memory
```

## Regbase和Membase编程调用层级<a name="section61258392540"></a>

在Membase架构中，基础API调用框架API或直接调用编译器BuiltIn API实现功能，而高阶API则通过调用基础API来实现功能。在Regbase架构中新增Reg矢量计算API，用户在算子实现中可以直接调用该API，高阶API和基础API也可以调用该API来实现功能，Reg矢量计算API则是直接调用编译器BuiltIn API实现功能。

在Regbase架构中，中间结果可暂存在寄存器中，无需数据搬出到Local Memory的开销；在Membase架构中，所有操作均基于内存进行，这意味着每次计算都需要从Local Memory加载数据，计算完成后将结果搬回Local Memory，中间计算结果都需要暂存在Local Memory上。

在Regbase架构中，寄存器容纳的最大数据长度为VL（Vector Length），由于寄存器容量的限制，每次只能处理VL长度的数据。因此，需要对数据进行切分，每次从Local Memory搬运VL长度的数据到寄存器中进行计算，计算完成后将结果搬回Local Memory。而在Membase架构中，则能够直接处理完整长度的LocalTensor，无需进行数据切分，从而简化了数据处理流程。

<!-- img2text -->
```
NPU架构版本270x                                      Ascend 950P/Ascend 950D对应的架构版本

┌──────────────────────────────┐                  ┌──────────────────────────────┐
│           算子实现           │                  │           算子实现           │
└──────────────────────────────┘                  └──────────────────────────────┘
           │            │                                      │          │
         调用          调用                                   调用        调用
           │            │                                      │          │
           ↓            ↓                                      ↓          ↓
   ┌──────────────┐                               ┌──────────────┐   ┌─────────────────────────────── ─ ─ ─ ┐
   │   高阶API    │                               │   高阶API    │   │                                   调用 │
   └──────────────┘                               └──────────────┘   │            基础API                   │
           │            ╲                                  │          │                                   │
         调用            ╲                               调用        │  ┌──────────────────────┐          │
           │              ╲                                │          │  │ Membase矢量计算API   │          │
           ↓               ↓                               ↓          │  │ （兼容、ISASI）      │          │
   ┌────────────────────────────┐                          │          │  └──────────────────────┘          │
   │ 基础API（兼容、ISASI）     │                          │          │               │                     │
   └────────────────────────────┘                          │          │               │                     │
           │                                               │          │               │                     │
           │                                               │          │               │                     │
           │                                               │          │      ┌──────────────────────┐      │
           │                                               │          │      │ Reg矢量计算API       │      │
           │                                               │          │      └──────────────────────┘      │
           │                                               │          │                                   │
           ↓                                               ↓          └───────────────────────────────────┘
┌──────────────────────────────────┐          ┌──────────────────────────────────┐
│ 框架API（TPipe、TQue、           │          │ 框架API（TPipe、TQue、           │
│ LocalTensor、GlobalTensor等）    │          │ LocalTensor、GlobalTensor等）    │
└──────────────────────────────────┘          └──────────────────────────────────┘
           │            │                                      │          │              │
           │            │                                      │          │              │
           ↓            ↓                                      ↓          ↓              ↓
┌──────────────────────────────┐                  ┌──────────────────────────────┐
│      编译器BuiltIn API       │                  │      编译器BuiltIn API       │
└──────────────────────────────┘                  └──────────────────────────────┘
```

说明:
- 左图包含：算子实现、高阶API、基础API（兼容、ISASI）、框架API（TPipe、TQue、LocalTensor、GlobalTensor等）、编译器BuiltIn API。
- 右图中虚线框标注为“基础API”，其中包含：Membase矢量计算API（兼容、ISASI）和 Reg矢量计算API。
- 两图顶部“算子实现”分别调用下层API；底部“框架API（TPipe、TQue、LocalTensor、GlobalTensor等）”与相关计算/API层共同落到“编译器BuiltIn API”。
- 右图中“高阶API”可调用“Membase矢量计算API（兼容、ISASI）”；“Reg矢量计算API”位于同一“基础API”虚线框内。

## Reg矢量计算调用层次<a name="section1048755114224"></a>

-    核函数，使用\_\_global\_\_  \_\_aicore\_\_标识的为核函数，是Device侧的入口函数，Host侧可以通过<<<...\>\>\>语法进行调用。
-   \_\_aicore\_\_函数，使用\_\_aicore\_\_标识该函数在Device侧执行。 核函数内可以调用\_\_aicore\_\_函数。
-   simd vf函数，使用\_\_simd\_vf\_\_标记，能被核函数通过simd vf函数调用。simd vf函数内只能调用\_\_simd\_callee\_\_函数和constexpr aicore。
-   \_\_simd\_callee\_\_子函数，在simd vf函数内可以调用子函数，并且这些子函数有可能需要返回值或者通过引用传参，这类子函数通过\_\_simd\_callee\_\_标识。\_\_simd\_callee\_\_函数内只能调用\_\_simd\_callee\_\_函数和constexpr aicore函数。

具体的调用关系图如下：

<!-- img2text -->
```text
                          __global__ __aicore__
┌───────────────────────────────────────────────────────────────────────────────┐
│                         __global__ __aicore__                                │
└───────────────────────────────────────────────────────────────────────────────┘
          │                            │                              │
          │                            ↓                              │
          │                  ┌───────────────────┐                   │
          │                  │    __aicore__     │◄──┐               │
          │                  └───────────────────┘   │               │
          │                        │      │          └──┐            │
          │                        │      │             │            │
          │  VF-CALL               │      │             │            │
          ↓                        │      ↓             │            ↓
┌───────────────────┐              │  constexpr __aicore__（scalar相关）
│    __simd_vf__    │◄─────────────┘      │
└───────────────────┘                     │
          │                               │
          ↓                               │
┌───────────────────┐                     │
│  __simd_callee__  │◄──┐                 │
└───────────────────┘   │                 │
          │             │                 │
          └─────────────┴─────────────────┴───────────────────────────────→
                    constexpr __aicore__（scalar相关）
┌───────────────────────────────────────────────────────────────────────────────┐
│                    constexpr __aicore__（scalar相关）                        │◄──┐
└───────────────────────────────────────────────────────────────────────────────┘   │
                                                                                    └──
```

以下为唯一合法函数调用链：

<!-- img2text -->
```text
                           MainScalar                                              SIMD

┌───────────┐     ┌───────────┐     ┌───────────┐     ┌─────┐     ┌───────────┐     ┌──────────────┐     ┌─────┐     ┌──────────────┐
│   Host    │ ─→  │ __global__│ ─→  │ __aicore__│ ─→  │ ... │ ─→  │ __aicore__│ ─→  │ __simd_vf__  │ ─→  │ ... │ ─→  │ __simd_callee__ │
└───────────┘     └───────────┘     └───────────┘     └─────┘     └───────────┘     └──────────────┘     └─────┘     └──────────────┘
                                                                                           │
                                                                                           │
                                                                                           └────────────────────────→ __simd_callee__
```

Regbase编程模型中允许定义simd vf函数，并且通过\_\_simd\_vf\_\_来进行标记，这种设计方案有如下优点：

-   \_\_aicore\_\_和\_\_simd\_vf\_\_代码隔离清晰，编译器可以对编译器BuiltIn API的使用范围是否合法做检测。
-   对函数调用做完善的检查报错，比如在\_\_simd\_vf\_\_内调用\_\_aicore\_\_函数或者simt函数等错误用法。
-   使用\_\_simd\_vf\_\_函数编程，用户可以控制某些优化选项（如多个simd vf函数融合）只针对特定函数生效，或针对特定函数关闭某些优化。

本示例中，在\_\_aicore\_\_函数Compute中调用了VF函数AddVF进行向量加法操作。

```
template <typename T> 
__aicore__ inline void Compute()      
{  
     //申请输出队列并读取输入结果
     ...
     //调用simd vf函数
     AscendC::VF_CALL<AddVF<T>>(dstAddr, src0Addr, src1Addr, count, oneRepeatSize, repeatTimes);
     //写入结果到输出队列并释放输入队列的内存
     ...
}
```

## Reg矢量计算寄存器<a name="section1910719438134"></a>

Reg矢量计算API操作的基础数据类型介绍如下，具体API请参考[Reg矢量计算](Reg矢量计算.md)。

-   **RegTensor**

    矢量数据寄存器，Reg矢量计算基本存储单元，用于矢量计算。RegTensor的位宽是VL（Vector Length），可存储VL/sizeof\(T\)的数据（T表示数据类型）。

-   **MaskReg**

    掩码寄存器，用于矢量计算中选择参与计算的元素。MaskReg的位宽是VL/8。

-   **UnalignRegForLoad & UnalignRegForStore**

    非对齐寄存器，作为缓冲区，用来优化UB和RegTensor之间的连续非对齐地址访问的开销。在读非对齐地址前，UnalignReg应该通过LoadUnAlignPre初始化，然后再使用LoadUnAlign。在写非对齐地址时，先使用StoreUnAlign，再使用StoreUnAlignPost进行后处理。

-   **AddrReg**

    地址寄存器，用于存储地址偏移量的寄存器。AddrReg通过CreateAddrReg初始化，然后在循环之中使用AddrReg存储地址偏移量。AddrReg在每层循环中根据所设置的stride进行自增。

    本示例中的AddVF函数通过Reg矢量计算API的add接口实现两组数据的相加操作，实现高效、灵活的向量计算。通过设置MaskReg掩码寄存器，根据实际有效数据长度count生成掩码mask，控制参与运算的数据元素的数量。通过LoadAlign/StoreAlign接口，实现UB和Reg矢量计算寄存器之间的数据搬运。

    本示例为连续对齐搬入搬出场景，使用到的寄存器类型为RegTensor、MaskReg和AddrReg。

    ```
    template<typename T>
    __simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__  T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg0;
        AscendC::MicroAPI::RegTensor<T> srcReg1;
        AscendC::MicroAPI::RegTensor<T> dstReg;
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::AddrReg aReg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);        
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr, aReg);
            AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr, aReg);
            AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr, dstReg, aReg, mask);
        }
    }
    ```

    本示例为连续非对齐搬入搬出场景，使用到的寄存器类型为RegTensor、MaskReg、AddrReg以及UnalignRegForLoad和UnalignRegForStore。

    ```
    template <typename T>
    __simd_vf__ inline void LoadUnAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg;
        AscendC::MicroAPI::UnalignRegForLoad ureg0;
        AscendC::MicroAPI::UnalignRegForStore ureg1;
        AscendC::MicroAPI::AddrReg aReg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
            AscendC::MicroAPI::LoadUnAlignPre(ureg0, srcAddr, aReg);
            AscendC::MicroAPI::LoadUnAlign(srcReg, ureg0, srcAddr, aReg, 0);
            AscendC::MicroAPI::StoreUnAlign(dstAddr, srcReg, ureg1, aReg);
        }
        AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg1, aReg);
    }
    ```

## 流水线同步控制<a name="section1633824511379"></a>

在SIMD的VF函数的编写中，有时候需要将不同的值根据循环写入到同一个地址中，或者目标dst和源src是同一个地址，这就涉及到不同流水的同步指令。SIMD VF函数内不同流水线之间的同步指令使用LocalMemBar来表示。该同步指令指定src源流水线和dst目的流水线，如下图所示，目的流水线将等待源流水线上所有指令完成才进行执行。写读场景下，当写指令使用的寄存器和读指令使用的寄存器相同时，可以触发寄存器保序，指令将会按照代码顺序执行，不需要插入同步指令，而当写指令使用的寄存器和读指令使用的的寄存器不同时，如果要确保两条指令顺序执行，则需要插入同步指令，写写场景同理。

<!-- img2text -->
```text
读写依赖

┌────────────────────────┐      ┌─────────────────────────────┐
│ SIMD_VF函数内读UB流水线 │────→│ 读UB,Load(ubAddr0, vReg0)   │──────────────────────── src源流水线
└────────────────────────┘      └─────────────────────────────┘
                                              ╲
                                               ╲  LocalMemBar<src, dst>()
                                                ▼
                                   ┌─────────────────────────────┐──────────────────────── dst目的流水线
                                   │ 写UB,Store(ubAddr0, vReg1)  │
                                   └─────────────────────────────┘


读写（寄存器保序）

┌────────────────────────┐      ┌─────────────────────────────┐
│ SIMD_VF函数内读UB流水线 │────→│ 读UB Load(ubAddr0, vReg0)   │──────────────────────── src源流水线
└────────────────────────┘      └─────────────────────────────┘
                                              ╲
                                               ╲  寄存器保序
                                                ▼
                                   ┌─────────────────────────────┐──────────────────────── dst目的流水线
                                   │ 写UB,Store(ubAddr0, vReg0)  │
                                   └─────────────────────────────┘


写写依赖

┌────────────────────────┐    ┌─────────────────────────────┐
│ SIMD_VF函数内写UB流水线 │──→│ 写UB,Store(ubAddr0, vReg0)  │──────────────────────── src源流水线
└────────────────────────┘    └─────────────────────────────┘
                                            ╲
                                             ╲  LocalMemBar<src, dst>()
                                              ▼
                                 ┌─────────────────────────────┐──────────────────────── dst目的流水线
                                 │ 写UB,Store(ubAddr0, vReg1)  │
                                 └─────────────────────────────┘

┌────────────────────────┐
│ SIMD_VF函数内写UB流水线 │
└────────────────────────┘
```

函数原型：

```
template <MemType src, MemType dst> 
__simd_callee__ inline void LocalMemBar()
```

## 如何使用Reg矢量计算API<a name="section1227863453815"></a>

基于寄存器的编程模型是指每次循环将一个VL长度的数据从从LocalTensor通过数据搬运指令加载到寄存器中，进行复杂的数学计算Compute后搬出到LocalTensor中，所有的计算逻辑均在寄存器中完成，从而减少LocalTensor间的数据搬运，大大提升了整体性能，具体流程如下所示：

<!-- img2text -->
```text
                              CopyIn                                                        CopyOut
                                │                                                             │
                                │ 输入                                                         │ 输出
                                ▼                                                             ▼

┌──────┐    →    ┌──────────────┐    →    ┌────────────────┐    →    ┌──────────────────────────────────────────────┐    →    ┌────────────────┐    →    ┌──────────────┐    →    ┌──────┐
│  GM  │         │  GM搬入单元   │         │ UB             │         │                  VF-Compute                  │         │ UB             │         │  GM搬出单元   │         │  GM  │
└──────┘         └──────────────┘         │ ( Local Tensor )│         │  ┌────────────────────────────────────────┐  │         │ ( Local Tensor )│         └──────────────┘         └──────┘
                                           └────────────────┘         │  │              VF执行单元                │  │         └────────────────┘
                                                                      │  │                                        │  │
                                                                      │  │               ┌────────┐               │  │
                                                                      │  │               │ 寄存器 │               │  │
                                                                      │  │               └────────┘               │  │
                                                                      │  │                ↓  EXE                  │  │
                                                                      │  │                                        │  │
                                                                      │  │  LD 搬入                              ST 搬出 │
                                                                      │  │   ↗                                    ↘ │  │
                                                                      │  │ ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │
                                                                      │  │ │寄存器搬入单元│  │寄存器执行单元│  │寄存器搬出单元│ │  │
                                                                      │  │ └────────────┘  └────────────┘  └────────────┘ │  │
                                                                      │  └────────────────────────────────────────┘  │
                                                                      └──────────────────────────────────────────────┘
```

以AddVF函数为例，首先定义三个矢量数据寄存器srcReg0、srcReg1和dstReg以及掩码寄存器mask，每次将一个VL长度的数据使用数据搬运函数从src0、src1搬入到数据寄存器srcReg0、srcReg1中，地址偏移是src0Addr+ i \* oneRepeatSize、src1Addr + i \* oneRepeatSize，然后调用Add函数，将结果存入到dstReg中（dstReg= srcReg0 + srcReg1\)，mask表示参与Add计算的元素个数，最后调用数据搬运函数将结果从dstReg中搬出到dst。

Add的原型定义如下：

```
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
__simd_callee__ inline void Add(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
```

其中模板参数T表示操作数数据类型，MaskMergeMode表示mask未筛选的元素在dst中置零或者保留原值，UpdateMask函数用于更新参与计算的mask元素，每次循环都会消耗一个VL长度的元素。LoadAlign和StoreAlign函数用于数据的搬入搬出，LoadAlign\(srcReg0, src0Addr + i \* oneRepeatSize\)表示数据从LocalTensor搬入到srcReg0寄存器，起始地址是src0Addr + i \* oneRepeatSize，StoreAlign\(dstAddr+ i \* oneRepeatSize, dstReg,  mask\)表示将dstReg搬出到LocalTensor，目标地址是dstAddr + i \* oneRepatSize, mask表示有多少元素参与搬出。

## Reg矢量计算编程示例<a name="section13119114173116"></a>

以Add函数为例，宏函数AddVF使用\_\_simd\_vf\_\_标记，这样的函数也被称为**SIMD VF**函数。AddVF包含6个参数。dstAddr表示输出数据，src0Addr和src1Addr表示输入数据。\_\_ubuf\_\_ 类型表示用于矢量计算的Local Memory（Unified Buffer），是LocalTensor实际存储的物理位置。count表示输入数据参与计算的元素个数，repeatTimes表示循环次数，oneRepeatSize表示每次循环参与的数据量。Add函数首先计算每次能搬入到寄存器中的数据量oneRepeatSize和循环次数repeatTimes，然后使用GetPhyAddr获取输入数据和输出数据的UB地址，并通过VF\_CALL<AddVF<T\>\>调用AddVF宏函数进行计算。

```
// SIMD函数
template <typename T> 
__simd_vf__ inline void AddVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count,  uint32_t oneRepeatSize, uint16_t repeatTimes) 
{     
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);  
        AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);
        AscendC::MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg , mask);
    }
}

template <typename T> 
__aicore__ inline void Compute()      
{  
     AscendC::LocalTensor<T> dst = outQueueZ.AllocTensor<T>();     
     AscendC::LocalTensor<T> src0 = inQueueX.DeQue<T>();
     AscendC::LocalTensor<T> src1 = inQueueY.DeQue<T>();
     constexpr uint32_t oneRepeatSize = AscendC::GetVecLen()/sizeof(T);
     uint32_t count = 512;
     // 向上取整，计算repeat
     uint16_t repeatTimes = AscendC::CeilDivision(count, oneRepeatSize);
     __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();
     __ubuf__ T* src0Addr = (__ubuf__ T*)src0.GetPhyAddr();
     __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
     AscendC::VF_CALL<AddVF<T>>(dstAddr, src0Addr, src1Addr, count, oneRepeatSize, repeatTimes);
     outQueueZ.EnQue(dst);
     inQueueX.FreeTensor(src0);
     inQueueY.FreeTensor(src1);
}
```

