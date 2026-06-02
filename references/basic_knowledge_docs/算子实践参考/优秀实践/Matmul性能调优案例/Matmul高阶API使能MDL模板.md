# Matmul高阶API使能MDL模板

**页面ID:** atlas_ascendc_best_practices_10_10009  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_10009.html

---

#### 案例介绍

本案例呈现了在矩阵乘算子场景中，使用Matmul高阶API进行矩阵乘法计算，使能MDL模板对算子性能的提升效果。在MDL模板中，MTE2流水从Global Memory到A1/B1的数据搬运为一次性大包搬运，即一次MTE2能搬入多个Matmul计算的基本块，提升带宽利用率，使后续的MTE1流水尽可能复用A1/B1内基本块的缓存数据，减少MTE2的搬运次数。MDL模板的详细介绍请参考MatmulConfig。

- MDL模板的适用场景

一般适用于MTE2循环搬运次数多的大shape场景，MDL模板在A1/B1中缓存多次计算需要的数据，避免MTE2频繁搬运。

- MDL模板的约束条件

MDL模板的TCubeTiling结构体需要满足TCubeTiling约束条件和MDL模板补充约束条件，具体请参考TCubeTiling结构体。

本案例的算子规格如下：

**表1 **算子规格

| 输入 | Shape | Data type | Format |
| --- | --- | --- | --- |
| a | 128, 1024 | float16 | ND |
| b | 1024, 30720 | float16 | ND |

当前案例使用的AI处理器共24个核，每个核中包含1个AIC核和2个AIV核。

Tiling参数如下：

- 原始shape：M=128, N=30720, K=1024。
- 单核shape：按24个AIC核进行切分，singleCoreM=128，singleCoreN=1280，singleCoreK=1024。

对于B矩阵，沿着N轴进行切分，切分成24份的singleCoreN，单核上处理K * SingleCoreN大小的数据。对于A矩阵，M轴不进行切分即singleCoreM=M，单核上处理singleCoreM * K大小的数据。总共24个核参与计算。

- 基本块shape：baseM=128，baseN=256，baseK=64。
- L1相关Tiling参数：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8。

#### 获取性能数据

使用msProf工具获取算子仿真流水图和上板Profiling数据，因为MDL模板主要优化MTE2搬运效率，重点分析MTE2的流水情况。

#### 分析主要瓶颈点

- 优化前的Profiling数据如下，Matmul默认为Norm模板。从C列的aic_time数据可以看出，多个核中最大算子执行耗时为83.68us。从C列的aic_time、L列的aic_mte2_time和M列的aic_mte2_ratio几组数据来看，MTE2平均耗时75.64us，耗时占比达到92%以上，因此需要优化MTE2流水的耗时。

![](images/atlas_ascendc_best_practices_10_10009_img_001.png)

- 优化前的流水图如下，MTE2分多次从Global Memory搬运基本块到A1/B1。由于输入的矩阵Shape较大，MTE2循环搬运的次数多，但每次只搬运1个基本块，导致带宽利用率低，整体的MTE2搬运耗时长。进而影响后续的MTE1和MMAD流水，导致流水之间同步等待时间偏长。如红框所示，第一个基本块（baseM*baseN）的计算需要调用16次MMAD指令（singleCoreK/baseK=16），从左侧的第1个MMAD指令调用开始，到右侧的第16个MMAD指令调用结束，期间耗时10.899us，其中大部分是流水同步等待耗时。

<!-- img2text -->
```text
时间轴
      5 us                10 us                15 us                20 us                25 us                30 us
      │───────────────────│────────────────────│────────────────────│────────────────────│────────────────────│
      ◄──────────────────────────── 10.899 us ────────────────────────────►
        ┌──────────────────────────────────────────────────────────────┐
        │                    红框标注区域（优化前）                    │
        └──────────────────────────────────────────────────────────────┘


SCALARDST (pid 20)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────


CUBE (pid 40)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
CUBE_1    │  │  │  │  │  │  │  │  │  │  │  │                         ││││                                  ││││
CUBE_2    │■■  ■■  ■■  ■■  ■■  ■■  ■■  ■■  ■■  ■■  ■■          ■■  ■■  ■■  ■■  ■■  ■■  ■■  ■■  ■■       ■■ ■■ ■■
CUBE_3    │[W...] ██ ██ [WA] ██ ██ [WA] ██ ██ [WA] ██ ██     ■■ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██            ■■ ■■ ■■
CUBE_4    │                                               [WAIT_FLAG]                  [WAIT_FLAG]


MTE1 (pid 50)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
MTE1_1    [WA ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██ ██]      [WAIT_FLAG]                         [WAIT_FLAG]
MTE1_2      │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
MTE1_3                    │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
MTE1_4      ▌  ▌  ▌  ▌  ▌                                  ▌      ▌▌▌▌                             ▌  ▌        ▌▌▌
MTE1_5                 ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌                  ▌▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌  ▌           ▌▌▌
MTE1_6
MTE1_7                                               ▌                                                ▌
MTE1_8


MTE2 (pid 60)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
MTE2_1          │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │                ││                    │ │ │
MTE2_2    [MO]  [MOV...]          │      │      │      │      │      │                │
MTE2_3      ▉                                          [MOV_OU...]                            ▉   [MOV_OU...]
MTE2_4        [M...]     │      │      [M...] [M...] [M...] [M...] [M...] [M...]      [MOV...] [M...] [M...] ...
MTE2_5           [MOV_OU...]      [M...] [M...] [M...] [M...]                [MOV...] ...
MTE2_6        [M...]   [MOV_OU...]                            [MOV...] [MOV_OU...] [MOV_OU...] [M...] ...
MTE2_7                 [M...] [M...] [M...] [MOV...] [MOV_OU...] [MOV_OU...] [MOV_OU...]     [MOV_OU...] ...
MTE2_8        [MOV...] [MOV_OU...] [MOV...] [MOV...] [MOV...]           [MOV...] [MOV...] [MOV...]     [MOV_OU...]
MTE2_9        [MOV...] [MOV_OU...]                            [MOV...] [MOV...] [MOV...]                      [MOV_O...]
MTE2_10         [MOV_OU...] [MOV_OU...]                      [MOV_OU...] [MOV_OU...]                    [MOV_OU...] [MOV...]
MTE2_11          [MOV_OU...] [MOV_OU...]                      [MOV_OU...] [MOV_OU...]                    [MOV_OU...] [MOV...]
MTE2_12                                                [MOV_OU...]   [MOV_OU...]


FIXP (pid 80)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
FIXP_1                                         [WAIT_FLAG]                               [WAIT_FLAG]
FIXP_2                                                     [FIX_LOC_TO_DST]                           [FIX_LOC_TO_DST]
```

说明:
- 图中是一个按时间展开的流水/trace 视图，横轴单位为 us。
- 左上方红框覆盖一段耗时 `10.899 us` 的区间。
- 主要分组与泳道标签包括：
  - `SCALARDST (pid 20)`
  - `CUBE (pid 40)`：`CUBE_1` ~ `CUBE_4`
  - `MTE1 (pid 50)`：`MTE1_1` ~ `MTE1_8`
  - `MTE2 (pid 60)`：`MTE2_1` ~ `MTE2_12`
  - `FIXP (pid 80)`：`FIXP_1` ~ `FIXP_2`
- 图中可识别的事件文字标签包括：
  - `W...`
  - `WA`
  - `WAIT_FLAG`
  - `MO`
  - `MOV...`
  - `MOV_OU...`
  - `M...`
  - `FIX_LOC_TO_DST`
- 由于原图为性能 trace 截图，包含大量密集且重复的小块事件，部分小块内文字仅能识别为截断形式（如 `MOV...`、`MOV_OU...`、`M...`、`W...`），无法更精确还原时以原图可见截断文本保留。

#### 设计优化方案

下图是默认的Norm模板的Matmul计算流水示意图，MTE2分多次从Global Memory搬运基本块到A1或B1，每次只搬运一个基本块。Norm模板的优势为启动开销小，可以提前启动MTE1流水；Norm模板的劣势为在大Shape场景，MTE2搬运次数多，搬运带宽利用率低，整体性能开销大。

**图1 **默认Norm模板流水示意图
<!-- img2text -->
```
Global Memory
┌──────────┐  ┌──────────┐
│ 基本块   │  │ 基本块   │
└──────────┘  └──────────┘
      │             │
      ▼             ▼
      A1            B1

                 ┌──────────┐  ┌──────────┐
                 │ 基本块   │  │ 基本块   │
                 └──────────┘  └──────────┘
                       │             │
                       ▼             ▼
                      A2            B2

                              ┌──────────┐  ┌──────────┐
                              │ 基本块   │  │ 基本块   │
                              └──────────┘  └──────────┘
                                    │             │
                                    ▼             ▼
                                   CO1           CO2
```

实现Norm模板的具体步骤如下：

1. 创建Matmul对象，使用默认的Norm模板参数CFG_NORM。

```
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj; // 使用CFG_NORM定义Matmul对象
```

下图是MDL模板的Matmul计算流水示意图，MTE2一次性从Global Memory搬运多个基本块到A1或B1，每次搬运stepM * stepKa个基本块到A1或搬运stepN * stepKb个基本块到B1。MDL模板的优势为MTE2一次性搬运多个基本块，带宽利用率高，后续的MTE1流水能尽可能复用A1或B1的缓存数据，MTE2重复搬运次数少。MDL模板的劣势为MTE2头开销时间较长，MTE1流水需要等待MTE2流水完成后才启动，MTE1启动时间晚。

**图2 **MDL模板流水示意图
<!-- img2text -->
```
┌──────────────────────────────┐
│                              │
└──────────────────────────────┘

                ┌──────────┐ ┌──────────┐
                │          │ │          │
                └──────────┘ └──────────┘
                      ┌──────────┐ ┌──────────┐
                      │          │ │          │
                      └──────────┘ └──────────┘
```

Matmul API使能MDL模板的完整样例请参考[Matmul API性能优化样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_perf)。使能MDL模板的主要步骤如下：

1. 创建Matmul对象，使用默认的MDL模板参数CFG_MDL。

```
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj; // 使用CFG_MDL定义Matmul对象
```

#### 验证优化方案性能收益

- 优化后的Profiling数据如下，从C列的aic_time数据可以看出，多个核中最大算子执行耗时为53.4us，相较于优化前的83.68us有较大提升。从L列的aic_mte2_time数据可以看出，MTE2平均耗时下降较多，从优化前的75.64us降低至46.24us。

![](images/atlas_ascendc_best_practices_10_10009_img_005.png)

- 优化后的流水图如下，MDL模板相较于默认的Norm模板，MTE2可以一次性搬运多个基本块，整体的MTE2搬运次数减少了。同时因为MTE2一次搬运多个基本块到A1/B1，后续的MTE1流水能尽量复用A1/B1的缓存数据，减少了流水同步等待，提升了算子整体性能。如红框所示，第一个基本块（baseM*baseN）的计算需要调用16次MMAD指令（singleCoreK/baseK=16），从左侧的第1个MMAD指令调用开始，到右侧的第16个MMAD指令调用结束耗时约5.198us，较优化前的10.899us提升较大，其中流水同步等待时间大幅减少。

<!-- img2text -->
```text
时间轴
5 us                         10 us                         15 us                         20 us                         25 us                         30 us
│----------------------------│-----------------------------│-----------------------------│-----------------------------│-----------------------------│

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Record │ Save │ Load │ trace.json                                                                                     Flow events │ Processes │ M │ View Options │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

SCALAR (pid 10)      ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SCALARDST (pid 20)   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

CUBE (pid 40)
├─ CUBE_1            ···························································································································
├─ CUBE_2            │ │ │  WAI   │ │ │ │ │ │ │ │ │ │ │ │      WA         │ │ │ │ │ │ │ │ │ │ │ │ │              │ │         │ │ │ │ │ │ │ │ │
├─ CUBE_3            ██ ██   ██ ██ ██ ██ ██ ██ ██   WAIT_FLAG      ██ ██ ██ ██ ██ ██ ██   WAIT_FLAG    ██ ██ ██ ██        WAIT_FLAG    ██ ██ ██ ██
└─ CUBE_4             ██ ██   ██ ██ ██ ██ ██ ██ ██      ██ ██ ██ ██ ██ ██ ██ ██      ██ ██ ██ ██      ██ ██ ██ ██      ██ ██ ██ ██

MTE1 (pid 50)
├─ MTE1_1            WAI  WAI_   W...   ...   ...   WAI...   ...   WAI...   ...   WAI...   ...   WAI...   ...   WAI...   ...   WAI...
├─ MTE1_2              │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
├─ MTE1_3             ▌ ▌   ▌▌▌▌  ▌   ▌▌▌ ▌   ▌▌    ▌▌▌▌▌     ▌   ▌▌▌▌       ▌▌▌      ▌▌▌       ▌▌▌       ▌▌
├─ MTE1_4            │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
├─ MTE1_5            ▌    ▌▌    ▌    ▌▌    ▌▌      ▌▌    ▌▌      ▌▌      ▌▌      ▌▌      ▌▌      ▌▌      ▌▌
├─ MTE1_6            LOA    LOA        LOA        LOA        LOA        LOA        LOA        LOA        LOA        LOA
└─ MTE1_7                    ▌                                            ▌                                              ▌

MTE2 (pid 60)
├─ MTE2_1            │        │      │        │      │        │      │        │      │        │      │        │      │        │
├─ MTE2_2            MOV      MOV
├─ MTE2_3                           MOV          MO  MOV...   MO  M...   MO   M...        MO   M...   MO   M...   MO   M...
├─ MTE2_4                 MOV...    MOV_OU...         MOV...               MOV_OU...          MOV_OU...          MOV_OU...
├─ MTE2_5                      MOV...                 MOV_OU...                 MOV_OU...            MOV_OU...          MOV_OU...
├─ MTE2_6                            MOV_OUT_TO...        MOV_OU...      MOV_OU...        MOV_OU...        MOV_OU...        MOV_OU...
└─ MTE2_7                       MOV...

FIXP (pid 80)
├─ FIXP_1            WAIT_FLAG                    WAIT_FLAG                    WAIT_FLAG                    WAIT_FLAG
└─ FIXP_2                               FIX_LOC...                  FIX_LOC...                  FIX_LOC...                  FIX_LOC...

红框标注区域
                     ┌─────────────────────────────── 5.198 us ───────────────────────────────┐
                     │                                                                          │
                     │  从左侧的第1个 MMAD 指令调用开始，到右侧的第16个 MMAD 指令调用结束      │
                     │                                                                          │
                     └──────────────────────────────────────────────────────────────────────────┘
                     ↑                                                                          ↑
                    约5 us                                                                     约10 us
```

#### 总结

大Shape输入、MTE2搬运次数多，且MTE1流水等MTE2流水的同步等待耗时较长的场景下，可以使能MDL模板。通过实现MTE2从Global Memory一次性搬入多个基本块到A1或B1，使后续的MTE1流水能尽量复用A1/B1的缓存数据，减少MTE2的搬运次数，从而提升算子性能。
