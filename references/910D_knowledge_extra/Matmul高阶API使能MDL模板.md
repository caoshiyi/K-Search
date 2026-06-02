# Matmul高阶API使能MDL模板<a name="ZH-CN_TOPIC_0000002523289136"></a>

## 案例介绍<a name="section5590915836"></a>

本案例呈现了在矩阵乘算子场景中，使用Matmul高阶API进行矩阵乘法计算，使能MDL模板对算子性能的提升效果。在MDL模板中，MTE2流水从Global Memory到A1/B1的数据搬运为一次性大包搬运，即一次MTE2能搬入多个Matmul计算的基本块，提升带宽利用率，使后续的MTE1流水尽可能复用A1/B1内基本块的缓存数据，减少MTE2的搬运次数。MDL模板的详细介绍请参考[MatmulConfig](MatmulConfig.md)。

-   MDL模板的适用场景

    一般适用于MTE2循环搬运次数多的大shape场景，MDL模板在A1/B1中缓存多次计算需要的数据，避免MTE2频繁搬运。

-   MDL模板的约束条件

    MDL模板的TCubeTiling结构体需要满足TCubeTiling约束条件和MDL模板补充约束条件，具体请参考[TCubeTiling结构体](TCubeTiling结构体.md)。

本案例的算子规格如下：

**表 1**  算子规格

<a name="table9129161439"></a>
<table><thead align="left"><tr id="row4121116114317"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.1"><p id="p1712111610439"><a name="p1712111610439"></a><a name="p1712111610439"></a>输入</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.2"><p id="p15129162437"><a name="p15129162437"></a><a name="p15129162437"></a>Shape</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.3"><p id="p31231684318"><a name="p31231684318"></a><a name="p31231684318"></a>Data type</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.4"><p id="p61211694311"><a name="p61211694311"></a><a name="p61211694311"></a>Format</p>
</th>
</tr>
</thead>
<tbody><tr id="row212716144319"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p61219167439"><a name="p61219167439"></a><a name="p61219167439"></a>a</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p17124161434"><a name="p17124161434"></a><a name="p17124161434"></a>128, 1024</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p312131614432"><a name="p312131614432"></a><a name="p312131614432"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p81213165432"><a name="p81213165432"></a><a name="p81213165432"></a>ND</p>
</td>
</tr>
<tr id="row49815284318"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p49845284319"><a name="p49845284319"></a><a name="p49845284319"></a>b</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p9419175534316"><a name="p9419175534316"></a><a name="p9419175534316"></a>1024, 30720</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p6419195512430"><a name="p6419195512430"></a><a name="p6419195512430"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p12420205594320"><a name="p12420205594320"></a><a name="p12420205594320"></a>ND</p>
</td>
</tr>
</tbody>
</table>

当前案例使用的AI处理器共24个核，每个核中包含1个AIC核和2个AIV核。

Tiling参数如下：

-   原始shape：M=128, N=30720, K=1024。
-   单核shape：按24个AIC核进行切分，singleCoreM=128，singleCoreN=1280，singleCoreK=1024。

    对于B矩阵，沿着N轴进行切分，切分成24份的singleCoreN，单核上处理K \* SingleCoreN大小的数据。对于A矩阵，M轴不进行切分即singleCoreM=M，单核上处理singleCoreM \* K大小的数据。总共24个核参与计算。

-   基本块shape：baseM=128，baseN=256，baseK=64。
-   L1相关Tiling参数：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8。

## 获取性能数据<a name="section443613301436"></a>

使用msProf工具获取[算子仿真流水图](获取性能数据.md#section17259539153513)和[上板Profiling](获取性能数据.md#section17953123893415)数据，因为MDL模板主要优化MTE2搬运效率，重点分析MTE2的流水情况。

## 分析主要瓶颈点<a name="section10759361232"></a>

-   优化前的Profiling数据如下，Matmul默认为Norm模板。从C列的aic\_time数据可以看出，多个核中最大算子执行耗时为83.68us。从C列的aic\_time、L列的aic\_mte2\_time和M列的aic\_mte2\_ratio几组数据来看，MTE2平均耗时75.64us，耗时占比达到92%以上，因此需要优化MTE2流水的耗时。

    <!-- img2text -->
```
优化前流水图

MTE2(多次搬运，每次1个基本块)                         MTE1 / MMAD
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐         ┌────┐ ┌────┐ ┌────┐ ┌────┐
│MTE2│→│MTE2│→│MTE2│→│MTE2│→│ ...│→│MTE2│ ─────→ │MTE1│→│MMAD│→│MMAD│→│ ...│
└────┘ └────┘ └────┘ └────┘ └────┘ └────┘         └────┘ └────┘ └────┘ └────┘
                                                      │
                                                      ↓
                                                   ┌────┐
                                                   │MMAD│
                                                   └────┘
                                                      │
                                                      ↓
                                                   ┌────┐
                                                   │... │
                                                   └────┘
                                                      │
                                                      ↓
                                                   ┌────┐
                                                   │MMAD│
                                                   └────┘
                                                   第16个MMAD

红框关注区间:
┌──────────────────────────────────────────────────────────────────────────────┐
│ 从左侧第1个MMAD指令调用开始，到右侧第16个MMAD指令调用结束                  │
│ 第一个基本块（baseM*baseN）的计算需要调用16次MMAD指令                     │
│ singleCoreK/baseK = 16                                                     │
│ 期间耗时 10.899us                                                          │
│ 其中大部分是流水同步等待耗时                                               │
└──────────────────────────────────────────────────────────────────────────────┘

Profiling数据表（可辨识部分）

┌──────────┬─────────────┬──────────────┬──────────────────┬───────────────┬────────────┬───────────────┬────────────┬──────────────┬────────────┬───────────────┬──────────────┬────────────┬──────────────┬────────────┬──────────────┬──────────────┬────────────┬──────────────┬────────────┐
│ block_id │ sub_block_id│ aic_time(us) │ aic_total_cycles │ aic_cube_time │ aic_ratio  │ aic_scalar_time│ aic_scalar │ aic_mte1_time│ aic_mte1   │ mte1_active_bw│ aic_mte2_time│ aic_mte2   │ mte2_active_bw│ aic_mte3_time│ aic_mte3   │ mte3_active_bw│ xpipe_time │ aic_xpipe_ratio│ xpipe_active_bw │
└──────────┴─────────────┴──────────────┴──────────────────┴───────────────┴────────────┴───────────────┴────────────┴──────────────┴────────────┴───────────────┴──────────────┴────────────┴──────────────┴────────────┴──────────────┴──────────────┴────────────┴──────────────┴────────────┘
```

说明:
- 图中核心信息可从上下文明确恢复：优化前 MTE2 分多次从 Global Memory 搬运基本块到 A1/B1；由于输入矩阵 Shape 较大，MTE2 循环次数多，但每次只搬运 1 个基本块，导致带宽利用率低，整体 MTE2 搬运耗时长。
- 这进一步影响后续 MTE1 和 MMAD 流水，导致流水之间同步等待时间偏长。
- 红框标注内容为：第一个基本块（baseM*baseN）的计算需要调用 16 次 MMAD 指令，singleCoreK/baseK=16；从第1个 MMAD 到第16个 MMAD 结束期间耗时 10.899us，其中大部分是流水同步等待耗时。
- 表格中能够从图像和上下文稳定识别的列名包括：
  - block_id
  - sub_block_id
  - aic_time(us)
  - aic_total_cycles
  - aic_cube_time(us)
  - aic_cube_ratio
  - aic_scalar_time(us)
  - aic_scalar_ratio
  - aic_mte1_time(us)
  - aic_mte1_ratio
  - mte1_active_bw(G)
  - aic_mte2_time(us)
  - aic_mte2_ratio
  - mte2_active_bw(G)
  - aic_mte3_time(us)
  - aic_mte3_ratio
  - mte3_active_bw(G)
  - xpipe_time(us)
  - aic_xpipe_ratio
  - xpipe_active_bw(G)
- 前文给出的关键数据：
  - 最大算子执行耗时为 83.68us
  - MTE2 平均耗时 75.64us
  - MTE2 耗时占比达到 92% 以上

-   优化前的流水图如下，MTE2分多次从Global Memory搬运基本块到A1/B1。由于输入的矩阵Shape较大，MTE2循环搬运的次数多，但每次只搬运1个基本块，导致带宽利用率低，整体的MTE2搬运耗时长。进而影响后续的MTE1和MMAD流水，导致流水之间同步等待时间偏长。如红框所示，第一个基本块（baseM\*baseN）的计算需要调用16次MMAD指令（singleCoreK/baseK=16），从左侧的第1个MMAD指令调用开始，到右侧的第16个MMAD指令调用结束，期间耗时10.899us，其中大部分是流水同步等待耗时。

    <!-- img2text -->
```
时间轴
      5 μs             10 μs             15 μs             20 μs             25 μs             30 μs
──────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬────

Record  Save  Load  trace.json                                              Flow events   Processes

SCALARDLST (pid 20)
──────────────────────────────────────────────────────────────────────────────────────────────────────

CUBE (pid 40)
├─ CUBE_1  ───────────────────────────────────────────────────────────────────────────────────────────
├─ CUBE_2        █  █  █  █  █  █  █  █  █  █                  █  █  █  █  █  █  █  █  █  █
├─ CUBE_3      [W...] █ █ WA █ █ WA █ █ WA █ █                █ █ █ █ █ █ █ █ █ █ █ █ █
├─ CUBE_4  ─────────────────────────────────[WAIT_FLAG]───────────────                [WAIT_FLAG]────

                 ┌──────────────────────────── 红框区域 ────────────────────────────┐
                 │                 <──────────── 10.899 μs ────────────>            │
                 │  从左侧第1个MMAD指令调用开始，到右侧第16个MMAD指令调用结束       │
                 └───────────────────────────────────────────────────────────────────┘

MTE1 (pid 50)
├─ MTE1_1   █ WA █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ [WAIT_FLAG] █ █ █ █ █ █ █ [WAIT_FLAG]
├─ MTE1_2     │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │             │ │ │ │ │ │ │ │ │ │ │ │
├─ MTE1_3           │    │    │    │    │    │    │             │   │ │ │       │   │ │ │
├─ MTE1_4   █  █  █  █  █                              █       █ █ █                █   █      █ █ █
├─ MTE1_5                █  █  █  █  █  █  █  █                █   █  █  █  █  █  █  █        █ █
├─ MTE1_6  ───────────────────────────────────────────────────────────────────────────────────────────
├─ MTE1_7                                      █                                         █
└─ MTE1_8  ───────────────────────────────────────────────────────────────────────────────────────────

MTE2 (pid 60)
├─ MTE2_1         █ █ █ █ █ █ █ █ █                       █ █ █ █ █ █ █ █ █                 █ █
├─ MTE2_2   [MO]   [MOV...]         █                █                                        █
├─ MTE2_3     █                        █            [MOV_OU...]     [MOV...] [M...]          █ [MOV_OU...]
├─ MTE2_4       [M...]      [MOV_OU...]   [M...] [M...] [M...] [M...] [M...] [M...]   █           [MOV...]
├─ MTE2_5         [M...]   [MOV_OU...]       [M...] [M...] [M...] [M...]     [MOV...] [MOV_OU...] [MOV_OU...]
├─ MTE2_6                      [MOV...] [MOV...] [MOV...] [MOV...]             [M...] [M...] [M...] [M...]
├─ MTE2_7          [MOV...] [MOV...]                 [MOV...] [MOV_OU...] [MOV_OU...]          [MOV_OU...] [MOV_OU...]
├─ MTE2_8          [MOV...] [MOV_OU...]   [MOV...] [MOV...] [MOV...]                 [MOV...] [MOV...] [MOV...]
├─ MTE2_9            [MOV_OU...] [MOV_OU...]                                   [MOV_OU...]    [MOV_OU...] [MOV...]
├─ MTE2_10             [MOV_OU...] [MOV_OU...]                                  [MOV_OU...]    [MOV_OU...]
├─ MTE2_11  ────────────────────────                          [MOV_OU...] [MOV...]              [MOV_OU...] [MOV...]
└─ MTE2_12  ─────────────────────────────────────────────────────[MOV_OU...] [MOV_OU...]────────────────────────

FIXP (pid 80)
├─ FIXP_1                               [WAIT_FLAG]                               [WAIT_FLAG]
└─ FIXP_2                                         [FIX_LOC_TO_DST]                           [FIX_LOC_TO_DST]
```

说明:
- 图中主要泳道/模块文字包括：SCALARDLST (pid 20)、CUBE (pid 40)、MTE1 (pid 50)、MTE2 (pid 60)、FIXP (pid 80)
- 子通道包括：CUBE_1、CUBE_2、CUBE_3、CUBE_4；MTE1_1 ~ MTE1_8；MTE2_1 ~ MTE2_12；FIXP_1、FIXP_2
- 红框区域上方标注为 `10.899 μs`
- CUBE 与 MTE1/FIXP 中可见标签：`W...`、`WA`、`WAIT_FLAG`
- MTE2 中可见标签：`MO`、`MOV...`、`MOV_OU...`、`M...`
- FIXP_2 中可见标签：`FIX_LOC_TO_DST`
- 该图是时序/流水 trace 示意，横向为时间轴；不同泳道中的小块表示各流水阶段在对应时间片上的执行情况
- 红框强调的是一次基本块（baseM*baseN）相关计算区间，对应文中“第1个MMAD指令调用开始，到第16个MMAD指令调用结束”的耗时区段，共 `10.899 μs`

## 设计优化方案<a name="section6251134218311"></a>

下图是默认的Norm模板的Matmul计算流水示意图，MTE2分多次从Global Memory搬运基本块到A1或B1，每次只搬运一个基本块。Norm模板的优势为启动开销小，可以提前启动MTE1流水；Norm模板的劣势为在大Shape场景，MTE2搬运次数多，搬运带宽利用率低，整体性能开销大。

**图 1**  默认Norm模板流水示意图<a name="fig108684561130"></a>  
<!-- img2text -->
```
┌──────┐
│ MTE2 │   ┌──────────┐ ┌──────────┐
└──────┘   │          │ │          │
           │          │ │          │
           └──────────┘ └──────────┘

┌──────┐
│ MTE1 │      ┌──────────┐ ┌──────────┐
└──────┘      │          │ │          │
              │          │ │          │
              └──────────┘ └──────────┘

┌──────┐
│ Cube │         ┌──────────┐ ┌──────────┐
└──────┘         │          │ │          │
                 │          │ │          │
                 └──────────┘ └──────────┘
```

实现Norm模板的具体步骤如下：

1.  创建Matmul对象，使用默认的Norm模板参数CFG\_NORM。

    ```
    #define ASCENDC_CUBE_ONLY
    #include "lib/matmul_intf.h"
    
    using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
    using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
    using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj; // 使用CFG_NORM定义Matmul对象
    ```

下图是MDL模板的Matmul计算流水示意图，MTE2一次性从Global Memory搬运多个基本块到A1或B1，每次搬运stepM \* stepKa个基本块到A1或搬运stepN \* stepKb个基本块到B1。MDL模板的优势为MTE2一次性搬运多个基本块，带宽利用率高，后续的MTE1流水能尽可能复用A1或B1的缓存数据，MTE2重复搬运次数少。MDL模板的劣势为MTE2头开销时间较长，MTE1流水需要等待MTE2流水完成后才启动，MTE1启动时间晚。

**图 2**  MDL模板流水示意图<a name="fig15563371352"></a>  
<!-- img2text -->
```text
MTE2  ┌────────────────────┐
      │                    │
      └────────────────────┘

MTE1                      ┌──────────┐ ┌──────────┐
                          │          │ │          │
                          └──────────┘ └──────────┘

CUBE                                  ┌──────────┐ ┌──────────┐
                                      │          │ │          │
                                      └──────────┘ └──────────┘
```

Matmul API使能MDL模板的完整样例请参考[Matmul API性能优化样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_perf)。使能MDL模板的主要步骤如下：

1.  创建Matmul对象，使用默认的MDL模板参数CFG\_MDL。

    ```
    #define ASCENDC_CUBE_ONLY
    #include "lib/matmul_intf.h"
    
    using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
    using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
    using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj; // 使用CFG_MDL定义Matmul对象
    ```

## 验证优化方案性能收益<a name="section36229589310"></a>

-   优化后的Profiling数据如下，从C列的aic\_time数据可以看出，多个核中最大算子执行耗时为53.4us，相较于优化前的83.68us有较大提升。从L列的aic\_mte2\_time数据可以看出，MTE2平均耗时下降较多，从优化前的75.64us降低至46.24us。

    ![](figures/zh-cn_image_0000002554329303.png)

-   优化后的流水图如下，MDL模板相较于默认的Norm模板，MTE2可以一次性搬运多个基本块，整体的MTE2搬运次数减少了。同时因为MTE2一次搬运多个基本块到A1/B1，后续的MTE1流水能尽量复用A1/B1的缓存数据，减少了流水同步等待，提升了算子整体性能。如红框所示，第一个基本块（baseM\*baseN）的计算需要调用16次MMAD指令（singleCoreK/baseK=16），从左侧的第1个MMAD指令调用开始，到右侧的第16个MMAD指令调用结束耗时约5.198us，较优化前的10.899us提升较大，其中流水同步等待时间大幅减少。

    <!-- img2text -->
```text
时间轴:        5 us                 10 us                 15 us                 20 us                 25 us                 30 us
              │                     │                     │                     │                     │                     │
              ├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────┤
                 ←────────────── 5.198 us ──────────────→
              ┌──────────────────────────────────────────┐
              │                                          │
              │              红框高亮区域               │
              │                                          │
              └──────────────────────────────────────────┘

SCALAR (pid 10)    ──┆┆┆┆┆┆─┆┆┆┆┆┆┆────┆┆┆┆┆┆──────┆┆┆┆┆───────
SCALARDST (pid 20) ──┆┆┆┆┆──┆┆┆┆┆┆──────┆┆┆┆───────┆┆┆┆────────

CUBE (pid 40)
├─ CUBE_1          ───W───────────────┆──────────────┆────────────────────────
├─ CUBE_2          ──────────WAI──────┆──────WA──────┆────────■──────────■────
├─ CUBE_3          ──■ ■─■ ■─■ ■─■ ■─WAIT_FLAG──■ ■─■─WAIT_FLAG──■ ■─WAIT_FLAG─
└─ CUBE_4          ── ■ ■─ ■─ ■─ ■ ■──────────■ ■ ■ ■───────■ ■ ■ ─────────────

MTE1 (pid 50)
├─ MTE1_1          ─WAI─WAI─W…────────WAI────────────WAI────────WAI────────────
├─ MTE1_2          ───┆──┆──┆─┆┆──────┆┆──────■──────┆┆─────────┆┆──────────────
├─ MTE1_3          ─────│  │  │──────│││││────│──────││─────────│──────────────
├─ MTE1_4          ─│────│──│──│─────│───│────│──────│───│──────│──────────────
├─ MTE1_5          ───■───■──■──■────────■────■───────■─────────■──────────────
├─ MTE1_6          ─■─■──■───■────LOA──■──■─LOA──■─────LOA──■────LOA──■──LOA───
└─ MTE1_7          ───────────────────────────────■────────────────■────────────

MTE2 (pid 60)
├─ MTE2_1          ──┆────┆┆────┆──────┆┆────┆──────┆┆──────┆──────┆┆───────────
├─ MTE2_2          ─MOV──MOV────────────┆───────────────────────────────────────
├─ MTE2_3          ────────MOV────MO────MOV…──MO──M…──MO────M…──MO──M…──MO──M…
├─ MTE2_4          ──MOV…──MOV_OU…──MOV_OU…────────────MOV_OU…────MOV_OU…──────
├─ MTE2_5          ─────MOV…──────MOV_OU…──MOV…────MOV_OU…──MOV…────MOV_OU…────
├─ MTE2_6          ───────────────MOV_OUT_TO…──MOV_OU…──MOV_OU…──MOV_OU…───────
└─ MTE2_7          ───────MOV_OU…────────────────────────────────────MOV_OU…────

FIXP (pid 80)
├─ FIXP_1          ─────WAIT_FLAG────────────WAIT_FLAG────────────WAIT_FLAG─────WAIT_FLAG──
└─ FIXP_2          ──────────FIX_LOC…────────────FIX_LOC…────────────FIX_LOC…──────FIX_LOC…
```

## 总结<a name="section1593812115413"></a>

大Shape输入、MTE2搬运次数多，且MTE1流水等MTE2流水的同步等待耗时较长的场景下，可以使能MDL模板。通过实现MTE2从Global Memory一次性搬入多个基本块到A1或B1，使后续的MTE1流水能尽量复用A1/B1的缓存数据，减少MTE2的搬运次数，从而提升算子性能。

