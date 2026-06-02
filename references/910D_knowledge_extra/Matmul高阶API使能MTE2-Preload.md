# Matmul高阶API使能MTE2 Preload<a name="ZH-CN_TOPIC_0000002523289096"></a>

## 案例介绍<a name="section17413194624510"></a>

本案例呈现了在矩阵乘算子场景中，使用Matmul高阶API进行矩阵乘法计算，使能MTE2 Preload对算子性能的提升效果**。**通过MatmulConfig中的doMTE2Preload参数开启矩阵M或N方向的预加载功能，预加载即在MTE2间隙提前加载A矩阵/B矩阵数据，开启预加载功能后，可以减少MTE2间隙，提升算子性能。doMTE2Preload参数的详细介绍请参考[MatmulConfig](MatmulConfig.md)。

-   使能MTE2 Preload的适用场景

    MTE2流水间隙较大，且M或N数值较大时。

-   使能MTE2 Preload的约束条件
    -   仅在使用MDL模板和SpecialMDL模板时，MTE2 Preload有效。
    -   开启M或N方向预加载功能时，需保证K方向数据全载，且M或N方向开启DoubleBuffer。
    -   K方向数据全载的条件是singleK <= baseK \* stepK。
    -   M方向开启DoubleBuffer的条件是depthA1 = stepM \* stepK \* 2。
    -   N方向开启DoubleBuffer的条件是depthB1 = stepN \* stepK \* 2。

本案例的算子规格如下：

**表 1**  算子规格

<a name="table101751125175213"></a>
<table><thead align="left"><tr id="row8175525185219"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.1"><p id="p1417582516529"><a name="p1417582516529"></a><a name="p1417582516529"></a>输入</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.2"><p id="p417532575212"><a name="p417532575212"></a><a name="p417532575212"></a>Shape</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.3"><p id="p1017582585214"><a name="p1017582585214"></a><a name="p1017582585214"></a>Data type</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.4"><p id="p317502512522"><a name="p317502512522"></a><a name="p317502512522"></a>Format</p>
</th>
</tr>
</thead>
<tbody><tr id="row217562525215"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p517515253529"><a name="p517515253529"></a><a name="p517515253529"></a>a</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p141751425165213"><a name="p141751425165213"></a><a name="p141751425165213"></a>128, 512</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p1517522515218"><a name="p1517522515218"></a><a name="p1517522515218"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p16176725105210"><a name="p16176725105210"></a><a name="p16176725105210"></a>ND</p>
</td>
</tr>
<tr id="row10176102512525"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p161761325185216"><a name="p161761325185216"></a><a name="p161761325185216"></a>b</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p4176112555211"><a name="p4176112555211"></a><a name="p4176112555211"></a>512, 24576</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p1176132515212"><a name="p1176132515212"></a><a name="p1176132515212"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p817692575212"><a name="p817692575212"></a><a name="p817692575212"></a>ND</p>
</td>
</tr>
</tbody>
</table>

当前案例使用的AI处理器共24个核，算子中使能高阶API Matmul的纯Cube模式。使用MDL模板，Tiling参数如下：

-   原始shape：M=128, N= 24576, K=512。
-   单核shape：singleCoreM=128，singleCoreN=1024，singleCoreK=512。
-   基本块shape：baseM=128，baseN=128，baseK=64。
-   L1缓存相关Tiling参数：stepM=1，stepN=1，stepKa=8，stepKb=8，depthA1=8，depthB1=16。

## 获取性能数据<a name="section851404010469"></a>

使用msProf工具获取[算子仿真流水图](获取性能数据.md#section17259539153513)和[上板Profiling](获取性能数据.md#section17953123893415)数据，重点分析Cube，Fixpipe的流水情况。

## 分析主要瓶颈点<a name="section221431704714"></a>

-   优化前的流水图如下，M和K方向全载，因此A矩阵只搬运一次。由于N较大，B矩阵会搬运多次，可以看到单次MTE2间存在间隙。<!-- img2text -->
```text
SCALAR (pid 10)       ──┬─┬───┬─┬──────┬───────┬────┬──────┬─────┬────────┬─────┬─────┬───────┬───
SCALARDST (pid 20)    ──┬────┬────┬───────────┬──────┬───────┬─────┬───────┬───────┬────┬──────┬──

CUBE (pid 40)
├─ CUBE_1             │            [ WAIT_FLAG ]   [ WAI... ]   [ WAI... ]   [ WAI... ]   [ WAI... ]
├─ CUBE_2             │                ││││││        ││││││       │││││││      │││││││      │││││││
├─ CUBE_3             │      █ █ █ █   █ █ █ █       █ █ █ █   [WAIT_FLAG] █   █ █ █ █ [WAIT_FLAG] █
│                     │                ███████       ███████       ███████      ███████      ███████
└─ CUBE_4             │      █ █ █ █   █ █ █ █       █ █ █ █       █ █ █ █      █ █ █ █      █ █ █ █

MTE1 (pid 50)
├─ MTE1_1             │    [ WAL... ][ WAIT_FLAG ] [ WAL... ][ WAIT_FLAG ] [ WAL... ][ WAIT_FLAG ]
├─ MTE1_2             │        ││││││      ││││││      ││││││      ││││││      ││││││      ││││││
├─ MTE1_3             │      ▓▓▓▓▓▓    ▓▓▓▓▓▓    ▓▓▓▓▓▓    ▓▓▓▓▓▓    ▓▓▓▓▓▓    ▓▓▓▓▓▓    ▓▓▓▓▓▓
├─ MTE1_4             │         █         █         █         █         █         █         █
├─ MTE1_5             │       █         █         █         █         █         █         █
└─ MTE1_6             │       █         █         █         █         █         █         █

MTE2 (pid 60)
├─ MTE2_1             │   │          │         │         │         │         │         │       ││
├─ MTE2_2             │  [ M... ]   [ MOV... ] [ MOV_OU... ] [ MOV... ] [ MOV_OU... ] [ MOV... ] [ MOV_OU... ] [ MOV... ]
├─ MTE2_3             │        │
└─ MTE2_4             │   [ MOV_OUT_TO... ]

FIXP (pid 80)
├─ FIXP_1             │            █    [ W... ]   [ W... ]   [ W... ]   [ W... ]   [ W... ]   [ W... ]
└─ FIXP_2             │             [FIX_LOC...]   [FIX_LOC...]   [FIX_LOC...]   [FIX_LOC...]

FLOWCTRL (pid 90)
├─ FLOWCTRL_1         │ ▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏ ▏▏▏▏▏▏ ▏▏▏▏▏▏▏ ▏▏▏▏▏▏▏ ▏▏▏▏▏▏ ▏▏▏▏▏▏ ▏▏▏▏▏▏ ▏▏▏▏▏
└─ FLOWCTRL_2         │                                                                                 ▏

ALL (pid 100)
```

说明:
- 图为 profiling 流水图时间轴示意，左侧为模块/子通道名称，右侧为各阶段在时间轴上的执行片段。
- 可识别文字标注包括：`SCALAR (pid 10)`、`SCALARDST (pid 20)`、`CUBE (pid 40)`、`CUBE_1`~`CUBE_4`、`MTE1 (pid 50)`、`MTE1_1`~`MTE1_6`、`MTE2 (pid 60)`、`MTE2_1`~`MTE2_4`、`FIXP (pid 80)`、`FIXP_1`、`FIXP_2`、`FLOWCTRL (pid 90)`、`FLOWCTRL_1`、`FLOWCTRL_2`、`ALL (pid 100)`。
- 图中可清晰识别的事件标签包括：`WAIT_FLAG`、`WAI...`、`WAL...`、`M...`、`MOV...`、`MOV_OU...`、`MOV_OUT_TO...`、`W...`、`FIX_LOC...`。
- 由于原图为高密度时间序列流水图，具体每个细条的精确长度与相互对齐关系无法在 ASCII 中无损复现；上图保留了主要分组结构、层级关系和可识别标签。
-   优化前的Profiling数据如下，aic\_time平均耗时30.88us。<!-- img2text -->
```
优化前的流水图

                     A矩阵只搬运一次
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                    Cube                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ ████████████████████████████████████████████████████████████████████████████ │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│    MTE2      │    │    MTE2      │        │    MTE2      │        │    MTE2      │
├──────────────┤    ├──────────────┤        ├──────────────┤        ├──────────────┤
│ ████████████ │    │ ████████████ │        │ ████████████ │        │ ████████████ │
└──────────────┘    └──────────────┘        └──────────────┘        └──────────────┘
                    ↑ 间隙                    ↑ 间隙                    ↑ 间隙

B矩阵会搬运多次，单次MTE2间存在间隙
```

## 设计优化方案<a name="section33901368431"></a>

使能MTE2 Preload功能：在创建Matmul对象时，开启doMTE2Preload开关。使能MTE2 Preload的完整样例请参考[M/N方向预加载Matmul算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_preload)。具体步骤如下：

1.  配置MDL模板参数，将其中的doMTE2Preload参数设置为2，使能N方向Preload功能。

    ```
     // preloadMode = 2
    static constexpr MatmulConfig MM_CFG = GetMDLConfig(false, false, preloadMode); 
    ```

2.  基于自定义MatmulConfig模板参数，创建Matmul对象。

    ```
    AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>, MM_CFG> matmulObj;
    ```

## 验证优化方案性能收益<a name="section19022397498"></a>

-   优化后的流水图如下，Tiling参数不变，可以看到，下一次计算使用的B矩阵数据提前加载，MTE2间的间隙缩短。<!-- img2text -->
```text
SCALAR (pid 10)     ─────────────────────────────────────────────────────────────────────────────────────────────── X
SCALARLDST (pid 20) ─────────────────────────────────────────────────────────────────────────────────────────────── X

CUBE (pid 40)       ─────────────────────────────────────────────────────────────────────────────────────────────── X
├─ CUBE_1
│                      ┌───────────┐         ┌──┐                       ┌──┐
│                      │ WAIT_FLAG │         │WA│                       │WA│
│                      └───────────┘         └──┘                       └──┘
│                               ││││││  │││││││    │││││││   ││││││    │││││││    │││││││
├─ CUBE_2
│                                   ││││││  │││││││    │││││││   ││││││    │││││││    │││││││
├─ CUBE_3
│                          █ █ █ █    █ █ █ █    ┌───────────┐   ┌──┐          █ █ █ █    ┌──┐      █ █ █ █
│                                              │ WAIT_FLAG │   │WA│          █ █ █ █    │WA│      █ █ █ █
│                                              └───────────┘   └──┘
├─ CUBE_4
│                          █ █ █ █    █ █ █ █      █ █ █ █   █ █      ┌───────────┐  █ █ █ █  █ █  ┌───────────┐  █ █ █ █  █ █ █
│                                                                            │ WAIT_FLAG │                    │ WAIT_FLAG │
│                                                                            └───────────┘                    └───────────┘

MTE1 (pid 50)       ─────────────────────────────────────────────────────────────────────────────────────────────── X
├─ MTE1_1
│                 ┌──┐┌───────────┐          ┌──┐┌───────────┐     ┌──┐┌───────────┐     ┌──┐┌───────────┐
│                 │WAL││ WAIT_FLAG │          │WAI││ WAIT_FLAG │     │WAI││ WAIT_FLAG │     │WAI││ WAIT_FLAG │
│                 └──┘└───────────┘          └──┘└───────────┘     └──┘└───────────┘     └──┘└───────────┘
│                              ▏▏▏▏▏▏        ▏▏▏▏▏▏        ▏▏▏▏▏▏        ▏▏▏▏▏▏
├─ MTE1_2
│                                ││││││   ││││││    ││││││   ││││││    ││││││   ││││││
├─ MTE1_3
│                           ██████   ██████   ██████   █████    ██████   █████    ██████   ██████
├─ MTE1_4
│                           █        █  █      █      █         █        █  █      █
├─ MTE1_5
│                           █        █  █      █      █         █        █  █      █
├─ MTE1_6
│                           █           █             █                   █         █

MTE2 (pid 60)       ─────────────────────────────────────────────────────────────────────────────────────────────── X
├─ MTE2_1
│                      │         │     │    │         │    │      │         │     │        ││
├─ MTE2_2
│                 ┌──────┐            ┌────────────┐┌────────┐    ┌────────────┐┌────────┐
│                 │MOV...│            │MOV_OUT_TO...││MOV_OU...│    │MOV_OUT_TO...││MOV_OU...│
│                 └──────┘            └────────────┘└────────┘    └────────────┘└────────┘
├─ MTE2_3
│                                              │                 │                 │
├─ MTE2_4
│                   ┌──────────────┐
│                   │MOV_OUT_TO... │
│                   └──────────────┘
├─ MTE2_5
│                            │
├─ MTE2_6
│                    ┌────────────────────┐
│                    │MOV_OUT_TO_L1_M...  │
│                    └────────────────────┘

FIXP (pid 80)       ─────────────────────────────────────────────────────────────────────────────────────────────── X
├─ FIXP_1
│                                        ┌───┐   ┌───┐     ┌───┐    ┌────┐      ┌────┐      ┌───┐      ┌───┐
│                                        │WAI│   │WAI│     │W...│    │WAI...│      │W...│      │W...│      │W...│
│                                        └───┘   └───┘     └───┘    └────┘      └────┘      └───┘      └───┘
├─ FIXP_2
│                                           ┌───┐   ┌──────────────┐  ┌───┐  ┌──────────────┐  ┌───┐  ┌────────┐  ┌───┐  ┌───┐
│                                           │FIX│   │FIX_LOC_TO_DST│  │FIX│  │FIX_LOC_TO_DST│  │FIX│  │FIX_LOC...│  │FIX│  │FIX│
│                                           └───┘   └──────────────┘  └───┘  └──────────────┘  └───┘  └────────┘  └───┘  └───┘

FLOWCTRL (pid 90)   ─────────────────────────────────────────────────────────────────────────────────────────────── X
└─ FLOWCTRL_1
    ▏▏ ▏▏▏ ▏  ▏ ▏▏▏▏▏ ███  ▏▏▏ ████████   █▏▏▏▏  ▏▏▏   ████████   ██████  ▏▏▏▏▏▏▏▏ ▏
```
-   优化后的Profiling数据如下，aic\_time平均耗时28.50us，较优化前的30.88us有所下降。![](figures/zh-cn_image_0000002523289234.png)

## 总结<a name="section8281219125011"></a>

当MTE2流水间隙较大，且M或N数值较大时，可以考虑使能MTE2 Preload功能，提前加载A矩阵或B矩阵数据。

