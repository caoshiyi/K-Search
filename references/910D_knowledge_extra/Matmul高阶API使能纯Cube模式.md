# Matmul高阶API使能纯Cube模式<a name="ZH-CN_TOPIC_0000002554328993"></a>

## 案例介绍<a name="section1698917333"></a>

本案例呈现了在矩阵乘算子场景中，使能Matmul高阶API的纯Cube模式对算子性能的提升效果。如下图所示，Matmul API默认使用MIX模式，即用户从AIV侧发起消息，通过消息通信框架中转消息后，在AIC侧执行Matmul计算。这套消息处理机制会带来额外的Scalar性能开销。相较于MIX模式，纯Cube模式可以直接跳过消息通信框架，完成Matmul计算，提升算子性能。

**图 1**  默认MIX模式的Matmul流程示意图<a name="fig0672118378"></a>  
<!-- img2text -->
```
                                  AIV侧                                              AIC侧

          用户
            │
            ▼
┌───────────────────────┐              ┌────────────────┐              ┌────────────────┬────────────────────┐
│                       │              │                │              │                │                    │
│    ┌──────────────┐   │     ─────▶   │  消息通信框架  │    ─────▶    │ Matmul Server  │ Matmul Implement   │
│    │ Matmul Client│   │              │                │              │                │                    │
│    └──────────────┘   │              └────────────────┘              └────────────────┴────────────────────┘
│                       │
│      AIV发消息        │                   框架
│                       │                  中转消息                         AIC执行
└───────────────────────┘                                                   Matmul计算
```

-   使能纯Cube模式的适用场景

    非融合算子，只有矩阵计算的场景。即相较于MIX模式（包含矩阵计算和矢量计算），没有矢量计算的场景。本案例的算子规格如下：

**表 1**  算子用例规格

<a name="table15465191317123"></a>
<table><thead align="left"><tr id="row184651013131217"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.1"><p id="p24653132122"><a name="p24653132122"></a><a name="p24653132122"></a>输入</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.2"><p id="p13465111311213"><a name="p13465111311213"></a><a name="p13465111311213"></a>Shape</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.3"><p id="p14465171371212"><a name="p14465171371212"></a><a name="p14465171371212"></a>Data type</p>
</th>
<th class="cellrowborder" valign="top" width="25%" id="mcps1.2.5.1.4"><p id="p74651713141213"><a name="p74651713141213"></a><a name="p74651713141213"></a>Format</p>
</th>
</tr>
</thead>
<tbody><tr id="row446561351212"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p517515253529"><a name="p517515253529"></a><a name="p517515253529"></a>a</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p141751425165213"><a name="p141751425165213"></a><a name="p141751425165213"></a>128, 64</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p1517522515218"><a name="p1517522515218"></a><a name="p1517522515218"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p16176725105210"><a name="p16176725105210"></a><a name="p16176725105210"></a>ND</p>
</td>
</tr>
<tr id="row44651313101220"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.1 "><p id="p161761325185216"><a name="p161761325185216"></a><a name="p161761325185216"></a>b</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.2 "><p id="p4176112555211"><a name="p4176112555211"></a><a name="p4176112555211"></a>64, 30720</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.3 "><p id="p1176132515212"><a name="p1176132515212"></a><a name="p1176132515212"></a>float16</p>
</td>
<td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.5.1.4 "><p id="p817692575212"><a name="p817692575212"></a><a name="p817692575212"></a>ND</p>
</td>
</tr>
</tbody>
</table>

当前案例使用的AI处理器共24个核，每个核中包含1个AIC核和2个AIV核。

Tiling参数如下：

-   原始shape：M=128, N=30720, K=64。
-   单核shape：
    -   MIX场景：按48个AIV核进行切分，singleCoreM=128，singleCoreN=640，singleCoreK=64。
    -   纯Cube场景：按24个AIC核进行切分，singleCoreM=128，singleCoreN=1280，singleCoreK=64。

-   基本块shape：baseM=128，baseN=256，baseK=64。
-   L1相关Tiling参数：stepM=1，stepN=1，stepKa=4，stepKb=4，depthA1=8，depthB1=8。

## 获取性能数据<a name="section1910315281533"></a>

使用msProf工具获取[算子仿真流水图](获取性能数据.md#section17259539153513)和[上板Profiling](获取性能数据.md#section17953123893415)数据。因为纯Cube模式主要优化Scalar流水性能，可以重点分析Scalar的流水情况。

## 分析主要瓶颈点<a name="section145803332032"></a>

-   优化前的Profiling数据如下，从C列的aic\_time数据可以看出，多个核中最大算子执行耗时为17.85us。从G列的aic\_scalar\_time数据可以看出，Scalar平均耗时为15.02us，性能瓶颈在Scalar流水。

    ![](figures/zh-cn_image_0000002554409073.png)

-   优化前的流水图如下，由于默认为MIX模式，每次Matmul计算均涉及消息通信框架对消息进行处理，Scalar流水重，性能开销较大，如下图红框所示。

    <!-- img2text -->
```text
时间轴 →
0 us                2 us                4 us                6 us                8 us               10 us               12 us

Record  Save  Load  trace.json
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

SCALAR (pid 10)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  灰色时间片段零散分布                                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

SCALARLDST (pid 20)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ SCALARLDST_1   │        │  LD_──────────────────────────────────────────────────────────────────────────────────────     │
│ SCALARLDST_2   │        │ ──────────────────────────────────────────────────────────────────────────────────────────     │
│ SCALARLDST_3   │   ││   │ ──────────────────────────────────────────────────────────────────────────────────────────     │
│ SCALARLDST_4   │        │ ST_───────────────────────────────────────────────────────────────────────────────────────     │
│ SCALARLDST_5   │   │    │   │││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││     │
│ SCALARLDST_6   │        │   ││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││     │
│ SCALARLDST_7   │        │   ││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││     │
│ SCALARLDST_8   │        │      ││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││││     │
│ SCALARLDST_9   │        │      ││ ││ ││ ││     │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │     │
│ SCALARLDST_10  │        │       ││ ││ ││ │                                                                                 │
│ SCALARLDST_11  │        │       ││                                                                                         │
│ SCALARLDST_12  │        │       ││             │                                                                           │
│ SCALARLDST_13  │        │       ││                                                                                         │
│ SCALARLDST_14  │        │        │                                                                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                              ┌──────────────────────────────────────────────────────────────────────────────────────────┐
                              │                        红框标注区域                                                     │
                              └──────────────────────────────────────────────────────────────────────────────────────────┘

CUBE (pid 40)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ CUBE_1        │                    W...        │     │     │      │      │      │                                      │
│ CUBE_2        │                                                                                                         │
│ CUBE_3        │                      MM       MM               MM         MM                                            │
│ CUBE_4        │                        WAIT_FLAG  WAIT_FLAG  WAIT_FLAG   WAIT_FLAG  WAIT_FLAG   W...                    │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

MTE1 (pid 50)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  灰色时间片段                                                                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

MTE2 (pid 60)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MTE2_1        │                 │       │  │  │      │                     │                                            │
│ MTE2_2        │                    MO   │MO│ MO           MOV_OU...                                                     │
│ MTE2_3        │                                                                 │        ││                             │
│ MTE2_4        │                     MOV                    MOV_OU...                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

MTE3 (pid 70)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  灰色时间片段                                                                                                            │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

FIXP (pid 80)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FIXP_1        │                    WAI...        │     │      │      │      │      │                                   │
│ FIXP_2        │                         FIX_LOC...  FIX_LOC...  FIX_LOC_TO_DST  W...  FIX_LOC_TO_DST  FIX_LOC...  FI... │
│ FIXP_3        │                               WAI                              WAI                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 设计优化方案<a name="section81265422311"></a>

默认MIX模式下，用户在AIV侧发起消息，通过消息通信框架中转消息后，在AIC侧执行Matmul计算。基于这样的流程，用户使用Matmul高阶API编写算子代码时，可以使用[REGIST\_MATMUL\_OBJ](REGIST_MATMUL_OBJ.md)宏，无需区分AIV和AIC，但也因这套消息处理机制导致产生了额外的性能开销，如[图1 默认MIX模式的Matmul流程示意图](#fig0672118378)所示。

实现默认MIX模式的具体步骤如下：

1.  Kernel侧，定义Matmul对象。

    ```
    #include "lib/matmul_intf.h"
    
    using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
    using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
    using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj;
    ```

2.  Host侧，Matmul多核Tiling对象调用SetDim接口设置参与运算的核数。

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
    int32_t numBlocks = ascendcPlatform->GetCoreNumAiv(); // MIX模式使用GetCoreNumAiv获取AI处理器可用的核数。
    cubeTiling.SetDim(numBlocks);
    ```

3.  调用核函数，参考[核函数定义和调用](核函数.md#zh-cn_topic_0000001447989210_section1915102519220)，设置核函数的numBlocks参数配置。

    ```
    matmul_custom_do(ascendcPlatform->GetCoreNumAic(), stream, x1, x2, bias, y, workspaceDevice, tilingDevice); // MIX模式下，启动时，按照AIV和AIC组合启动，numBlocks用于设置启动多少个AI Core。
    ```

在没有矢量计算的算子场景下，可以跳过消息通信框架的机制，使能纯Cube模式完成Matmul计算，减少消息通信的性能开销，提升算子性能。

**图 2**  纯Cube模式的Matmul流程示意图<a name="fig20558182319127"></a>  
<!-- img2text -->
```text
           AIC侧
┌───────────────────────┐
│  ┌─────────────────┐  │
│  │     Matmul      │  │
│  │    Implement    │  │
│  └─────────────────┘  │
└───────────────────────┘
用户 ───────────────────→

        AIC执行
      Matmul计算
```

Matmul API使能纯Cube模式的完整样例请参考[Matmul API性能优化样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_perf)。使能纯Cube模式的主要步骤如下：

1.  Kernel侧，在定义Matmul对象的代码中，包含matmul\_intf.h头文件前设置ASCENDC\_CUBE\_ONLY宏。

    ```
    #define ASCENDC_CUBE_ONLY // 在#include "lib/matmul_intf.h"前，设置ASCENDC_CUBE_ONLY宏
    #include "lib/matmul_intf.h"
    
    using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
    using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
    using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
    using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
    AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj;
    ```

2.  Host侧，Matmul多核Tiling对象调用SetDim接口设置参与运算的核数。

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    matmul_tiling::MultiCoreMatmulTiling cubeTiling(*ascendcPlatform);
    int32_t numBlocks = ascendcPlatform->GetCoreNumAic(); // 纯Cube模式使用GetCoreNumAic接口获取AI处理器可用的核数。
    cubeTiling.SetDim(numBlocks);
    ```

3.  调用核函数，参考[核函数定义和调用](核函数.md#zh-cn_topic_0000001447989210_section1915102519220)，设置核函数的numBlocks参数配置。

    ```
    matmul_custom_do(ascendcPlatform->GetCoreNumAic(), stream, x1, x2, bias, y, workspaceDevice, tilingDevice); // 仅包含Cube计算的算子，numBlocks用于设置启动多少个AIC。
    ```

4.  Kernel侧，核函数实现中增加AIV侧返回分支。

    ```
    extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b,
        GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
    {
        if (g_coreType == AscendC::AIV) { // 纯Cube模式，AIV侧直接return
            return;
        }
        ...
        // 其他代码
    }
    ```

## 验证优化方案性能收益<a name="section36229589310"></a>

-   优化后的Profiling数据如下，从C列的aic\_time数据来看，多个核中最大算子执行耗时为11.21us，较优化前的17.85us有较大提升。从G列的aic\_scalar\_time数据来看，Scalar平均耗时从优化前的15.02us降低至5.17us。

    ![](figures/zh-cn_image_0000002523289174.png)

-   优化后的流水图如下。对比优化前的流水图，红框所示位置的Scalar流水明显变稀疏。纯Cube模式相较于MIX模式，减少了对消息通信的处理，优化了整体Scalar性能开销。

    <!-- img2text -->
```text
时间轴:         0 μs              2 μs              4 μs              6 μs              8 μs             10 μs
                │                 │                 │                 │                 │                 │

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Record  Save  Load  trace.json                                                               Flow events  Processes │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

SCALAR (pid 10)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  灰色执行条                                                                                                         │
│      ▓▓▓   ▓ ▓   ▓▓  ││ │ │  ││││  │││││││││││││││││││││││││││││   │││                               │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

SCALARLDST (pid 20)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ SCALARLDST_1   │  │ │ █ │ │ │ │ │ │ │ │ │ ││ ████████████████████████    │ ███ │                             │ │  │
│ SCALARLDST_2   │    █   │ │ │ █ │ │ │ █ │ │ │ ████████████████████████    │ ██  │                             │ │  │
│ SCALARLDST_3   │  █   ███│ │ │ │ │ █ │ │ │ │ ████████████████████████      │ │   │                             │ │  │
│ SCALARLDST_4   │      █  │ │ █ │ │ │ │ │ │ │ │││││││││││││││││││││││       │ │   │                             │ █  │
│ SCALARLDST_5   │    █    │ █ │ │ │ │ │ │ │ │ ████████████████████████      │ │   │                             │    │
│ SCALARLDST_6   │  █   █  │ │ │ │ │ │ █ │ │ │ ████████████████████████          │                             │    │
│ SCALARLDST_7   │        █│ █ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │                                              │
│ SCALARLDST_8   │        █│ █ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │                                              │
│ SCALARLDST_9   │        █│ █ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │                                              │
│ SCALARLDST_10  │        █  │ │ │ │ │ │ │ │ │   │   │   │   │                                                     │
│ SCALARLDST_11  │        █  │ │ │ │ │ │ │ │ │   │   │   │   │                                                     │
│ SCALARLDST_12  │        █    █    │ │ │ │ │    │   │   │   │                                                     │
│ SCALARLDST_13  │        █         │ │ │ │ │    │   │   │   │                                                     │
│ SCALARLDST_14  │        █                                                                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                    ┌────────────────────────────── 红框区域 ──────────────────────────────┐
                    │        Scalar 流水较稀疏 / 中段有密集消息处理                         │
                    └───────────────────────────────────────────────────────────────────────┘
                                                                 ┌────────── 红框区域 ──────────┐
                                                                 │      Scalar 流水明显变稀疏   │
                                                                 └──────────────────────────────┘

CUBE (pid 40)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ CUBE_1     │              │                         WAIT_FLAG                 │        │        │        │         │
│ CUBE_2     │                                        │           │           │        │        │        │         │
│ CUBE_3     │                             MMA                 MMA                MMA      MMA      MMA              │
│ CUBE_4     │                                              WAIT_FLAG    WAIT_FLAG   WAIT_FLAG   WAIT_FLAG          │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

MTE1 (pid 50)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                   ▓▓   ▓▓▓▓     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                                 │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

MTE2 (pid 60)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                   ▓▓   ▓▓        ▓ ▓▓▓▓▓▓▓▓▓▓▓▓   ▓▓▓▓▓▓                                         │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

FIXP (pid 80)
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ FIXP_1                                  WAI...                                                                       │
│ FIXP_2                                    FIX_LOC_TO_DST    FIX_LOC_TO_DST    FIX_LOC_TO_DST    FIX_LOC_TO_DST    │
│                                                                                                      FIX_LOC_TO_DST │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 总结<a name="section252011820413"></a>

在只有矩阵计算，没有矢量计算的场景下，可以考虑使能纯Cube模式，优化Matmul计算中的消息通信性能开销，提升算子性能。

