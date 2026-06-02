# MatmulPolicy<a name="ZH-CN_TOPIC_0000002523344754"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="41.980000000000004%" id="mcps1.1.6.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" valign="top" width="14.37%" id="mcps1.1.6.1.2"><p id="p9688124518177"><a name="p9688124518177"></a><a name="p9688124518177"></a>MatmulPolicy</p>
</th>
<th class="cellrowborder" valign="top" width="13.139999999999999%" id="mcps1.1.6.1.3"><p id="p134861747161717"><a name="p134861747161717"></a><a name="p134861747161717"></a>TrianUpperMatmulPolicy/TrianLowerMatmulPolicy</p>
</th>
<th class="cellrowborder" valign="top" width="12.26%" id="mcps1.1.6.1.4"><p id="p888344921719"><a name="p888344921719"></a><a name="p888344921719"></a>NBuffer33MatmulPolicy</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="18.25%" id="mcps1.1.6.1.5"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>MatmulWithScalePolicy/SplitMMatmulPolicy/SplitNMatmulPolicy</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="41.980000000000004%" headers="mcps1.1.6.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="14.37%" headers="mcps1.1.6.1.2 "><p id="p86881345151713"><a name="p86881345151713"></a><a name="p86881345151713"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="13.139999999999999%" headers="mcps1.1.6.1.3 "><p id="p114863471177"><a name="p114863471177"></a><a name="p114863471177"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="12.26%" headers="mcps1.1.6.1.4 "><p id="p1288324991719"><a name="p1288324991719"></a><a name="p1288324991719"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="18.25%" headers="mcps1.1.6.1.5 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section451064612817"></a>

模板参数MatmulPolicy用于定义Matmul可拓展模块策略。目前支持设置以下四种Matmul内置模板策略。

-   MatmulPolicy（默认模板策略）

    使能Matmul API的默认实现策略。

-   TrianUpperMatmulPolicy（上三角模板策略）

    一次矩阵乘指令计算的结果为[baseM \* baseN](TCubeTiling结构体.md#p17899165811566)大小的矩阵块，称该矩阵块为基本块。若Matmul结果矩阵C中的基本块位于下三角位置，则Matmul内部做数据计算和数据搬出时，将忽略该基本块，最后得到的矩阵C为一个上三角矩阵。上三角模板策略如下图所示，图示中矩阵形状的相关大小为M=N=512，K=256，baseM=baseN=baseK=32。

    **图 1**  上三角模板策略示意图<a name="fig981211473248"></a>  
    <!-- img2text -->
```text
                         A矩阵                                              B矩阵

                    ┌────────────────┐                              ┌────────────────────────────────┐
                    │      256       │                              │              512               │
                    └────────────────┘                              └────────────────────────────────┘
               ┌────┬──┬──┬──┬──┬──┬──┬──┐                     ┌────┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
               │    │  │  │  │  │  │  │  │                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
               │    ├──┼──┼──┼──┼──┼──┼──┤                     │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
               │    │  │  │  │  │  │  │  │                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
               │    ├──┼──┼──┼──┼──┼──┼──┤                     │256 ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
               │    │  │  │  │  │  │  │  │                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
               │512 ├──┼──┼──┼──┼──┼──┼──┤                     │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
               │    │  │  │  │  │  │  │  │                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
               │    ├──┼──┼──┼──┼──┼──┼──┤                     │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
               │    │  │  │  │  │  │  │  │                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
               │    ├──┼──┼──┼──┼──┼──┼──┤                     └────┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
               │    │  │  │  │  │  │  │  │                                              │
               │    ├──┼──┼──┼──┼──┼──┼──┤                                              │
               │    │  │  │  │  │  │  │  │                                              │
               │    ├──┼──┼──┼──┼──┼──┼──┤                                              ↓
               │    │  │  │  │  │  │  │  │                                          C
               │    ├──┼──┼──┼──┼──┼──┼──┤                               ┌────────────────────────────────┐
               │    │  │  │  │  │  │  │  │                               │              512               │
               │    ├──┼──┼──┼──┼──┼──┼──┤                               └────────────────────────────────┘
               │    │  │  │  │  │  │  │  │                          ┌────┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
               │    ├──┼──┼──┼──┼──┼──┼──┤                          │    │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│
               │    │  │  │  │  │  │  │  │                          │    ├──┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┤
               │    ├──┼──┼──┼──┼──┼──┼──┤                          │    │  │  │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│
               │    │  │  │  │  │  │  │  │                          │    ├──┼──┼──┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┤
               │    ├──┼──┼──┼──┼──┼──┼──┤                          │    │  │  │  │  │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│
               │    │  │  │  │  │  │  │  │                          │512 ├──┼──┼──┼──┼──┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┤
               │    ├──┼──┼──┼──┼──┼──┼──┤                          │    │  │  │  │  │  │  │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│
               │    │  │  │  │  │  │  │  │                          │    ├──┼──┼──┼──┼──┼──┼──┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┤
               │    └──┴──┴──┴──┴──┴──┴──┘                          │    │  │  │  │  │  │  │  │  │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│
               └────────────────────╲                                │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┤
                                     ╲                               │    │  │  │  │  │  │  │  │  │  │  │▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│▓▓│
                                      ╲                              │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┼▓▓┤
                                       ╲                             │    │  │  │  │  │  │  │  │  │  │  │  │  │▓▓│▓▓│▓▓│▓▓│▓▓│
                                        ↘                            │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼▓▓┼▓▓┼▓▓┼▓▓┤
                                                                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │▓▓│▓▓│▓▓│
                                                                     │    ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼▓▓┼▓▓┤
                                                                     │    │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │▓▓│
                                                                     └────┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

-   TrianLowerMatmulPolicy（下三角模板策略）

    一次矩阵乘指令计算的结果为[baseM \* baseN](TCubeTiling结构体.md#p17899165811566)大小的矩阵块，称该矩阵块为基本块。若Matmul结果矩阵C中的基本块位于上三角位置，则Matmul内部做数据计算和数据搬出时，将忽略该基本块，最后得到的矩阵C为一个下三角矩阵。下三角模板策略如下图所示，图示中矩阵形状的相关大小为M=N=512，K=256，baseM=baseN=baseK=32。

    **图 2**  下三角模板策略示意图<a name="fig131201039102716"></a>  
    <!-- img2text -->
```text
A矩阵
      256
┌────────────────────────┐
│┌─┬─┬─┬─┬─┬─┬─┬─┐       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│├─┼─┼─┼─┼─┼─┼─┼─┤       │
│└─┴─┴─┴─┴─┴─┴─┴─┘       │
└────────────────────────┘
512

                                     B矩阵
                                           512
                            ┌────────────────────────────────┐
                        256 │┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐│
                            │├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                            │├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                            │├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                            │├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                            │├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                            │├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                            │└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘│
                            └────────────────────────────────┘
                                               │
                                               │
                                               │ C
                                               ▼

                            512
                  ┌────────────────────────────────┐
              512 │┌█┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐│
                  │├█┼█┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼─┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼─┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼─┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼─┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼─┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼─┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼─┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼─┼─┤│
                  │├█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼█┼─┤│
                  │└█┴█┴█┴█┴█┴█┴█┴█┴█┴█┴█┴█┴█┴█┴█┴█┘│
                  └────────────────────────────────┘

A矩阵 ───────────────────────────────↘
                                        ↘
                                          ↘
                                            ↘
                                              ↘
                                                ↘
                                                  ↘
                                                    C
```

-   NBuffer33MatmulPolicy（NBuffer33模板策略）

    一次矩阵乘指令计算的结果为[baseM \* baseN](TCubeTiling结构体.md#p17899165811566)大小的矩阵块，称该矩阵块为基本块。单核计算的A矩阵切分为3x3个基本块，该3x3个A矩阵的基本块全载和保持在L1 Buffer中，每次与3x1个B矩阵的基本块计算矩阵乘，同时[DoubleBuffer](DoubleBuffer.md)并行搬入下次计算所需的3x1个B矩阵基本块，直到singleCoreN方向的矩阵乘计算完成。NBuffer33模板策略如下图所示，图中[singleCoreM、singleCoreN、singleCoreK](TCubeTiling结构体.md#p11899125875617)表示单核内A、B矩阵的shape大小，单核计算的A矩阵切分为3x3个基本块，3x3个基本块全载在L1 Buffer上，这些基本块每次与B矩阵的3x1个基本块计算矩阵乘。

    **图 3**  NBuffer33模板策略示意图<a name="fig173669299148"></a>  
    <!-- img2text -->
```text
                         singleCoreK                                               singleCoreN
          ┌──────────┬──────────┬──────────┐                     ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
          │          │          │          │                     │          │          │          │          │          │          │
singleCoreM│          │          │          │                     │          │          │          │          │          │          │
          ├──────────┼──────────┼──────────┤                     ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
          │          │          │          │                     │          │          │          │          │          │          │
          │          │          │          │                     │          │          │          │          │          │          │
          ├──────────┼──────────┼──────────┤                     ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
          │          │          │          │                     │          │          │          │          │          │          │
          │          │          │          │                     │          │          │          │          │          │          │
          └──────────┴──────────┴──────────┘                     └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                         A矩阵                                                    B矩阵
                                        ↑
                                  singleCoreK
```

-   MatmulWithScalePolicy（MxMatmul模板策略）

    实现带有量化系数的矩阵乘法，即左矩阵和右矩阵均有对应的量化系数矩阵，左量化系数矩阵scaleA和右量化系数矩阵scaleB。MxMatmul场景中，左量化系数矩阵与左矩阵乘积，右量化系数矩阵与右矩阵乘积，对两个乘积的结果做矩阵乘法。

    图4 MxMatmul模板策略示意图

    <!-- img2text -->
```
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐   ┌──────────────────────┐   ┌───────────┐
│                 K                     │   │                  N                    │   │          N           │   │     N     │
│                                       │   │                                       │   │ bias                 │   │           │
│ M  ┌───────┐  ´ M   ┌──────────────┐  │ * │ K/32  ┌──────────────┐  ´ K  ┌──────┐ │ + │ bias                 │ = │  ┌──────┐ │
│    │scaleA │        │      A       │  │   │       │    scaleB    │       │  B   │ │   │ bias                 │   │  │  C   │ │
│    └───────┘        └──────────────┘  │   │       └──────────────┘       └──────┘ │   │ ...                  │   │  └──────┘ │
│    K/32                               │   │                           N           │   │ bias                 │   │           │
└───────────────────────────────────────┘   └───────────────────────────────────────┘   └──────────────────────┘   └───────────┘
```

- 左侧括号内：
  - 上方标注：K
  - 左侧标注：M
  - 小矩阵：scaleA，尺寸标注为 M × K/32
  - 大矩阵：A，尺寸标注为 M × K

- 中间括号内：
  - 上方标注：N
  - 小矩阵：scaleB，尺寸标注为 K/32 × N
  - 大矩阵：B，尺寸标注为 K × N

- 右侧偏置项：
  - 上方标注：N
  - 左侧标注：M
  - 内容为按列/块堆叠的 bias
  - 图中文字依次为：
    - bias
    - bias
    - bias
    - ...
    - bias

- 结果矩阵：
  - C，尺寸标注为 M × N

- 整体公式关系：
  - [scaleA, A] * [scaleB, B] + bias = C

-   SplitMMatmulPolicy（SplitM模板策略）

    Matmul一次[Iterate](Iterate.md)的计算结果从L0C Buffer搬到Unified Buffer时，采用双输出模式，即在分离模式下，AIC核与AIV核的核数比为1：2时，在调用[GetTensorC](GetTensorC.md)接口后，Matmul一次Iterate的计算结果在矩阵的M方向一分为二，将被切分后的两块结果数据分别搬运到两个AIV核的Unified Buffer。模板策略示意图如下所示。

    **图 4**  SplitM模板策略示意图<a name="fig1338985851817"></a>  
    <!-- img2text -->
```text
               baseN
        ┌────────────────┐
baseM   │                │───────────────┐
        │      AIC       │               │────────→ ┌──────────────┐
        │                │               │          │     AIV0     │
        ├────────────────┤               │          └──────────────┘
        │                │───────────────┘
        │                │
        └────────────────┘
                                           ────────→ ┌──────────────┐
                                                     │     AIV1     │
                                                     └──────────────┘
```

-   SplitNMatmulPolicy（SplitN模板策略）

    Matmul一次[Iterate](Iterate.md)的计算结果从L0C Buffer搬到Unified Buffer时，采用双输出模式，即在分离模式下，AIC核与AIV核的核数比为1：2时，在调用[GetTensorC](GetTensorC.md)接口后，Matmul一次Iterate的计算结果在矩阵的N方向一分为二，将被切分后的两块结果数据分别搬运到两个AIV核的Unified Buffer。模板策略示意图如下所示。

    **图 5**  SplitN模板策略示意图<a name="fig1117644818584"></a>  
    <!-- img2text -->
```text
                 baseN
      ┌───────────────────────────────┐
      │               │               │
baseM │               │               │
      │     AIC       │               │
      │               │               │
      └───────┬───────┴───────┬───────┘
              │               │
              │               │
              ↓               ↓
        ┌───────────┐   ┌───────────┐
        │   AIV0    │   │   AIV1    │
        └───────────┘   └───────────┘
```

## 约束说明<a name="section176217152520"></a>

-   TrianUpperMatmulPolicy当前只支持[Norm模板](MatmulConfig.md#p159827389308)和[MDL模板](MatmulConfig.md#p109823386305)。
-   TrianLowerMatmulPolicy当前只支持[Norm模板](MatmulConfig.md#p159827389308)和[MDL模板](MatmulConfig.md#p109823386305)。
-   NBuffer33MatmulPolicy：
    -   当前只支持[MDL模板](MatmulConfig.md#p109823386305)。
    -   A矩阵、B矩阵的内存逻辑位置只支持TPosition::GM。
    -   暂不支持MIX模式（包含矩阵计算和矢量计算），仅支持纯Cube模式（只有矩阵计算）。
    -   只支持通过[IterateAll](IterateAll.md)接口获取Matmul的计算结果C矩阵。
    -   [stepM、stepKa、stepKb](TCubeTiling结构体.md#p139009583566)小于等于3，且满足：stepKa=stepKb=ceil\([singleCoreK](TCubeTiling结构体.md#p11899125875617)/baseK\)。
    -   A矩阵全载的基本块大小与B矩阵载入的基本块大小之和不超过L1 Buffer大小。
    -   在使用[GetTiling](GetTiling.md)接口生成Tiling参数前，必须通过[SetMatmulConfigParams](SetMatmulConfigParams.md)接口将scheduleTypeIn参数设置为ScheduleType::N\_BUFFER\_33，以启用NBuffer33模板策略的Tiling生成逻辑。

-   MatmulWithScalePolicy当前只支持[Norm模板](MatmulConfig.md#p159827389308)和[MDL模板](MatmulConfig.md#p109823386305)。
-   SplitMMatmulPolicy：
    -   只支持C矩阵输出到Unified Buffer。
    -   A矩阵、B矩阵类型信息MatmulType中的参数[IBSHARE](Matmul使用说明.md#p1613334125414)必须为true。

-   SplitNMatmulPolicy：
    -   只支持C矩阵输出到Unified Buffer。
    -   baseN必须满足是16的倍数。
    -   [Tiling参数](TCubeTiling结构体.md#table1563162142915)必须满足：singleCoreM = baseM，singleCoreN = baseN，singleCoreK = baseK。
    -   A矩阵、B矩阵类型信息MatmulType中的参数[IBSHARE](Matmul使用说明.md#p1613334125414)必须为true。

## 调用示例<a name="section2366122094418"></a>

默认模板策略MatmulPolicy为模板参数的默认值，下面主要介绍TrianUpperMatmulPolicy（上三角模板策略）和TrianLowerMatmulPolicy（下三角模板策略）的使用方式。

-   上三角模板策略使用示例

    完整的算子样例请参考[matmul\_triangle样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_triangle)。

    ```
    #include "lib/matmul_intf.h"
    
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    // Matmul定义时传入TrianUpperMatmulPolicy
    AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::TrianUpperMatmulPolicy> mm; 
    
    // 常规Matmul计算，最后输出上三角形式的结果
    TPipe pipe;
    TCubeTiling tiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gmA, isTransposeA);
    mm.SetTensorB(gmB, isTransposeB);
    if (tiling.isBias) {
        mm.SetBias(gmBias);
    }
    mm.IterateAll(gmC);
    mm.End();
    ```

-   下三角模板策略使用示例

    完整的算子样例请参考[matmul\_triangle样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_triangle)。

    ```
    #include "lib/matmul_intf.h"
    
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    // Matmul定义时传入TrianLowerMatmulPolicy
    AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::TrianLowerMatmulPolicy> mm; 
    
    // 常规Matmul计算，最后输出下三角形式的结果
    TPipe pipe;
    TCubeTiling tiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gmA, isTransposeA);
    mm.SetTensorB(gmB, isTransposeB);
    if (tiling.isBias) {
        mm.SetBias(gmBias);
    }
    mm.IterateAll(gmC);
    mm.End();
    ```

-   NBuffer33模板策略使用示例

    完整的算子样例请参考[matmul\_nbuffer33样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_nbuffer33)。

    ```
    #include "lib/matmul_intf.h"
    
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    // Matmul定义时传入NBuffer33MatmulPolicy
    
    AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::NBuffer33MatmulPolicy> mm; 
    
    // 常规Matmul计算，最后输出下三角形式的结果
    TPipe pipe;
    TCubeTiling tiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gmA, isTransposeA);
    mm.SetTensorB(gmB, isTransposeB);
    if (tiling.isBias) {
        mm.SetBias(gmBias);
    }
    mm.IterateAll(gmC);
    mm.End();
    ```

-   MxMatmul模板策略使用示例

    ```
    #include "lib/matmul_intf.h"
    typedef MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, AType, isTransposeA> aType;
    typedef MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, BType, isTransposeB> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    // Matmul定义时传入MatmulWithScalePolicy
    AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy> mm; 
    
    // MxMatmul计算逻辑，最后输出结果
    TPipe pipe;
    TCubeTiling tiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gmA, isTransposeA);
    mm.SetTensorB(gmB, isTransposeB);
    mm.SetTensorScaleA(gm_scaleA, isTransposeScaleA);
    mm.SetTensorScaleB(gm_scaleB, isTransposeScaleB);
    if (tiling.isBias) {
        mm.SetBias(gmBias);
    }
    mm.IterateAll(gmC);
    mm.End();
    ```

-   SplitM模板策略使用示例

    更多算子样例请参考[matmul\_splitm样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/matmul_splitm)。

    ```
    #include "lib/matmul_intf.h"
    
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, LayoutMode::NONE, true> aType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, LayoutMode::NONE, true> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    // Matmul定义时传入SplitMMatmulPolicy
    AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::SplitMMatmulPolicy> mm;
    // Matmul计算
    TPipe pipe;
    TCubeTiling tiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gmA, isTransposeA);
    mm.SetTensorB(gmB, isTransposeB);
    if (tiling.isBias) {
        mm.SetBias(gmBias);
    }
    
    // 调用GetTensorC接口后，将Matmul一次Iterate的计算结果一分为二，搬运到两个AIV核的Unified Buffer。
    pipe.InitBuffer(resultCMatrix, 1, tiling.M * tiling.N * sizeof(C_T));
    mm.template Iterate<false>();
    bufferC = resultCMatrix.AllocTensor<C_T>();
    uint16_t nIter_ = Ceil(tiling.singleCoreN, tiling.baseN);
    uint16_t mIter_ = Ceil(tiling.singleCoreM, tiling.baseM);
    uint16_t mnIter_ = nIter_ * mIter_;
    uint16_t size = tiling.baseM / 2 * tiling.baseN;
    for (int i = 0; i < mnIter_; i++) {
         mm.template GetTensorC<false>(bufferC, false, false);  // false // kfc vec0 iterate             
         PipeBarrier<PIPE_ALL>();
    }
    mm.End();
    resultCMatrix.EnQue(bufferC);
    bufferC = resultCMatrix.DeQue<C_T>();
    
    uint16_t baseOffset = tiling.M / 2 * tiling.N;
    uint16_t stride = tiling.M / 2 * tiling.N * sizeof(C_T) / 32;  // 32B
    const uint16_t blockCount = tiling.M / tiling.M;
    if (GetSubBlockIdxImpl() == 0) {
        DataCopy(gmC, bufferC, {blockCount, stride, stride, stride});
    } else {
        DataCopy(gmC[baseOffset], bufferC, {blockCount, stride, stride, stride});
    }
    ```

-   SplitN模板策略使用示例

    ```
    #include "lib/matmul_intf.h"
    
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, LayoutMode::NONE, true> aType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, LayoutMode::NONE, true> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
    // Matmul定义时传入SplitNMatmulPolicy
    AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::SplitNMatmulPolicy> mm;
    // Matmul计算
    TPipe pipe;
    TCubeTiling tiling;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    mm.SetTensorA(gmA, isTransposeA);
    mm.SetTensorB(gmB, isTransposeB);
    if (tiling.isBias) {
        mm.SetBias(gmBias);
    }
    
    // 调用GetTensorC接口后，将Matmul一次Iterate的计算结果一分为二，搬运到两个AIV核的Unified Buffer。
    pipe.InitBuffer(resultCMatrix, 1, tiling.M * tiling.N * sizeof(C_T));
    mm.template Iterate<false>();
    bufferC = resultCMatrix.AllocTensor<C_T>();
    uint16_t nIter_ = Ceil(tiling.singleCoreN, tiling.baseN);
    uint16_t mIter_ = Ceil(tiling.singleCoreM, tiling.baseM);
    uint16_t mnIter_ = nIter_ * mIter_;
    uint16_t size = tiling.baseM / 2 * tiling.baseN;
    for (int i = 0; i < mnIter_; i++) {
         mm.template GetTensorC<false>(bufferC, false, false);  // false // kfc vec0 iterate             
         PipeBarrier<PIPE_ALL>();
    }
    mm.End();
    resultCMatrix.EnQue(bufferC);
    bufferC = resultCMatrix.DeQue<C_T>();
    
    uint16_t baseOffset = tiling.N / 2;
    uint16_t blockCount = tiling.M;
    uint16_t blockLen = (tiling.N / 2 * sizeof(C_T)) / 32;
    uint16_t srcStride = 0;
    uint16_t dstStride = (tiling.N / 2 * sizeof(C_T)) / 32;
    if (GetSubBlockIdxImpl() == 0) {
        DataCopy(gmC, bufferC, {blockCount, blockLen, srcStride, dstStride});
    } else {
        DataCopy(gmC[baseOffset], bufferC, {blockCount, blockLen, srcStride, dstStride});
    }
    ```

