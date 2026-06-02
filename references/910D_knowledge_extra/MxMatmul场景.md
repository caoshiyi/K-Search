# MxMatmul场景<a name="ZH-CN_TOPIC_0000002554329027"></a>

## 背景介绍<a name="zh-cn_topic_0000002270097206_section182549454374"></a>

浮点数在科学计算、图像处理、神经网络等领域应用广泛。以AI训练为例，现有的浮点数格式或数值范围不足，或精度不高，这影响了模型的收敛速度和性能。如果要同时满足数值范围和精度的要求，将会导致内存占用过大，从而增加数据存储和传输的成本。基于此种情况，业内提出了一种新的浮点数格式——微缩放（Microscaling，MX）格式。MX格式的浮点数可以支持更低比特位宽的AI训练和推理，并且占用的内存更少。符合MX标准的数据格式在使用8位或更低比特位的情况下，能够实现稳健的AI训练和推理模型精度。

MX格式是一种块数据格式，若干个数据可以组成一个块（或者一个组），数据以块为单位。MX格式的数据由三部分构成：

-   共享缩放因子X，位宽为w bits；
-   私有元素P<sub>i</sub>，位宽为d bits；
-   块大小k，表示多少个低比特数据形成一个块；

所有k个元素P<sub>i</sub>有相同的位宽和数据类型，并且共享一个缩放因子X，每个包含k个元素的块可以使用（w+k\*d）位进行编码。元素的数据类型和缩放因子可以独立选择。

下图为MX格式的浮点数的数据结构，S、E和M分别用于表示浮点数的符号、指数和尾数字段的值。其中，共享缩放因子X是一个用于整个数据块的缩放比例因子，它决定了数据块中所有元素的动态范围。通过引入共享缩放因子，MX格式的数据能够在保持低位宽的同时，灵活地表示不同范围的数据。块大小k指的是组成一个数据块（或组）的低比特数据的数量。私有元素P<sub>i</sub>是指数据块中的每个低比特数据元素。这些元素经过缩放因子X的调整后，共同表示了一个高精度的浮点数或整数。

**图 1**  MX格式组成示意图<a name="zh-cn_topic_0000002270097206_fig11888194743220"></a>  
<!-- img2text -->
```
                          d bits
                 ┌───────────────────────────────┐
                 │                               │
         P1      │ ┌────┬────────┬────────────┐  │
                 │ │ S1 │   E1   │     M1     │  │
                 │ └────┴────────┴────────────┘  │
                 │              ·                │
                 │              ·                │
                 │              ·                │
         PK      │ ┌────┬────────┬────────────┐  │
                 │ │ Sk │   Ek   │     Mk     │  │
                 │ └────┴────────┴────────────┘  │
                 └───────────────────────────────┘
                                                 │
                                                 ↕ k

w bits
┌──────────────────────┐
│          X           │
│   (shared scale)     │
└──────────────────────┘
```

MX格式的数据类型包含多种，例如MXFP8、MXFP4、MXFP16、MXINT4等。下表列举了[MxMatmul场景](#zh-cn_topic_0000002270097206_section310824820358)（全称Microscaling Matmul）支持的数据类型。

**表 1**  MxMatmul支持MX格式的数据类型

<a name="zh-cn_topic_0000002270097206_table5383144710452"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002270097206_row193837473457"><th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.1"><p id="zh-cn_topic_0000002270097206_p410517278313"><a name="zh-cn_topic_0000002270097206_p410517278313"></a><a name="zh-cn_topic_0000002270097206_p410517278313"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.2"><p id="zh-cn_topic_0000002270097206_p12383184711457"><a name="zh-cn_topic_0000002270097206_p12383184711457"></a><a name="zh-cn_topic_0000002270097206_p12383184711457"></a>私有元素数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.3"><p id="zh-cn_topic_0000002270097206_p2383647184516"><a name="zh-cn_topic_0000002270097206_p2383647184516"></a><a name="zh-cn_topic_0000002270097206_p2383647184516"></a>私有元素位宽（d）</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.4"><p id="zh-cn_topic_0000002270097206_p16383847154510"><a name="zh-cn_topic_0000002270097206_p16383847154510"></a><a name="zh-cn_topic_0000002270097206_p16383847154510"></a>块大小(k)</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.5"><p id="zh-cn_topic_0000002270097206_p1238334784520"><a name="zh-cn_topic_0000002270097206_p1238334784520"></a><a name="zh-cn_topic_0000002270097206_p1238334784520"></a>共享缩放因子数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="16.666666666666664%" id="mcps1.2.7.1.6"><p id="zh-cn_topic_0000002270097206_p113831847114514"><a name="zh-cn_topic_0000002270097206_p113831847114514"></a><a name="zh-cn_topic_0000002270097206_p113831847114514"></a>共享缩放因子位宽(w)</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002270097206_row238320470455"><td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.1 "><p id="zh-cn_topic_0000002270097206_p9383134734518"><a name="zh-cn_topic_0000002270097206_p9383134734518"></a><a name="zh-cn_topic_0000002270097206_p9383134734518"></a>MXFP8</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p11383447164519"><a name="zh-cn_topic_0000002270097206_p11383447164519"></a><a name="zh-cn_topic_0000002270097206_p11383447164519"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.3 "><p id="zh-cn_topic_0000002270097206_p33831947194516"><a name="zh-cn_topic_0000002270097206_p33831947194516"></a><a name="zh-cn_topic_0000002270097206_p33831947194516"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.4 "><p id="zh-cn_topic_0000002270097206_p1338311471456"><a name="zh-cn_topic_0000002270097206_p1338311471456"></a><a name="zh-cn_topic_0000002270097206_p1338311471456"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.5 "><p id="zh-cn_topic_0000002270097206_p1661204742410"><a name="zh-cn_topic_0000002270097206_p1661204742410"></a><a name="zh-cn_topic_0000002270097206_p1661204742410"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="zh-cn_topic_0000002270097206_p149798459481"><a name="zh-cn_topic_0000002270097206_p149798459481"></a><a name="zh-cn_topic_0000002270097206_p149798459481"></a>8</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row133831647204514"><td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.1 "><p id="p448511510259"><a name="p448511510259"></a><a name="p448511510259"></a>MXFP8</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p09723273916"><a name="zh-cn_topic_0000002270097206_p09723273916"></a><a name="zh-cn_topic_0000002270097206_p09723273916"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.3 "><p id="p175322662417"><a name="p175322662417"></a><a name="p175322662417"></a>8</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.4 "><p id="p11711191614244"><a name="p11711191614244"></a><a name="p11711191614244"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.5 "><p id="p75995345248"><a name="p75995345248"></a><a name="p75995345248"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p11842154562416"><a name="p11842154562416"></a><a name="p11842154562416"></a>8</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row1138444719456"><td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.1 "><p id="zh-cn_topic_0000002270097206_p173841478457"><a name="zh-cn_topic_0000002270097206_p173841478457"></a><a name="zh-cn_topic_0000002270097206_p173841478457"></a>MXFP4</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p158615210394"><a name="zh-cn_topic_0000002270097206_p158615210394"></a><a name="zh-cn_topic_0000002270097206_p158615210394"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.3 "><p id="zh-cn_topic_0000002270097206_p238444704512"><a name="zh-cn_topic_0000002270097206_p238444704512"></a><a name="zh-cn_topic_0000002270097206_p238444704512"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.4 "><p id="zh-cn_topic_0000002270097206_p2038434716454"><a name="zh-cn_topic_0000002270097206_p2038434716454"></a><a name="zh-cn_topic_0000002270097206_p2038434716454"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.5 "><p id="p205991534102413"><a name="p205991534102413"></a><a name="p205991534102413"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p1842145182418"><a name="p1842145182418"></a><a name="p1842145182418"></a>8</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row12384154710452"><td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.1 "><p id="p10647179102513"><a name="p10647179102513"></a><a name="p10647179102513"></a>MXFP4</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p8952103819241"><a name="zh-cn_topic_0000002270097206_p8952103819241"></a><a name="zh-cn_topic_0000002270097206_p8952103819241"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.3 "><p id="p69001011132410"><a name="p69001011132410"></a><a name="p69001011132410"></a>4</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.4 "><p id="p250011195246"><a name="p250011195246"></a><a name="p250011195246"></a>32</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.5 "><p id="p125992348249"><a name="p125992348249"></a><a name="p125992348249"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="16.666666666666664%" headers="mcps1.2.7.1.6 "><p id="p284212453244"><a name="p284212453244"></a><a name="p284212453244"></a>8</p>
</td>
</tr>
</tbody>
</table>

## 功能介绍<a name="zh-cn_topic_0000002270097206_section310824820358"></a>

MxMatmul（全称Microscaling Matmul）为带有量化系数的矩阵乘法，即左矩阵和右矩阵均有对应的量化系数矩阵，左量化系数矩阵scaleA和右量化系数矩阵scaleB。MxMatmul场景中，左量化系数矩阵与左矩阵乘积，右量化系数矩阵与右矩阵乘积，对两个乘积的结果做矩阵乘法。

MxMatmul的计算公式为：C = \(scaleA ⊗ A\) \* \(scaleB ⊗ B\) + Bias，“⊗”表示广播乘法，左/右矩阵与左/右量化系数矩阵做乘积时，K方向上每32个元素共享一个量化因子，如[图2](#zh-cn_topic_0000002270097206_fig1942919398330)所示。

-   A、scaleA、B、scaleB为源操作数。A为左矩阵，形状为\[M, K\]；scaleA为左量化系数矩阵，形状为\[M, K/32\]；B为右矩阵，形状为\[K, N\]；scaleB为右量化系数矩阵，形状为\[K/32, N\]。
-   C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N\]。
-   Bias为矩阵乘偏置，形状为\[1, N\]。对\(scaleA ⊗ A\) \* \(scaleB ⊗ B\)结果矩阵的每一行都采用该Bias进行偏置。

**图 2**  MxMatmul矩阵乘示意图<a name="zh-cn_topic_0000002270097206_fig1942919398330"></a>  
<!-- img2text -->
```
           K/32                 K                              N                    N                           N                    N
        ┌────────┐          ┌──────────┐                  ┌────────┐          ┌──────────┐                ┌──────────┐          ┌──────────┐
        │ 0 1 2… │          │          │                  │ 4      │          │          │                │   bias   │          │          │
M       │ scaleA │ ⊗  M     │    A     │        K/32      │ 5      │ ⊗  K     │    B     │        +  M    │   bias   │   =  M   │    C     │
        │        │          │          │                  │ 6      │          │          │                │   bias   │          │          │
        └────────┘          └──────────┘                  │ …      │          └──────────┘                │   …      │          └──────────┘
                                                          │ scaleB │                                        └──────────┘
                                                          └────────┘


                                  ↓                                                ↓

┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   K方向共享量化因子，等效图                                               │
│                                                                                                              │
│   32个                                                                                                       │
│   <───>                                                                                                      │
│       K                                                                                                      │
│   ┌──────────────┐                                                      N                                    │
│   │000 ...111...222...│                                          32个                                       │
│ M │    scaleA     │                                              <───>                                      │
│   │               │                                                  K                                       │
│   └──────────────┘                                              ┌──────────────┐                           │
│                                                                  │ 4            │                           │
│                                                                  │ 4            │                           │
│                                                                  │ 4            │                           │
│                                                                  │ 4            │                           │
│                                                                  │ 5            │                           │
│                                                                  │ 5            │                           │
│                                                                  │ 5            │                           │
│                                                                  │ …            │                           │
│                                                                  │   scaleB     │                           │
│                                                                  └──────────────┘                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

矩阵A、scaleA、B、scaleB在不同位置中的排布格式分别如下图所示。

**图 3**  A矩阵在不同位置的排布格式<a name="zh-cn_topic_0000002270097206_fig384531413014"></a>  
<!-- img2text -->
```
                                   ┌───────────────────────────────┐      ┌───────────────────────────────┐
                                   │              GM               │      │         GM/A1/VECOUT          │
                                   │             ┌────┐            │      │             ┌────┐            │
                                   │             │ ND │            │      │             │ NZ │            │
                                   │             └────┘            │      │             └────┘            │
                                   └───────────────────────────────┘      └───────────────────────────────┘
                                                    ⋮                                      ⋮
                                                    ⋮                                      ⋮

矩阵A：[M, K]

                 K                                                    K
      ┌──────────────────────────────┐                    ┌───────┬───────┬───────┬───────┐
  M   │ ───────────────────────────→ │                M   │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
      │                            ╱ │                    │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
      │                          ╱   │                    │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
      │                        ╱     │                    │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
      │                      ╱       │                    │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
      │                    ╱         │                    └───────┴───────┴───────┴───────┘
      │                  ╱           │
      │                ╱             │
      │              ╱               │
      │            ╱                 │
      │          ╱                   │
      │        ╱                     │
      │      ╱                       │
      │    ╱                         │
      │  ↙─────────────────────────→ │
      └──────────────────────────────┘


矩阵A：[K, M]

                            M                                             M
               ┌────────────────────┐                        ┌───────┬───────┐
               │ ────────────────→  │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │                ╱   │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │              ╱     │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │            ╱       │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
           K   │          ╱         │                    K   │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │        ╱           │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │      ╱             │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │    ╱               │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │  ╱                 │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               │↙───────────────→   │                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
               └────────────────────┘                        │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
                                                             │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
                                                             │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
                                                             │↔↔↔↔↔↔↕│↔↔↔↔↔↔↕│
                                                             └───────┴───────┘
```

说明:
- 左右两列由竖向虚线分隔，左侧为 `GM + ND`，右侧为 `GM/A1/VECOUT + NZ`。
- 图中右侧 NZ 区域内的粉色斜条纹仅用于视觉区分分块，未在 ASCII 框图中单独表示颜色。
- `矩阵A：[M, K]` 的 NZ 排布被划分为 4 个纵向块；`矩阵A：[K, M]` 的 NZ 排布被划分为 2 个纵向块。
- 左侧 ND 图中的箭头表示线性访问/排布顺序；右侧 NZ 图中的每个小块内为往返式横向访问箭头。

**图 4**  B矩阵在不同位置的排布格式<a name="zh-cn_topic_0000002270097206_fig674419451902"></a>  
<!-- img2text -->
```
                                   ┌───────────────────────────────┐         ┌───────────────────────────────┐
                                   │             GM                │         │         GM/B1/VECOUT         │
                                   │             ND                │         │              NZ               │
                                   └───────────────────────────────┘         └───────────────────────────────┘

矩阵B：[K, N]

                         N                                                   N
                         │                                                   │
                ┌────────────────┐                                 ┌────────────────────┐
                │ ─────────────→ │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │              ╱ │                                 │ ↔  ↔ │ ↔  ↔ │      │
              K │            ╱   │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │          ╱     │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │        ╱       │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │      ╱         │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │    ╱           │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │  ╱             │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │╱               │                                 │ ↔  ↔ │ ↔  ↔ │      │
                │ ─────────────→ │                                 │ ↔  ↔ │ ↔  ↔ │      │
                └────────────────┘                                 │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                 K │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                   │ ↔  ↔ │ ↔  ↔ │      │
                                                                   └────────────────────┘


矩阵B：[N, K]

                                   K                                                   K
                                   │                                                   │
                ┌───────────────────────────────┐                     ┌───────────────────────────────┐
                │ ───────────────────────────→  │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │                           ╱   │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │                        ╱      │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
              N │                     ╱         │                   N │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │                  ╱            │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │               ╱               │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │            ╱                  │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │         ╱                     │                     │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ ↔  ↔ │ │
                │      ╱                        │                     └───────────────────────────────┘
                │   ╱                           │
                │ ←───────────────────────────  │
                └───────────────────────────────┘
```

说明:
- 左侧为 `GM / ND` 排布，右侧为 `GM/B1/VECOUT / NZ` 排布。
- 上半部分对应 `矩阵B：[K, N]`，下半部分对应 `矩阵B：[N, K]`。
- 左侧 ND 图中的箭头表示矩阵在线性内存中的访问/排布方向：
  - `[K, N]`：顶行从左到右，底行从左到右，内部用对角线连接。
  - `[N, K]`：顶行从左到右，底行从右到左，内部用对角线连接。
- 右侧 NZ 图中，每个竖向小块表示分块后的 NZ 排布单元，块内多行双向箭头表示按块交错存放/访问。
- 原图右侧 NZ 区域带有淡红色斜条背景，仅用于视觉区分分块区域。

**图 5**  scaleA矩阵在不同位置的排布格式<a name="zh-cn_topic_0000002270097206_fig107863142019"></a>  
<!-- img2text -->
```
                         ┌───────────────────────┬────────┐      ┆      ┌──────────────────────────────┬────────┐
                         │          GM           │   ND   │      ┆      │         GM/A1/VECOUT         │   NZ   │
                         └───────────────────────┴────────┘      ┆      └──────────────────────────────┴────────┘


scaleA不转置：[M, K/64,2]                                          ┆

                              <──────────── K/32 ────────────>
                              ┌─┬─┬─┬─────┬─┐
                              │2│2│2│ ... │2│
                              ├──────────────────────────────┤
                           M  │ → → → →   ...   → → → →    │
                              │ ← ← ← ←   ...   ← ← ← ←    │
                              │ → → → →   ...   → → → →    │
                              │      ⋮                      │
                              │ → → → →   ...   → → → →    │
                              │ ← ← ← ←   ...   ← ← ← ←    │
                              └──────────────────────────────┘


scaleA转置：[K/64,M,2]                                             ┆                        K/32
                                                                 ┆                   <────────────────>
                              <──────────── K/32 ────────────>   ┆                      <─2 Byte─>
                              ┌─┬─┬─┬─────┬─┐                    ┆                    ┌─┬─┬─┬─────┬─┐
                              │2│2│2│ ... │2│                    ┆                 16 │↕│↕│↕│ ... │↕│
                              ├──────────────────────────────┤    ┆                    │↕│↕│↕│     │↕│
                           M  │ ↓  ↓  ↓      ...      ↓     │    ┆                    ├──────────────┤
                              │ ↓  ↓  ↓               ↓     │    ┆                  M │↔↔↔↔   ...  ↔│
                              │ ↓  ↓  ↓               ↓     │    ┆                    ├──────────────┤
                              │ ↓  ↓  ↓               ↓     │    ┆                    │↕│↕│↕│ ... │↕│
                              │ ...                        │    ┆                    │↕│↕│↕│     │↕│
                              │ ↓  ↓  ↓      ...      ↓  → │    ┆                    └─┴─┴─┴─────┴─┘
                              └──────────────────────────────┘    ┆
```

**图 6**  scaleB矩阵在不同位置的排布格式<a name="zh-cn_topic_0000002270097206_fig76682054103416"></a>  
<!-- img2text -->
```
┌───────────────────────────────────────┐     ┌───────────────────────────────────────┐
│                  GM                   │     │              GM/B1/VECOUT             │
│                  ND                   │     │                  NZ                   │
└───────────────────────────────────────┘     └───────────────────────────────────────┘


scaleB不转置：[K/64,N,2]                    N

                     K/32
              <────────────────>
                2    2    2      2
              <─><─><─>      <─>
              ┌──┬──┬──┬──────┬──┐
              │╲↓│╲↓│╲↓│  ... │ ↓│
              │ ╲│ ╲│ ╲│      │ ││
              │↓╲│↓╲│↓╲│      │ ││
              │ ╲│ ╲│ ╲│      │ ││
              │↓╲│↓╲│↓╲│      │ ││
              │ ╲│ ╲│ ╲│      │ ││
              │↓→│↓→│↓→│      │↓→│
              └──┴──┴──┴──────┴──┘


scaleB转置：[N,K/64,2]                      N

                     K/32
              <────────────────>
                2    2    2      2
              <─><─><─>      <─>
              ┌──┬──┬──┬──────┬──┐
              │→→│→→│→→│ ...  │→→│
              │←←│←←│←←│ ...  │←←│
              │                    │
              │                    │
              │         ⋮          │
              │                    │
              │→→│→→│→→│ ...  │→→│
              │←←│←←│←←│ ...  │←←│
              └──┴──┴──┴──────┴──┘


                                   16
                              <────────>
                                 N
                           ┌────────────────┐
                     2 Byte │↓↔│↓↔│↓↔│↓↔│↓↔│
                           │↓↔│↓↔│↓↔│↓↔│↓↔│
                       K/32│↓↔│↓↔│↓↔│↓↔│↓↔│
                           │↓↔│↓↔│↓↔│↓↔│↓↔│
                           │↓↔│↓↔│↓↔│↓↔│↓↔│
                           │↓↔│↓↔│↓↔│↓↔│↓↔│
                           │↓↔│↓↔│↓↔│↓↔│↓↔│
                           │↓↔│↓↔│↓↔│↓↔│↓↔│
                           └────────────────┘
```

说明:
- 左侧为 ND 格式下 scaleB 矩阵在 GM 中的两种排布：
  - `scaleB不转置：[K/64,N,2]`
  - `scaleB转置：[N,K/64,2]`
- 右侧为 `GM/B1/VECOUT` 中的 `NZ` 排布。
- 图中顶部标注保留为：
  - 左：`GM`、`ND`
  - 右：`GM/B1/VECOUT`、`NZ`
- 尺寸标注保留为：`K/32`、`N`、`16`、`2 Byte`、以及各分块上的 `2`
- 图中粉色斜纹区域表示若干分组块，ASCII 中以带斜向/纵向箭头的分块示意，`...`/`⋮` 表示中间省略的重复部分。

## 使用场景<a name="zh-cn_topic_0000002270097206_section118051016163613"></a>

矩阵计算之前，需要对A、B矩阵进行量化操作的场景。当前该场景下，Matmul输入输出矩阵支持的数据类型如下表所示。

**表 2**  MxMatmul支持的量化场景

<a name="zh-cn_topic_0000002270097206_table2844113219191"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002270097206_row4844732121915"><th class="cellrowborder" valign="top" width="10.41%" id="mcps1.2.7.1.1"><p id="zh-cn_topic_0000002270097206_p1084417322199"><a name="zh-cn_topic_0000002270097206_p1084417322199"></a><a name="zh-cn_topic_0000002270097206_p1084417322199"></a>A矩阵</p>
</th>
<th class="cellrowborder" valign="top" width="20.599999999999998%" id="mcps1.2.7.1.2"><p id="zh-cn_topic_0000002270097206_p178441532121916"><a name="zh-cn_topic_0000002270097206_p178441532121916"></a><a name="zh-cn_topic_0000002270097206_p178441532121916"></a>B矩阵</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.2.7.1.3"><p id="zh-cn_topic_0000002270097206_p9844203201914"><a name="zh-cn_topic_0000002270097206_p9844203201914"></a><a name="zh-cn_topic_0000002270097206_p9844203201914"></a>ScaleA矩阵/ScaleB矩阵</p>
</th>
<th class="cellrowborder" valign="top" width="18.85%" id="mcps1.2.7.1.4"><p id="zh-cn_topic_0000002270097206_p6532145910546"><a name="zh-cn_topic_0000002270097206_p6532145910546"></a><a name="zh-cn_topic_0000002270097206_p6532145910546"></a>Bias矩阵</p>
</th>
<th class="cellrowborder" valign="top" width="17.72%" id="mcps1.2.7.1.5"><p id="zh-cn_topic_0000002270097206_p131201522142011"><a name="zh-cn_topic_0000002270097206_p131201522142011"></a><a name="zh-cn_topic_0000002270097206_p131201522142011"></a>C矩阵</p>
</th>
<th class="cellrowborder" valign="top" width="17.080000000000002%" id="mcps1.2.7.1.6"><p id="zh-cn_topic_0000002270097206_p1012052214207"><a name="zh-cn_topic_0000002270097206_p1012052214207"></a><a name="zh-cn_topic_0000002270097206_p1012052214207"></a>支持平台</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002270097206_row12844932171910"><td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.2.7.1.1 "><p id="zh-cn_topic_0000002270097206_p7777112717207"><a name="zh-cn_topic_0000002270097206_p7777112717207"></a><a name="zh-cn_topic_0000002270097206_p7777112717207"></a>fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.599999999999998%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p57771127152013"><a name="zh-cn_topic_0000002270097206_p57771127152013"></a><a name="zh-cn_topic_0000002270097206_p57771127152013"></a>fp4x2_e1m2_t/fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.7.1.3 "><p id="zh-cn_topic_0000002270097206_p1377792713202"><a name="zh-cn_topic_0000002270097206_p1377792713202"></a><a name="zh-cn_topic_0000002270097206_p1377792713202"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.85%" headers="mcps1.2.7.1.4 "><p id="zh-cn_topic_0000002270097206_p4532125918544"><a name="zh-cn_topic_0000002270097206_p4532125918544"></a><a name="zh-cn_topic_0000002270097206_p4532125918544"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.7.1.5 "><p id="zh-cn_topic_0000002270097206_p6777827162018"><a name="zh-cn_topic_0000002270097206_p6777827162018"></a><a name="zh-cn_topic_0000002270097206_p6777827162018"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.7.1.6 "><p id="p355413382268"><a name="p355413382268"></a><a name="p355413382268"></a><span id="zh-cn_topic_0000002270097206_ph3777122762015"><a name="zh-cn_topic_0000002270097206_ph3777122762015"></a><a name="zh-cn_topic_0000002270097206_ph3777122762015"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row11845153220193"><td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.2.7.1.1 "><p id="zh-cn_topic_0000002270097206_p377742710204"><a name="zh-cn_topic_0000002270097206_p377742710204"></a><a name="zh-cn_topic_0000002270097206_p377742710204"></a>fp4x2_e2m1_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.599999999999998%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p177771427132012"><a name="zh-cn_topic_0000002270097206_p177771427132012"></a><a name="zh-cn_topic_0000002270097206_p177771427132012"></a>fp4x2_e2m1_t/fp4x2_e1m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.7.1.3 "><p id="zh-cn_topic_0000002270097206_p17777162715206"><a name="zh-cn_topic_0000002270097206_p17777162715206"></a><a name="zh-cn_topic_0000002270097206_p17777162715206"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.85%" headers="mcps1.2.7.1.4 "><p id="zh-cn_topic_0000002270097206_p853295911547"><a name="zh-cn_topic_0000002270097206_p853295911547"></a><a name="zh-cn_topic_0000002270097206_p853295911547"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.7.1.5 "><p id="zh-cn_topic_0000002270097206_p57779278201"><a name="zh-cn_topic_0000002270097206_p57779278201"></a><a name="zh-cn_topic_0000002270097206_p57779278201"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.7.1.6 "><p id="p15680154020265"><a name="p15680154020265"></a><a name="p15680154020265"></a><span id="zh-cn_topic_0000002270097206_ph1677719278204"><a name="zh-cn_topic_0000002270097206_ph1677719278204"></a><a name="zh-cn_topic_0000002270097206_ph1677719278204"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row0845173241915"><td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.2.7.1.1 "><p id="zh-cn_topic_0000002270097206_p1677712716209"><a name="zh-cn_topic_0000002270097206_p1677712716209"></a><a name="zh-cn_topic_0000002270097206_p1677712716209"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.599999999999998%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p87771927182015"><a name="zh-cn_topic_0000002270097206_p87771927182015"></a><a name="zh-cn_topic_0000002270097206_p87771927182015"></a>fp8_e4m3fn_t/fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.7.1.3 "><p id="zh-cn_topic_0000002270097206_p1577832742016"><a name="zh-cn_topic_0000002270097206_p1577832742016"></a><a name="zh-cn_topic_0000002270097206_p1577832742016"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.85%" headers="mcps1.2.7.1.4 "><p id="zh-cn_topic_0000002270097206_p6532125914546"><a name="zh-cn_topic_0000002270097206_p6532125914546"></a><a name="zh-cn_topic_0000002270097206_p6532125914546"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.7.1.5 "><p id="zh-cn_topic_0000002270097206_p37781627172014"><a name="zh-cn_topic_0000002270097206_p37781627172014"></a><a name="zh-cn_topic_0000002270097206_p37781627172014"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.7.1.6 "><p id="p16938423262"><a name="p16938423262"></a><a name="p16938423262"></a><span id="zh-cn_topic_0000002270097206_ph77781527182016"><a name="zh-cn_topic_0000002270097206_ph77781527182016"></a><a name="zh-cn_topic_0000002270097206_ph77781527182016"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row178451932101918"><td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.2.7.1.1 "><p id="zh-cn_topic_0000002270097206_p3778172722012"><a name="zh-cn_topic_0000002270097206_p3778172722012"></a><a name="zh-cn_topic_0000002270097206_p3778172722012"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="20.599999999999998%" headers="mcps1.2.7.1.2 "><p id="zh-cn_topic_0000002270097206_p17778727112016"><a name="zh-cn_topic_0000002270097206_p17778727112016"></a><a name="zh-cn_topic_0000002270097206_p17778727112016"></a>fp8_e4m3fn_t/fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.2.7.1.3 "><p id="zh-cn_topic_0000002270097206_p6778627122015"><a name="zh-cn_topic_0000002270097206_p6778627122015"></a><a name="zh-cn_topic_0000002270097206_p6778627122015"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" width="18.85%" headers="mcps1.2.7.1.4 "><p id="zh-cn_topic_0000002270097206_p7532145915540"><a name="zh-cn_topic_0000002270097206_p7532145915540"></a><a name="zh-cn_topic_0000002270097206_p7532145915540"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.72%" headers="mcps1.2.7.1.5 "><p id="zh-cn_topic_0000002270097206_p18778102710208"><a name="zh-cn_topic_0000002270097206_p18778102710208"></a><a name="zh-cn_topic_0000002270097206_p18778102710208"></a>float/half/bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="17.080000000000002%" headers="mcps1.2.7.1.6 "><p id="p1366354422615"><a name="p1366354422615"></a><a name="p1366354422615"></a><span id="zh-cn_topic_0000002270097206_ph1877832711205"><a name="zh-cn_topic_0000002270097206_ph1877832711205"></a><a name="zh-cn_topic_0000002270097206_ph1877832711205"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
</tbody>
</table>

## 实现流程<a name="zh-cn_topic_0000002270097206_section17128015184410"></a>

Host侧自动获取Tiling参数的关键步骤介绍如下：

1.  **创建Tiling对象**。

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling cubeTiling(ascendcPlatform); 
    ```

    传入硬件平台信息创建PlatformAscendC对象，然后创建Tiling对象，硬件平台信息可以通过GetPlatformInfo获取。

2.  **设置A、B、C、Bias的内存逻辑位置、格式、数据类型以及是否转置的信息，设置scaleA、scaleB的内存逻辑位置、格式以及是否转置的信息。**

    调用[SetScaleAType](SetScaleAType.md)、[SetScaleBType](SetScaleBType.md)接口，设置scaleA、scaleB的内存逻辑位置、格式以及是否转置。

    ```
    cubeTiling.SetAType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, false);
    cubeTiling.SetBType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT8_E5M2, true);
    cubeTiling.SetScaleAType(AscendC::TPosition::GM, CubeFormat::ND, false);
    cubeTiling.SetScaleBType(AscendC::TPosition::GM, CubeFormat::ND, true);
    cubeTiling.SetCType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(AscendC::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    ```

3.  **使能MxMatmul场景**。

    调用[SetMadType](SetMadType.md)接口，设置Tiling计算逻辑为MxMatmul场景。

    ```
    cubetiling.SetMadType(MatrixMadType::MXMODE);
    ```

4.  **设置矩阵shape信息。**

    ```
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K); // 设置原始完整的形状M、N、K
    ```

5.  **设置可用空间大小信息。**

    设置Matmul计算时可用的L1 Buffer/L0C Buffer/Unified Buffer空间大小，-1表示AI处理器对应Buffer的大小。

    ```
    cubeTiling.SetBufferSpace(-1, -1, -1);
    ```

6.  **按需设置其他参数，比如设置bias参与计算。**

    ```
    cubeTiling.EnableBias(true);
    ```

7.  **获取Tiling参数。**

    ```
    MatmulCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1){ 
        return ge::GRAPH_FAILED;  
    }
    ```

8.  Tiling参数的序列化保存等其他操作。

Kernel侧的关键步骤介绍如下：

1.  **创建Matmul对象。**

    ```
    // MxMatmul场景通过MatmulTypeWithScale定义A、scaleA、B、scaleB的参数类型信息
    typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, isTransposeA> aType; 
    typedef AscendC::MatmulTypeWithScale<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, fp8_e5m2_t, isTransposeB> bType;
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
    typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 
    // 定义matmul对象时，传入MatmulWithScalePolicy表明使能MxMatmul模板策略
    AscendC::Matmul<aType, bType, cType, biasType, CFG_MDL, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::MatmulWithScalePolicy> mm; 
    ```

    创建对象时需要传入A、scaleA、B、scaleB、C、Bias的参数类型信息， A、scaleA、B、scaleB类型信息通过[MatmulTypeWithScale](#zh-cn_topic_0000002270097206_table14759942142014)来定义，C、Bias类型信息通过[MatmulType](Matmul使用说明.md#table1188045714378)来定义，包括：内存逻辑位置、数据格式、数据类型、转置信息。同时，通过模板参数[MatmulPolicy](MatmulPolicy.md)传入[MatmulWithScalePolicy](MatmulPolicy.md#li19878115120421)表明使能MxMatmul场景。

    ```
    template <TPosition POSITION, TPosition SCALE_POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false, TPosition SRCPOS = TPosition::GM, CubeFormat SCALE_FORMAT = FORMAT, bool SCALE_ISTRANS = ISTRANS, TPosition SCALE_SRCPOS = SRCPOS>
    struct MatmulTypeWithScale: public MatmulType<POSITION, FORMAT, TYPE, ISTRANS> {
        constexpr static TPosition scalePosition = SCALE_POSITION;
        constexpr static CubeFormat scaleFormat = SCALE_FORMAT;
        constexpr static bool isScaleTrans = SCALE_ISTRANS;
        constexpr static TPosition srcScalePos = SCALE_SRCPOS;
    };
    ```

2.  **初始化操作。**

    ```
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化
    ```

3.  **设置左矩阵A、右矩阵B、左量化系数矩阵scaleA、右量化系数矩阵scaleB、Bias。**

    通过[SetTensorScaleA](SetTensorScaleA.md)、[SetTensorScaleB](SetTensorScaleB.md)设置左量化系数矩阵scaleA、右量化系数矩阵scaleB。

    ```
    mm.SetTensorA(gm_a, isTransposeA);    // 设置左矩阵A
    mm.SetTensorB(gm_b, isTransposeB);    // 设置右矩阵B
    mm.SetTensorScaleA(gm_scaleA, isTransposeScaleA);    // 设置左量化系数矩阵scaleA
    mm.SetTensorScaleB(gm_scaleB, isTransposeScaleB);    // 设置右量化系数矩阵scaleB
    mm.SetBias(gm_bias);    // 设置Bias
    ```

4.  **完成矩阵乘操作。**
    -   调用[Iterate](Iterate.md)完成单次迭代计算，叠加while循环完成单核全量数据的计算。Iterate方式，可以自行控制迭代次数，完成所需数据量的计算，方式比较灵活。

        ```
        while (mm.Iterate()) {   
            mm.GetTensorC(gm_c); 
        }
        ```

    -   调用[IterateAll](IterateAll.md)完成单核上所有数据的计算。IterateAll方式，无需循环迭代，使用比较简单。

        ```
        mm.IterateAll(gm_c);
        ```

5.  **结束矩阵乘操作。**

    ```
    mm.End();
    ```

## 参数说明<a name="zh-cn_topic_0000002270097206_section2756107144914"></a>

**表 3**  MatmulTypeWithScale参数说明

<a name="zh-cn_topic_0000002270097206_table14759942142014"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002270097206_row07595429207"><th class="cellrowborder" valign="top" width="18.11%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002270097206_p1818821912211"><a name="zh-cn_topic_0000002270097206_p1818821912211"></a><a name="zh-cn_topic_0000002270097206_p1818821912211"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="81.89%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002270097206_p9188219102113"><a name="zh-cn_topic_0000002270097206_p9188219102113"></a><a name="zh-cn_topic_0000002270097206_p9188219102113"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002270097206_row207591442162020"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p57591442162010"><a name="zh-cn_topic_0000002270097206_p57591442162010"></a><a name="zh-cn_topic_0000002270097206_p57591442162010"></a>POSITION</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p126441432182115"><a name="zh-cn_topic_0000002270097206_p126441432182115"></a><a name="zh-cn_topic_0000002270097206_p126441432182115"></a>左右矩阵的内存逻辑位置。</p>
<p id="zh-cn_topic_0000002270097206_p11644432192117"><a name="zh-cn_topic_0000002270097206_p11644432192117"></a><a name="zh-cn_topic_0000002270097206_p11644432192117"></a>针对<span id="zh-cn_topic_0000002270097206_ph498220912720"><a name="zh-cn_topic_0000002270097206_ph498220912720"></a><a name="zh-cn_topic_0000002270097206_ph498220912720"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p199760149278"><a name="p199760149278"></a><a name="p199760149278"></a>A矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</p>
<p id="p1297671462717"><a name="p1297671462717"></a><a name="p1297671462717"></a>B矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</p>
<p id="zh-cn_topic_0000002270097206_p1267723333010"><a name="zh-cn_topic_0000002270097206_p1267723333010"></a><a name="zh-cn_topic_0000002270097206_p1267723333010"></a>注意：A、B矩阵设置为TPosition::TSCM时，对应的Format仅支持CubeFormat::NZ。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row13470153112518"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p9507104014512"><a name="zh-cn_topic_0000002270097206_p9507104014512"></a><a name="zh-cn_topic_0000002270097206_p9507104014512"></a>SCALE_POSITION</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p3507174020512"><a name="zh-cn_topic_0000002270097206_p3507174020512"></a><a name="zh-cn_topic_0000002270097206_p3507174020512"></a>量化系数矩阵的内存逻辑位置。</p>
<p id="zh-cn_topic_0000002270097206_p145073401055"><a name="zh-cn_topic_0000002270097206_p145073401055"></a><a name="zh-cn_topic_0000002270097206_p145073401055"></a>针对<span id="zh-cn_topic_0000002270097206_ph185821514192716"><a name="zh-cn_topic_0000002270097206_ph185821514192716"></a><a name="zh-cn_topic_0000002270097206_ph185821514192716"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p28081820162716"><a name="p28081820162716"></a><a name="p28081820162716"></a>scaleA矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</p>
<p id="p08081820132714"><a name="p08081820132714"></a><a name="p08081820132714"></a>scaleB矩阵可设置为TPosition::GM，TPosition::VECOUT，TPosition::TSCM</p>
<p id="zh-cn_topic_0000002270097206_p274171193714"><a name="zh-cn_topic_0000002270097206_p274171193714"></a><a name="zh-cn_topic_0000002270097206_p274171193714"></a>注意：scaleA、scaleB矩阵设置为TPosition::TSCM时，对应的<a href="#zh-cn_topic_0000002270097206_p11226185115114">SCALE_FORMAT</a>仅支持CubeFormat::NZ。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row1175904217207"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p5537172115157"><a name="zh-cn_topic_0000002270097206_p5537172115157"></a><a name="zh-cn_topic_0000002270097206_p5537172115157"></a>FORMAT</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p6310416287"><a name="zh-cn_topic_0000002270097206_p6310416287"></a><a name="zh-cn_topic_0000002270097206_p6310416287"></a>数据的物理排布格式，详细介绍请参考<a href="基础知识.md#zh-cn_topic_0000001622194138_section1453415011">数据格式</a>。</p>
<p id="zh-cn_topic_0000002270097206_p1175934213209"><a name="zh-cn_topic_0000002270097206_p1175934213209"></a><a name="zh-cn_topic_0000002270097206_p1175934213209"></a>针对<span id="zh-cn_topic_0000002270097206_ph1712219180275"><a name="zh-cn_topic_0000002270097206_ph1712219180275"></a><a name="zh-cn_topic_0000002270097206_ph1712219180275"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p1638262618270"><a name="p1638262618270"></a><a name="p1638262618270"></a>A矩阵可设置为CubeFormat::ND，CubeFormat::NZ，CubeFormat::VECTOR</p>
<p id="p183823268273"><a name="p183823268273"></a><a name="p183823268273"></a>B矩阵可设置为CubeFormat::ND，CubeFormat::NZ</p>
<p id="zh-cn_topic_0000002270097206_p9185171813367"><a name="zh-cn_topic_0000002270097206_p9185171813367"></a><a name="zh-cn_topic_0000002270097206_p9185171813367"></a>注意：NZ排布格式，A/B的排布格式请参考<a href="基础知识.md#zh-cn_topic_0000001622194138_section1453415011">数据格式</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row167591242142019"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p675984214203"><a name="zh-cn_topic_0000002270097206_p675984214203"></a><a name="zh-cn_topic_0000002270097206_p675984214203"></a>TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="p33861110112812"><a name="p33861110112812"></a><a name="p33861110112812"></a>数据类型。</p>
<p id="zh-cn_topic_0000002270097206_p86091832163315"><a name="zh-cn_topic_0000002270097206_p86091832163315"></a><a name="zh-cn_topic_0000002270097206_p86091832163315"></a>针对<span id="zh-cn_topic_0000002270097206_ph11118321102719"><a name="zh-cn_topic_0000002270097206_ph11118321102719"></a><a name="zh-cn_topic_0000002270097206_ph11118321102719"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p9107163120275"><a name="p9107163120275"></a><a name="p9107163120275"></a>A矩阵可设置为fp4x2_e1m2_t、fp4x2_e2m1_t、fp8_e4m3fn_t、fp8_e5m2_t</p>
<p id="p161081731182712"><a name="p161081731182712"></a><a name="p161081731182712"></a>B矩阵可设置为fp4x2_e1m2_t、fp4x2_e2m1_t、fp8_e4m3fn_t、fp8_e5m2_t</p>
<p id="zh-cn_topic_0000002270097206_p1732612596387"><a name="zh-cn_topic_0000002270097206_p1732612596387"></a><a name="zh-cn_topic_0000002270097206_p1732612596387"></a><strong id="zh-cn_topic_0000002270097206_b8326205913819"><a name="zh-cn_topic_0000002270097206_b8326205913819"></a><a name="zh-cn_topic_0000002270097206_b8326205913819"></a>注意：具体数据类型组合关系请参考</strong><a href="#zh-cn_topic_0000002270097206_table5383144710452">MxMatmul支持数据类型</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row67591429205"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p1475904232016"><a name="zh-cn_topic_0000002270097206_p1475904232016"></a><a name="zh-cn_topic_0000002270097206_p1475904232016"></a>ISTRANS</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p15691340143420"><a name="zh-cn_topic_0000002270097206_p15691340143420"></a><a name="zh-cn_topic_0000002270097206_p15691340143420"></a>是否开启使能A、B矩阵转置的功能。默认值为false。参数支持的取值如下：</p>
<p id="p11558102715289"><a name="p11558102715289"></a><a name="p11558102715289"></a>true：开启使能矩阵转置的功能，开启后，分别通过<a href="zh-cn_topic_0000002554423665.html">SetTensorA</a>和<a href="zh-cn_topic_0000002554343499.html">SetTensorB</a>中的isTransposeA、isTransposeB参数设置A、B矩阵是否转置。若设置A、B矩阵转置，Matmul会认为A矩阵形状为[K, M]，B矩阵形状为[N, K]。</p>
<p id="p45581527162811"><a name="p45581527162811"></a><a name="p45581527162811"></a>false：不开启使能矩阵转置的功能，通过<a href="zh-cn_topic_0000002554423665.html">SetTensorA</a>和<a href="zh-cn_topic_0000002554343499.html">SetTensorB</a>不能设置A、B矩阵的转置情况。Matmul会认为A矩阵形状为[M, K]，B矩阵形状为[K, N]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row68812038163520"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p54471329153612"><a name="zh-cn_topic_0000002270097206_p54471329153612"></a><a name="zh-cn_topic_0000002270097206_p54471329153612"></a>SRCPOS</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p204471629153618"><a name="zh-cn_topic_0000002270097206_p204471629153618"></a><a name="zh-cn_topic_0000002270097206_p204471629153618"></a>A/B矩阵的POSITION参数配置为TPosition::TSCM时，要设置TSCM中矩阵数据的来源的内存逻辑位置，默认为TPosition::GM。</p>
<p id="zh-cn_topic_0000002270097206_p2231172914619"><a name="zh-cn_topic_0000002270097206_p2231172914619"></a><a name="zh-cn_topic_0000002270097206_p2231172914619"></a>针对<span id="zh-cn_topic_0000002270097206_ph1231182974620"><a name="zh-cn_topic_0000002270097206_ph1231182974620"></a><a name="zh-cn_topic_0000002270097206_ph1231182974620"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p13320143342815"><a name="p13320143342815"></a><a name="p13320143342815"></a>A矩阵可设置为TPosition::GM，TPosition::VECOUT</p>
<p id="p12320163310288"><a name="p12320163310288"></a><a name="p12320163310288"></a>B矩阵可设置为TPosition::GM，TPosition::VECOUT</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row7921164017113"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p11226185115114"><a name="zh-cn_topic_0000002270097206_p11226185115114"></a><a name="zh-cn_topic_0000002270097206_p11226185115114"></a>SCALE_FORMAT</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p78145661612"><a name="zh-cn_topic_0000002270097206_p78145661612"></a><a name="zh-cn_topic_0000002270097206_p78145661612"></a>量化系数矩阵的物理排布格式，详细介绍请参考<a href="#zh-cn_topic_0000002270097206_fig107863142019">数据格式</a>。默认值为<a href="#zh-cn_topic_0000002270097206_p5537172115157">FORMAT</a>。</p>
<p id="zh-cn_topic_0000002270097206_p158141465164"><a name="zh-cn_topic_0000002270097206_p158141465164"></a><a name="zh-cn_topic_0000002270097206_p158141465164"></a>针对<span id="zh-cn_topic_0000002270097206_ph695882732713"><a name="zh-cn_topic_0000002270097206_ph695882732713"></a><a name="zh-cn_topic_0000002270097206_ph695882732713"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p15845121173011"><a name="p15845121173011"></a><a name="p15845121173011"></a>scaleA矩阵可设置为CubeFormat::ND，CubeFormat::NZ，CubeFormat::VECTOR</p>
<p id="p884571116308"><a name="p884571116308"></a><a name="p884571116308"></a>scaleB矩阵可设置为CubeFormat::ND，CubeFormat::NZ</p>
<p id="zh-cn_topic_0000002270097206_p7332111273413"><a name="zh-cn_topic_0000002270097206_p7332111273413"></a><a name="zh-cn_topic_0000002270097206_p7332111273413"></a>注意：</p>
<p id="p15352181633019"><a name="p15352181633019"></a><a name="p15352181633019"></a>NZ排布格式请参考<a href="数据排布格式.md#li19960204116136">NZ</a>。MxMatmul场景，scaleA、scaleB的数据类型为fp8_e8m0_t，分形大小H0=16，W0=2。</p>
<p id="p43523162303"><a name="p43523162303"></a><a name="p43523162303"></a>在Scale矩阵为ND格式的场景中，当通过<a href="SetTensorScaleA.md">SetTensorScaleA</a>接口设置scaleA矩阵转置时，scaleA内存排布格式必须按照(K/64, M，2)排布，通过<a href="SetTensorScaleB.md">SetTensorScaleB</a>接口设置scaleB矩阵不转置时，scaleB内存排布格式必须按照(K/64，N， 2)排布，详细介绍请参考<a href="#zh-cn_topic_0000002270097206_fig76682054103416">数据格式</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row0325310101910"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p19617394191"><a name="zh-cn_topic_0000002270097206_p19617394191"></a><a name="zh-cn_topic_0000002270097206_p19617394191"></a>SCALE_ISTRANS</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p22682252237"><a name="zh-cn_topic_0000002270097206_p22682252237"></a><a name="zh-cn_topic_0000002270097206_p22682252237"></a>是否开启使能scaleA、scaleB矩阵转置的功能。默认值为<a href="#zh-cn_topic_0000002270097206_p1475904232016">ISTRANS</a>。参数支持的取值如下：</p>
<p id="p1407842193011"><a name="p1407842193011"></a><a name="p1407842193011"></a>true：开启使能矩阵转置的功能。开启后，分别通过<a href="SetTensorScaleA.md">SetTensorScaleA</a>和<a href="SetTensorScaleB.md">SetTensorScaleB</a>中的isTransposeScaleA、isTransposeScaleB参数设置scaleA、scaleB矩阵是否转置。在Scale矩阵为ND格式的场景中，若设置scaleA、scaleB矩阵转置，Matmul会认为scaleA矩阵形状为[Ceil(K/64), M, 2]，scaleB矩阵形状为[N, Ceil(K/64), 2]。</p>
<p id="p19408124217308"><a name="p19408124217308"></a><a name="p19408124217308"></a>false：不开启使能矩阵转置的功能。通过<a href="SetTensorScaleA.md">SetTensorScaleA</a>和<a href="SetTensorScaleB.md">SetTensorScaleB</a>不能设置scaleA、scaleB矩阵的转置情况。Matmul会认为scaleA矩阵形状为[M, Ceil(K/64), 2]，scaleB矩阵形状为[Ceil(K/64), N, 2]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002270097206_row11254846161912"><td class="cellrowborder" valign="top" width="18.11%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002270097206_p6158144912199"><a name="zh-cn_topic_0000002270097206_p6158144912199"></a><a name="zh-cn_topic_0000002270097206_p6158144912199"></a>SCALE_SRCPOS</p>
</td>
<td class="cellrowborder" valign="top" width="81.89%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002270097206_p642071011551"><a name="zh-cn_topic_0000002270097206_p642071011551"></a><a name="zh-cn_topic_0000002270097206_p642071011551"></a>scaleA、scaleB矩阵的<a href="#zh-cn_topic_0000002270097206_p9507104014512">SCALE_POSITION</a>参数设置为TPosition::TSCM时，需要通过本参数设置TSCM中矩阵数据来源的内存逻辑位置，默认值为<a href="#zh-cn_topic_0000002270097206_p54471329153612">SRCPOS</a>。</p>
<p id="zh-cn_topic_0000002270097206_p134201910115519"><a name="zh-cn_topic_0000002270097206_p134201910115519"></a><a name="zh-cn_topic_0000002270097206_p134201910115519"></a>针对<span id="zh-cn_topic_0000002270097206_ph2420111019559"><a name="zh-cn_topic_0000002270097206_ph2420111019559"></a><a name="zh-cn_topic_0000002270097206_ph2420111019559"></a>Ascend 950PR/Ascend 950DT</span>：</p>
<p id="p138204923010"><a name="p138204923010"></a><a name="p138204923010"></a>scaleA矩阵可设置为TPosition::GM，TPosition::VECOUT</p>
<p id="p68316492306"><a name="p68316492306"></a><a name="p68316492306"></a>scaleB矩阵可设置为TPosition::GM，TPosition::VECOUT</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000002270097206_section14160134220363"></a>

-   MxMatmul场景仅支持[Norm模板](MatmulConfig.md)和[MDL模板](MatmulConfig.md)。

-   在MxMatmul场景中，如果A与B矩阵的位置同时为GM，对singleKIn没有特殊限制，在这种情况下，若scaleA和scaleB的K方向大小（即Ceil\(singleKIn, 32\)）为奇数，用户需自行在scaleA和scaleB的K方向补0至偶数。例如，当singleKIn为30时，Ceil\(singleKIn, 32\)为1，用户需要自行在scaleA和scaleB的K方向补0，使K方向为偶数。对于其它A、B矩阵逻辑位置的组合情况，即A与B矩阵的位置不同时为GM，singleKIn以32个元素向上对齐后的数值必须是32的偶数倍。
-   在MxMatmul场景中，当输入数据类型为fp4x2\_e2m1\_t/fp4x2\_e1m2\_t时，内轴必须为偶数。
-   在MxMatmul场景中，通过将A矩阵和scaleA矩阵的数据格式设置为VECTOR，来开启[GEMV模式](矩阵向量乘.md)。在此模式下，A和scaleA矩阵仅支持内存逻辑位置为GM，并且均不支持转置。
-   A矩阵、B矩阵为UB输入时，矩阵的内轴需要向上32字节对齐，例如，A矩阵的形状为\(M, K\)时，将K对齐到32字节；A矩阵的形状为\(K, M\)时，将M对齐到32字节。
-   scaleA矩阵、scaleB矩阵为UB输入时，矩阵的内轴需要向上32字节对齐，例如，scaleA矩阵的形状为\(M, K/32\)时，将K/32对齐到32字节；scaleA矩阵的形状为\(K/32, M\)时，将M对齐到32字节。
-   当scaleA和scaleB矩阵以ND格式输入时，高阶API在内部实现格式转换时，需要占用UB临时空间。开发者需使用[SetLocalWorkspace](SetLocalWorkspace.md)接口配置临时空间，临时空间大小（单位字节）的计算公式如下。

    ```
    int32_t scaleATmpBuf = 0;
    int32_t scaleBTmpBuf = 0;
    if constexpr (A_TYPE::scalePosition == TPosition::VECOUT) {
        if (A_TYPE::isScaleTrans) {
            scaleATmpBuf = CeilAlign(SingleCoreM, 32) * scaleK;
        } else {
            scaleATmpBuf = CeilAlign(scaleK, 32) * SingleCoreM;
        }
    }
    if constexpr (B_TYPE::scalePosition == TPosition::VECOUT) {
        if (B_TYPE::isScaleTrans) {
            scaleBTmpBuf = SingleCoreN * CeilAlign(scaleK, 32);
        } else {
            scaleBTmpBuf = scaleK * CeilAlign(SingleCoreN, 32);
        }
    }
    int32_t totalTmpBuf = scaleATmpBuf + scaleBTmpBuf;
    ```

