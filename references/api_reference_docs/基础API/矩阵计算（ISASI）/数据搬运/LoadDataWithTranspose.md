# LoadDataWithTranspose

**页面ID:** atlasascendc_api_07_0239  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0239.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

该接口实现带转置的2D格式数据从A1/B1到A2/B2的加载。

下面通过示例来讲解接口功能和关键参数：下文图中一个N形或者一个Z形代表一个分形。

- 对于uint8_t/int8_t数据类型，每次迭代处理32*32*1B数据，可处理2个分形（一个分形512B），每次迭代中，源操作数中2个连续的16*32分形将被合并为1个32*32的方块矩阵，基于方块矩阵做转置，转置后分裂为2个16*32分形，根据目的操作数分形间隔等参数可以有不同的排布。如下图示例：

  - 共需要处理3072B的数据，每次迭代处理32*32*1B数据，需要3次迭代可以完成，repeatTime = 3；
  - srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：32*32*1B），这里的单位实际上是拼接后的方块矩阵的大小；
  - dstGap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
  - dstFracGap = 0，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为0（单位：512B）。

<!-- img2text -->
```text
                    dstFracGap = 0
                          ↕
                         ─┬─
                          │                      dstGap = 1
                          │                 <────────────────>
                          │
        <────────────────>
         srcStride = 1

src                                                   dst
┌──────────────────────┐                     ┌────────┬────────┐
│╲──────────────→      │                     │  ╱   ↓ │  ╱   ↓ │
│  ╲────────────→      │                     │ ╱      │ ╱      │
│──── ─ ─ ─ ─ ─        │                     ├────────┼────────┤
│╲──────────────→      │                     │  ╱   ↓ │  ╱   ↓ │
│  ╲────────────→      │                     │ ╱      │ ╱      │
├──────────────────────┤                     ├────────┼────────┤
│╲──────────────→      │                     │  ╱   ↓ │  ╱   ↓ │
│  ╲────────────→      │                     │ ╱      │ ╱      │
│──── ─ ─ ─ ─ ─        │                     ├────────┼────────┤
│╲──────────────→      │                     │  ╱   ↓ │  ╱   ↓ │
│  ╲────────────→      │                     │ ╱      │ ╱      │
├──────────────────────┤                     ├────────┼────────┤
│╲──────────────→      │                     │  ╱   ↓ │  ╱   ↓ │
│  ╲────────────→      │                     │ ╱      │ ╱      │
│──── ─ ─ ─ ─ ─        │                     ├────────┼────────┤
│╲──────────────→      │                     │  ╱   ↓ │  ╱   ↓ │
│  ╲────────────→      │                     │ ╱      │ ╱      │
└──────────────────────┘                     └────────┴────────┘
```

说明:
- 左侧 `src` 为 3 个迭代数据块纵向排列，`srcStride = 1` 标注的是相邻迭代间源操作数起始地址间隔。
- 右侧 `dst` 为 3 行 × 2 列分形排列。
- `dstFracGap = 0` 表示每个迭代内左右两个分形之间无间隔。
- `dstGap = 1` 表示相邻迭代间，前一个迭代第一个分形结束地址到下一个迭代第一个分形起始地址的间隔为 1。
- 图中箭头表示从 `src` 中的数据搬运/重排到 `dst` 各分形中的方向关系。

如下图示例：

  - repeatTime和srcStride的解释和上图示例一致。
  - dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
  - dstFracGap = 2，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为2（单位：512B）。

<!-- img2text -->
```text
src
┌────────────────────────────┐
│            ╱╲      →       │
├────────────────────────────┤
│      ┈┈┈  ╱╲        →      │
├────────────────────────────┤
│            ╱╲      →       │
├────────────────────────────┤
│      ┈┈┈  ╱╲        →      │
├────────────────────────────┤
│            ╱╲      →       │
├────────────────────────────┤
│      ┈┈┈  ╱╲        →      │
└────────────────────────────┘
↑                            ↑
│                            │
└──────── srcStride = 1 ─────┘


dst
┌──────────────┬──────────────┐
│      ╱│      │      ╱│      │
│     ╱ │      │     ╱ │      │
│    ╱  │   ↓  │    ╱  │   ↓  │
├──────────────┼──────────────┤
│      ╱│      │      ╱│      │
│     ╱ │      │     ╱ │      │
│    ╱  │   ↓  │    ╱  │   ↓  │
├──────────────┼──────────────┤
│    ┈┈  ╱│    │    ╱  │      │
│      ╱ │     │   ╱   │      │
│     ╱  │  ↓  │  ╱    │  ↓   │
└──────────────┴──────────────┘
↑
│ dstGap = 0
↓
↑
│
└──────────── dstFracGap = 2 ────────────┘
```

- 对于half/bfloat16_t数据类型，每次迭代处理16*16*2B数据，可处理1个分形（一个分形512B），每次迭代中，源操作数中1个16*16分形将被转置。

  - 共需要处理1536B的数据，每次迭代处理16*16*2B数据，需要3次迭代可以完成，repeatTime = 3；
  - srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1 （单位：16*16*2B）；
  - dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址无间隔；
  - 该场景下，因为其分形即为方块矩阵，每个迭代处理一个分形，不存在迭代内分形的间隔，该参数设置无效。

<!-- img2text -->
```text
                 ↑
                 │
      srcStride = 1
                 │
                 ↓

src                                        dst
┌──────────────────────┐          ┌──────────────────────┐
│                      │          │                      │
│  ────────────────╲   │          │  │              ╱│   │
│                ╱     │          │  │            ╱  │   │
│              ╱       │          │  │          ╱    │   │
│            ╱         │          │  │        ╱      │   │
│  ╲────────────────→  │          │  │      ╱        │   │
├──────────────────────┤  dstGap   ├──────────────────────┤
│                      │    = 0    │                      │
│   ─ ─ ─ ─ ─ ─ ╲      │          │  │   ╱ ─ ─ ─ ─ ─    │
│              ╱       │          │  │ ╱                │
│            ╱         │          │  │╱                 │
│          ╱           │          │  │                  │
│  ╲────────────────→  │          │  │                ↓ │
├──────────────────────┤          ├──────────────────────┤
│                      │          │                      │
│   ─ ─ ─ ─ ─ ─ ╲      │          │  │   ╱ ─ ─ ─ ─ ─    │
│              ╱       │          │  │ ╱                │
│            ╱         │          │  │╱                 │
│          ╱           │          │  │                  │
│  ╲────────────────→  │          │  │                ↓ │
└──────────────────────┘          └──────────────────────┘
```

- 对于float/int32_t/uint32_t数据类型，每次迭代处理16*16*4B数据，可处理2个分形（一个分形512B），每次迭代中，源操作数2个连续的16*8分形将被合并为1个16*16的方块矩阵，基于方块矩阵做转置，转置后分裂为2个16*8分形，根据目的操作数分形间隔等参数可以有不同的排布。如下图示例：

  - 共需要处理3072B的数据，每次迭代处理16*16*4B数据，需要3次迭代可以完成，repeatTime = 3；
  - srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：16*16*4B），这里的单位实际上是拼接后的方块矩阵的大小；
  - dstGap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
  - dstFracGap = 0，表示每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为0（单位：512B）。

<!-- img2text -->
```text
                     <────────────────────────>
                           srcStride = 1

src
┌──────────────────┬──────────────────┬──────────────────┐
│ ┌──────┐ ┌──────┐ │ ┌──────┐ ┌──────┐ │ ┌──────┐ ┌──────┐ │
│ │      ╲│ │      ╲│ │      ╲│ │      ╲│ │      ╲│ │      ╲│ │
│ │       ╲ │       ╲ │       ╲ │       ╲ │       ╲ │       ╲ │
│ │       │↓│       │↓│       │↓│       │↓│       │↓│       │↓│
│ │      ╱ │ │      ╱ │ │      ╱ │ │      ╱ │ │      ╱ │ │      ╱ │
│ └─────→┘ └─────→┘ │ └─────→┘ └─────→┘ │ └─────→┘ └─────→┘ │
└──────────────────┴──────────────────┴──────────────────┘

dst
┌──────────────────┬──────────────────┬──────────────────┐
│ ┌────────────→   │ ┌────────────→   │ ┌────────────→   │
│ │                │ │                │ │                │
│ │                │ │                │ │                │
│ └────────────────│ └────────────────│ └────────────────│
│        ↓         │        ↓         │        ↓         │
├──────────────────┼──────────────────┼──────────────────┤
│ ┌────────────→   │ ┌────────────→   │ ┌────────────→   │
│ │                │ │                │ │                │
│ │                │ │                │ │                │
│ └────────────────│ └────────────────│ └────────────────│
│        ↓         │        ↓         │        ↓         │
└──────────────────┴──────────────────┴──────────────────┘
↑
│ dstFracGap = 0
↓
↕
dstGap = 1
```

如下图示例：

  - repeatTime和srcStride的解释和上图示例一致。
  - dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
  - dstFracGap = 2，表示每个迭代内目的操作数前一个分形结束地址与后一个分形起始地址的间隔为2（单位：512B）。

<!-- img2text -->
``` 
srcStride = 1
<────────────────────────────────────────────────────────────────────>

src
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│              │              │              │              │              │              │
│      ╲       │      ╲       │      ╲       │      ╲       │      ╲       │      ╲       │
│       ╲      │       ╲      │       ╲      │       ╲      │       ╲      │       ╲      │
│       ╱      │       ╱      │       ╱      │       ╱      │       ╱      │       ╱      │
│      ╱  ↓    │      ╱  ↓    │      ╱  ↓    │      ╱  ↓    │      ╱  ↓    │      ╱  ↓    │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘


dstGap = 0
<────────────────────────>

dstFracGap = 2
                        <────────────────────────────────────────────>

dst
┌───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ ╲                           ↓ │ ╲                           ↓ │ ╲                           ↓ │
│  ╲                           │ │  ╲                           │ │  ╲                           │ │
│   ╲                          │ │   ╲                          │ │   ╲                          │ │
│    ╲                         │ │    ╲                         │ │    ╲                         │ │
├───────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
│ ╲                           ↓ │ ╲                           ↓ │ ╲                           ↓ │
│  ╲                           │ │  ╲                           │ │  ╲                           │ │
│   ╲                          │ │   ╲                          │ │   ╲                          │ │
│    ╲                         │ │    ╲                         │ │    ╲                         │ │
└───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

说明:
- src 上方标注为 `srcStride = 1`
- dst 上方左侧标注为 `dstGap = 0`
- dst 上方右侧标注为 `dstFracGap = 2`
- src 区域由 6 个连续块组成，前 4 个块为一组连续处理分形，后 2 个块继续按相同规则示意
- dst 区域为 2 行 × 3 列布局，共 6 个分形位置
- 图中斜线与向下箭头表示 int4b_t 数据拼成 int8_t/uint8_t 后的排布/写入方向示意
- 颜色仅用于区分分组，框图中未体现颜色信息

- 对于int4b_t数据类型，每次迭代处理64*64*0.5B数据，可处理4个分形（一个分形512B），每次迭代中，源操作数中4个连续的16*64分形将被合并为1个64*64的方块矩阵，基于方块矩阵做转置，转置后分裂为4个16*64分形，根据目的操作数分形间隔等参数可以有不同的排布。

int4b_t数据类型需要两个数拼成一个int8_t或uint8_t的数，拼凑的规则如下：

<!-- img2text -->
```text
┌───────────────────────┬───────────────────────┬───────────────────────┐
│         1Byte         │         1Byte         │         1Byte         │
├───────────┬───────────┼───────────┬───────────┼───────────┬───────────┤
│     1     │     0     │     3     │     2     │     5     │     4     │
└───────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

如下图示例：

  - 共需要处理6144B的数据，每次迭代处理64*64*0.5B数据，需要3次迭代可以完成，repeatTime = 3；
  - srcStride = 1，表示相邻迭代间，源操作数前一个方块矩阵与后一个方块矩阵起始地址的间隔为1（单位：64*64*0.5B），这里的单位实际上是拼接后的方块矩阵的大小；
  - dstGap = 1，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔为1（单位：512B）；
  - dstFracGap = 0，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为0（单位：512B）。

<!-- img2text -->
```text
                          dstFracGap = 0
                               ↓
srcStride = 1        ┌────────────────────┐                    ◄────────►
      ↑              │                    │                      dstGap = 1
      │              │                    │
      │              │────────────────────│                ┌────────────────────┐
      │              │                    │                │         │          │
      │              │                    │                │   ╱     │    ╱     │
      │              │                    │                │  ╱      │   ╱      │
      │              │                    │                │ ╱       │  ╱       │
      │              │                    │                │ ╲       ↓  ╲       │
      │              │────────────────────│                │  ╲      │   ╲      ↓
      │              │                    │                │   ╲     │    ╲     │
      │              │                    │                │         │          │
      ↓              │                    │                │────────────────────│
                     │────────────────────│                │         │          │
                     │                    │                │   ╱     │    ╱     │
                     │                    │                │  ╱      │   ╱      │
                     │                    │                │ ╱       │  ╱       │
                     │                    │                │ ╲       ↓  ╲       │
                     │────────────────────│                │  ╲      │   ╲      ↓
                     │                    │                │   ╲     │    ╲     │
                     │                    │                │         │          │
                     │                    │                │────────────────────│
                     │────────────────────│                │         │          │
                     │                    │                │   ╱     │    ╱     │
                     │                    │                │  ╱      │   ╱      │
                     │                    │                │ ╱       │  ╱       │
                     │                    │                │ ╲       ↓  ╲       │
                     │                    │                │  ╲      │   ╲      ↓
                     │                    │                │   ╲     │    ╲     │
                     └────────────────────┘                └────────────────────┘
                              src                                      dst
```

如下图示例：

  - repeatTime和srcStride的解释和上图示例一致。
  - dstGap = 0，表示相邻迭代间，目的操作数前一个迭代第一个分形的结束地址和下一个迭代第一个分形起始地址无间隔。
  - dstFracGap = 2，表示每个迭代内目的操作数前一个分形的结束地址与后一个分形起始地址的间隔为2（单位：512B）。

<!-- img2text -->
```text
src                                                dst
                                                   
srcStride = 1                                      dstGap = 0
↑                                                  ↑ ↓
│                                                  │
│      ┌────────────────────┐                      │      ┌────────┬────────┐
│      │                    │                      │      │        │        │
│      │  /────────────→    │                      │      │   /│   │   /│   │
│      │    ──────────→     │                      │      │  / │   │  / │   │
│      │  /────────────→    │                      │      │ ↓  │   │ ↓  │   │
│      │                    │                      │      ├────────┼────────┤
│      ├────────────────────┤                      │      │        │        │
│      │                    │                      │      │   /│   │   /│   │
│      │  /────────────→    │                      │      │  / │   │  / │   │
│      │    ──────────→     │                      │      │ ↓  │   │ ↓  │   │
│      │  /────────────→    │                      │      ├────────┼────────┤
│      │                    │                      │      │        │        │
│      ├────────────────────┤                      │      │   /│   │   /│   │
│      │                    │                      │      │  / │   │  / │   │
│      │  /────────────→    │                      │      │ ↓  │   │ ↓  │   │
│      │    ──────────→     │                      │      └────────┴────────┘
│      │  /────────────→    │                      │
│      │                    │                      │
↓      └────────────────────┘                      ↓

                                                   dstFracGap = 2
                                                   ↑
                                                   │
                                                   │
                                                   ↓

src                                                dst
```

说明:
- 左侧为 `src`，右侧为 `dst`
- `srcStride = 1` 标注在 `src` 整体高度侧边
- `dstGap = 0` 标注在 `dst` 顶部两列分形之间
- `dstFracGap = 2` 标注在 `dst` 左侧，表示迭代内上下分形之间的间隔
- `dst` 由 `2` 列 × `3` 行分块组成
- 图中斜线箭头表示各分形内的数据访问/搬运方向
- `src` 中部有一块竖向高亮区域，`dst` 中部右侧分块内也有对应高亮区域

#### 函数原型

```
template <typename T>
__aicore__ inline void LoadDataWithTranspose(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2dTransposeParams& loadDataParams)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的数据类型为：int4b_t/int8_t/uint8_t/half/bfloat16_t/float/int32_t/uint32_t。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的数据类型为：int4b_t/int8_t/uint8_t/half/bfloat16_t/float/int32_t/uint32_t。 Atlas 200I/500 A2 推理产品，支持的数据类型为：int4b_t/uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float。 其中int4b_t数据类型仅在LocalTensor的TPosition为B2时支持。 |

**表2 **参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| dst | 输出 | 目的操作数，结果矩阵，类型为LocalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的TPosition为A2/B2。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的TPosition为A2/B2。 Atlas 200I/500 A2 推理产品，支持的TPosition为A2/B2。 LocalTensor的起始地址需要保证512字节对齐。 数据类型和src的数据类型保持一致。 |
| src | 输入 | 源操作数，类型为LocalTensor。 Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的TPosition为A1/B1。 Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的TPosition为A1/B1。 Atlas 200I/500 A2 推理产品，支持的TPosition为A1/B1。 LocalTensor的起始地址需要保证32字节对齐。 数据类型和dst的数据类型保持一致。 |
| loadDataParams | 输入 | LoadDataWithTranspose相关参数，类型为LoadData2dTransposeParams。 具体定义请参考${INSTALL_DIR}/include/ascendc/basic_api/interface/kernel_struct_mm.h，${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。 参数说明请参考表3。 |

**表3 **LoadData2dTransposeParams结构体内参数说明

| 参数名称 | 输入/输出 | 含义 |
| --- | --- | --- |
| startIndex | 输入 | 方块矩阵ID，搬运起始位置为源操作数中第几个方块矩阵（0 为源操作数中第1个方块矩阵）。取值范围：startIndex∈[0, 65535] 。默认为0。 例如，源操作数中有20个大小为16*8*4B的分形（数据类型为float），startIndex=1表示搬运起始位置为第2个方块矩阵，即将第3和第4个分形从源操作数中转置到目的操作数中（第1、2个分形组成第1个方块矩阵，第3、4个分形组成第2个方块矩阵）。 |
| repeatTimes | 输入 | 迭代次数。 对于uint8_t/int8_t数据类型，每次迭代处理32*32*1B数据； 对于half/bfloat16_t数据类型，每次迭代处理16*16*2B数据； 对于float/int32_t/uint32_t数据类型，每次迭代处理16*16*4B数据。 对于int4b_t数据类型，每次迭代处理16*64*0.5B数据。 取值范围：repeatTimes∈[0, 255]。默认为0。 |
| srcStride | 输入 | 相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔。这里的单位实际上是拼接后的方块矩阵的大小。 对于uint8_t/int8_t数据类型，单位是32*32*1B； 对于half/bfloat16_t数据类型，单位是16*16*2B； 对于float/int32_t/uint32_t数据类型，单位是16*16*4B。 对于int4b_t数据类型，每次迭代处理16*64*0.5B数据。 取值范围：srcStride∈[0, 65535]。默认为0。 |
| dstGap | 输入 | 相邻迭代间，目的操作数前一个迭代第一个分形的结束地址到下一个迭代第一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。 |
| dstFracGap | 输入 | 每个迭代内目的操作数转置前一个分形结束地址与后一个分形起始地址的间隔，单位为512B，仅在数据类型为float/int32_t/uint32_t/uint8_t/int8_t/int4b_t时有效。取值范围：dstFracGap∈[0, 65535]。默认为0。 |
| addrMode | 输入 | 预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。 |

#### 约束说明

- repeat=0表示不执行搬运操作。
- 开发者需要保证目的操作数转置后的分形没有重叠。

#### 调用示例

- 示例1：该示例输入a矩阵为int8_t类型，shape为[16,32]，输入b矩阵为int8_t类型，shape为[32,64]，输出c的类型为int32_t。a矩阵从A1->A2不转置，b矩阵从B1->B2转置，之后进行Mmad计算和Fixpipe计算。

```
#include "kernel_operator.h"

template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelMatmul {
public:
    __aicore__ inline KernelMatmul()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        nBlocks = n / 16;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ fmap_T *)a);
        bGM.SetGlobalBuffer((__gm__ weight_T *)b);
        cGM.SetGlobalBuffer((__gm__ dstCO1_T *)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(fmap_T));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(fmap_T));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(weight_T));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(weight_T));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.AllocTensor<fmap_T>();
        AscendC::LocalTensor<weight_T> b1Local = inQueueB1.AllocTensor<weight_T>();

        AscendC::Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);

        AscendC::Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.DeQue<fmap_T>();
        AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.AllocTensor<fmap_T>();

        AscendC::LoadData2DParams loadL0AParams;
        loadL0AParams.repeatTimes = aSize * sizeof(fmap_T) / 512;
        loadL0AParams.srcStride = 1;
        loadL0AParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadL0AParams);

        inQueueA2.EnQue<fmap_T>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<weight_T> b1Local = inQueueB1.DeQue<weight_T>();
        AscendC::LocalTensor<weight_T> b2Local = inQueueB2.AllocTensor<weight_T>();

        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        nBlockSize = 32;
        loadDataParams.repeatTimes = n / nBlockSize;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 1;
        loadDataParams.dstFracGap = 0;
        AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<weight_T>(b2Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.DeQue<fmap_T>();
        AscendC::LocalTensor<weight_T> b2Local = inQueueB2.DeQue<weight_T>();
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<dstCO1_T>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    // dst queue
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<fmap_T> aGM;
    AscendC::GlobalTensor<weight_T> bGM;
    AscendC::GlobalTensor<dst_T> cGM;

    uint16_t m = 16, k = 32, n = 64;
    uint8_t nBlockSize = 16;
    uint16_t c0Size = 16;
    uint16_t aSize, bSize, cSize, nBlocks;
};

extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator_int8_t(__gm__ uint8_t *a, __gm__ uint8_t *b,
    __gm__ uint8_t *c)
{
    KernelMatmul<dst_type, fmap_type, weight_type, dstCO1_type> op;
    op.Init(a, b, c);
    op.Process();
}
```

- 示例2：该示例输入a矩阵为half类型，shape为[16,32]，输入b矩阵为half类型，shape为[32,32]，输出c的类型为float。a矩阵从A1->A2不转置，b矩阵从B1->B2转置，之后进行Mmad计算和Fixpipe计算。

```
#include "kernel_operator.h"

template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelMatmul {
public:
    __aicore__ inline KernelMatmul()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        nBlocks = n / 16;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ fmap_T *)a);
        bGM.SetGlobalBuffer((__gm__ weight_T *)b);
        cGM.SetGlobalBuffer((__gm__ dstCO1_T *)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(fmap_T));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(fmap_T));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(weight_T));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(weight_T));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.AllocTensor<fmap_T>();
        AscendC::LocalTensor<weight_T> b1Local = inQueueB1.AllocTensor<weight_T>();

        AscendC::Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);

        AscendC::Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.DeQue<fmap_T>();
        AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.AllocTensor<fmap_T>();

        AscendC::LoadData2DParams loadL0AParams;
        loadL0AParams.repeatTimes = aSize * sizeof(fmap_T) / 512;
        loadL0AParams.srcStride = 1;
        loadL0AParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadL0AParams);

        inQueueA2.EnQue<fmap_T>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<weight_T> b1Local = inQueueB1.DeQue<weight_T>();
        AscendC::LocalTensor<weight_T> b2Local = inQueueB2.AllocTensor<weight_T>();

        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        nBlockSize = 16;
        loadDataParams.repeatTimes = k / nBlockSize;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 1;
        for (int i = 0; i < (n / nBlockSize); ++i) {
            AscendC::LoadDataWithTranspose(b2Local[i * 16 * nBlockSize], b1Local[i * k * nBlockSize], loadDataParams);
        }

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<weight_T>(b2Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.DeQue<fmap_T>();
        AscendC::LocalTensor<weight_T> b2Local = inQueueB2.DeQue<weight_T>();
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<dstCO1_T>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    // dst queue
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<fmap_T> aGM;
    AscendC::GlobalTensor<weight_T> bGM;
    AscendC::GlobalTensor<dst_T> cGM;

    uint16_t m = 16, k = 32, n = 32;
    uint8_t nBlockSize = 16;
    uint16_t c0Size = 16;
    uint16_t aSize, bSize, cSize, nBlocks;
};

extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator_half(__gm__ uint8_t *a, __gm__ uint8_t *b,
    __gm__ uint8_t *c)
{
    KernelMatmul<dst_type, fmap_type, weight_type, dstCO1_type> op;
    op.Init(a, b, c);
    op.Process();
}
```

- 示例3：该示例输入a矩阵为float类型，shape为[16,16]，输入b矩阵为float类型，shape为[16,32]，输出c的类型为float。a矩阵从A1->A2不转置，b矩阵从B1->B2转置，之后进行Mmad计算和Fixpipe计算。

```
#include "kernel_operator.h"

template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelMatmul {
public:
    __aicore__ inline KernelMatmul()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
        nBlocks = n / 16;
    }
    __aicore__ inline void Init(__gm__ uint8_t *a, __gm__ uint8_t *b, __gm__ uint8_t *c)
    {
        aGM.SetGlobalBuffer((__gm__ fmap_T *)a);
        bGM.SetGlobalBuffer((__gm__ weight_T *)b);
        cGM.SetGlobalBuffer((__gm__ dstCO1_T *)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(fmap_T));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(fmap_T));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(weight_T));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(weight_T));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(dstCO1_T));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.AllocTensor<fmap_T>();
        AscendC::LocalTensor<weight_T> b1Local = inQueueB1.AllocTensor<weight_T>();

        AscendC::Nd2NzParams dataCopyA1Params;
        dataCopyA1Params.ndNum = 1;
        dataCopyA1Params.nValue = m;
        dataCopyA1Params.dValue = k;
        dataCopyA1Params.srcNdMatrixStride = 0;
        dataCopyA1Params.srcDValue = k;
        dataCopyA1Params.dstNzC0Stride = m;
        dataCopyA1Params.dstNzNStride = 1;
        dataCopyA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, dataCopyA1Params);

        AscendC::Nd2NzParams dataCopyB1Params;
        dataCopyB1Params.ndNum = 1;
        dataCopyB1Params.nValue = k;
        dataCopyB1Params.dValue = n;
        dataCopyB1Params.srcNdMatrixStride = 0;
        dataCopyB1Params.srcDValue = n;
        dataCopyB1Params.dstNzC0Stride = k;
        dataCopyB1Params.dstNzNStride = 1;
        dataCopyB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, dataCopyB1Params);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<fmap_T> a1Local = inQueueA1.DeQue<fmap_T>();
        AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.AllocTensor<fmap_T>();

        AscendC::LoadData2DParams loadL0AParams;
        loadL0AParams.repeatTimes = aSize * sizeof(fmap_T) / 512;
        loadL0AParams.srcStride = 1;
        loadL0AParams.ifTranspose = false;
        AscendC::LoadData(a2Local, a1Local, loadL0AParams);

        inQueueA2.EnQue<fmap_T>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<weight_T> b1Local = inQueueB1.DeQue<weight_T>();
        AscendC::LocalTensor<weight_T> b2Local = inQueueB2.AllocTensor<weight_T>();

        AscendC::LoadData2dTransposeParams loadDataParams;
        loadDataParams.startIndex = 0;
        nBlockSize = 16;
        loadDataParams.repeatTimes = n / nBlockSize;
        loadDataParams.srcStride = 1;
        loadDataParams.dstGap = 0;
        loadDataParams.dstFracGap = n / nBlockSize - 1;
        AscendC::LoadDataWithTranspose(b2Local, b1Local, loadDataParams);

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<weight_T>(b2Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<fmap_T> a2Local = inQueueA2.DeQue<fmap_T>();
        AscendC::LocalTensor<weight_T> b2Local = inQueueB2.DeQue<weight_T>();
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.AllocTensor<dstCO1_T>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

        outQueueCO1.EnQue<dstCO1_T>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    // dst queue
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    AscendC::GlobalTensor<fmap_T> aGM;
    AscendC::GlobalTensor<weight_T> bGM;
    AscendC::GlobalTensor<dst_T> cGM;

    uint16_t m = 16, k = 16, n = 32;
    uint8_t nBlockSize = 16;
    uint16_t c0Size = 16;
    uint16_t aSize, bSize, cSize, nBlocks;
};

extern "C" __global__ __aicore__ void cube_matmul_loaddata_operator_float(__gm__ uint8_t *a, __gm__ uint8_t *b,
    __gm__ uint8_t *c)
{
    KernelMatmul<dst_type, fmap_type, weight_type, dstCO1_type> op;
    op.Init(a, b, c);
    op.Process();
}
```
