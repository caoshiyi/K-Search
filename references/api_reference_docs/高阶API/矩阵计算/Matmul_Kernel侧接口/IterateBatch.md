# IterateBatch

**页面ID:** atlasascendc_api_07_0642  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0642.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

单次Matmul计算处理的shape比较小时，由于每次计算均涉及到内部的通信，可能会影响性能，该接口提供批量处理Matmul的功能，调用一次IterateBatch，可以计算出多个singleCoreM * singleCoreN大小的C矩阵。

在使用该接口前，需要了解一些必备的数据排布格式：

- **通用数据格式（NORMAL）**：BMNK的数据排布格式；B：Batch，批处理的大小；M、N、K为矩阵乘[M, K]*[K, N]的矩阵维度；其数据排布格式如下： 

<!-- img2text -->
```text
BMNK ( B = 2 , M = 3 , N = 4 , K = 4 )

矩阵A
┌────┬────┬───────────────────────┐
│    │    │           K           │
├────┼────┼─────┬─────┬─────┬─────┤
│    │    │  1  │  2  │  3  │  4  │
│    │ M  ├─────┼─────┼─────┼─────┤
│ B  │    │  5  │  6  │  7  │  8  │
│    │    ├─────┼─────┼─────┼─────┤
│    │    │  9  │ 10  │ 11  │ 12  │
├────┼────┼─────┼─────┼─────┼─────┤
│    │    │ 13  │ 14  │ 15  │ 16  │
│    │ M  ├─────┼─────┼─────┼─────┤
│    │    │ 17  │ 18  │ 19  │ 20  │
│    │    ├─────┼─────┼─────┼─────┤
│    │    │ 21  │ 22  │ 23  │ 24  │
└────┴────┴─────┴─────┴─────┴─────┘

数据顺序:
1 → 2 → 3 → 4
                ↓
5 → 6 → 7 → 8
                ↓
9 → 10 → 11 → 12
↓
13 → 14 → 15 → 16
                 ↓
17 → 18 → 19 → 20
                 ↓
21 → 22 → 23 → 24

矩阵B
┌────┬────┬───────────────────────┐
│    │    │           N           │
├────┼────┼─────┬─────┬─────┬─────┤
│    │    │  1  │  2  │  3  │  4  │
│    │ K  ├─────┼─────┼─────┼─────┤
│ B  │    │  5  │  6  │  7  │  8  │
│    │    ├─────┼─────┼─────┼─────┤
│    │    │  9  │ 10  │ 11  │ 12  │
│    │    ├─────┼─────┼─────┼─────┤
│    │    │ 13  │ 14  │ 15  │ 16  │
├────┼────┼─────┼─────┼─────┼─────┤
│    │    │ 17  │ 18  │ 19  │ 20  │
│    │ K  ├─────┼─────┼─────┼─────┤
│    │    │ 21  │ 22  │ 23  │ 24  │
│    │    ├─────┼─────┼─────┼─────┤
│    │    │ 25  │ 26  │ 27  │ 28  │
│    │    ├─────┼─────┼─────┼─────┤
│    │    │ 29  │ 30  │ 31  │ 32  │
└────┴────┴─────┴─────┴─────┴─────┘

数据顺序:
1 → 2 → 3 → 4
                ↓
5 → 6 → 7 → 8
                ↓
9 → 10 → 11 → 12
                 ↓
13 → 14 → 15 → 16
↓
17 → 18 → 19 → 20
                 ↓
21 → 22 → 23 → 24
                 ↓
25 → 26 → 27 → 28
                 ↓
29 → 30 → 31 → 32
```

- **BSH/SBH：**B：Batch，批处理的大小； S：sequence length，序列长度；H = N * D，其中，N为head的数量，D为head的大小。Layout格式如下图所示： 

<!-- img2text -->
```
BSH (B = 2, S = 2, H = 12)

                        H
        ┌───────────────────────────────────────────────────────┐
        │  1   2   3   4   5   6   7   8   9   10  11  12      │
┌────┬──┼───────────────────────────────────────────────────────┤
│    │S │ 13  14  15  16  17  18  19  20  21  22  23  24  →    │
│ B  ├──┼───────────────────────────────────────────────────────┤
│    │S │ 25  26  27  28  29  30  31  32  33  34  35  36      │
├────┼──┼───────────────────────────────────────────────────────┤
│    │S │ 37  38  39  40  41  42  43  44  45  46  47  48      │
└────┴──┴───────────────────────────────────────────────────────┘
```

<!-- img2text -->
```
SBH ( S = 2 , B = 2 , H = 12 )

                                B
        ┌───────────────────────────────────────┬───────────────────────────────────────┐
        │                   H                   │                   H                   │
┌───────┼───────────────────────────────────────┼───────────────────────────────────────┤
│   S   │  1   2   3   4   5   6   7   8   9  10  11  12 │ 13  14  15  16  17  18  19  20  21  22  23  24 │
├───────┼───────────────────────────────────────┼───────────────────────────────────────┤
│       │ 25  26  27  28  29  30  31  32  33  34  35  36 │ 37  38  39  40  41  42  43  44  45  46  47  48 │
└───────┴───────────────────────────────────────┴───────────────────────────────────────┘
```

- **BSNGD：**为原始BSH shape做reshape后的shape，S和D为单Batch的矩阵乘的M轴（或N轴）和K轴，一个SD为一个batch的计算数据，Layout格式如下图所示： 

<!-- img2text -->
```text
BSNGD (B = 2, S = 3, N = 3, G = 2, D = 2)

┌────┬────┬─────────────────────────────────────────────────────────────── N ───────────────────────────────────────────────────────────────┐
│    │    │                                  G                                  │                                  G                                  │                                  G                                  │
│    │    ├─────────────────────────────────────┬─────────────────────────────────────┼─────────────────────────────────────┬─────────────────────────────────────┼─────────────────────────────────────┬─────────────────────────────────────┤
│    │    │                  D                  │                  D                  │                  D                  │                  D                  │                  D                  │                  D                  │
├────┼────┼────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┤
│    │ S  │   1    │   2    │   3    │   4    │   5    │   6    │   7    │   8    │   9    │  10    │  11    │  12    │
│    │    │  13    │  14    │  15    │  16    │  17    │  18    │  19    │  20    │  21    │  22    │  23    │  24    │
│ B  │    │  25    │  26    │  27    │  28    │  29    │  30    │  31    │  32    │  33    │  34    │  35    │  36    │
├────┼────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│    │ S  │  37    │  38    │  39    │  40    │  41    │  42    │  43    │  44    │  45    │  46    │  47    │  48    │
│    │    │  49    │  50    │  51    │  52    │  53    │  54    │  55    │  56    │  57    │  58    │  59    │  60    │
│    │    │  61    │  62    │  63    │  64    │  65    │  66    │  67    │  68    │  69    │  70    │  71    │  72    │
└────┴────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

红色访问/排布示意:
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12
                                              ↙
25 → 26 → 27 → 28 → 29 → 30 → 31 → 32 → 33 → 34 → 35 → 36
↙
37 → 38 → 39 → 40 → 41 → 42 → 43 → 44 → 45 → 46 → 47 → 48
                                              ↙
61 → 62 → 63 → 64 → 65 → 66 → 67 → 68 → 69 → 70 → 71 → 72 →
```

说明:
- 左侧维度为 B，B 内部分成两个 S 区块，每个 S 区块包含 3 行数据。
- 顶部维度为 N，N=3；每个 N 下有一个 G，且每个 G 包含 2 个 D，因此横向共 3×2×2=12 列。
- 图中数字 1~72 按单元格顺序填充。
- 红线表示示意的线性排布/访问顺序，分段连接为：
  - 1 → 12
  - 25 → 36
  - 37 → 48
  - 61 → 72
- 红色斜线还表示跨段连接关系：
  - 12 → 25
  - 36 → 37
  - 48 → 61

- **SBNGD：**为原始SBH shape做reshape后shape，S和D为矩阵乘的M轴（或N轴）和K轴，一个SD为一个Batch的计算数据，Layout格式如下图所示： 

<!-- img2text -->
```text
SBNGD ( S = 3 , B = 2 , N = 3 , G = 2 , D = 2 )

                                        B
┌────┬──────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
│    │                              N                               │                              N                               │
├────┼──────────────────────────────┬───────────────────────────────┼──────────────────────────────┬───────────────────────────────┤
│    │              G               │               G               │              G               │               G               │
├────┼──────────────┬───────────────┼──────────────┬────────────────┼──────────────┬───────────────┼──────────────┬────────────────┤
│    │      D       │       D       │      D       │       D        │      D       │       D       │      D       │       D        │
├────┼──────┬───────┬──────┬────────┬──────┬───────┬──────┬─────────┬──────┬───────┬──────┬────────┬──────┬───────┬──────┬─────────┤
│    │  1   │   2   │  3   │   4    │  5   │   6   │  7   │   8     │  9   │  10   │ 11   │  12    │ 13   │  14   │ 15   │  16     │
│ S  ├──────┼───────┼──────┼────────┼──────┼───────┼──────┼─────────┼──────┼───────┼──────┼────────┼──────┼───────┼──────┼─────────┤
│    │ 25   │  26   │ 27   │  28    │ 29   │  30   │ 31   │  32     │ 33   │  34   │ 35   │  36    │ 37   │  38   │ 39   │  40     │
│    ├──────┼───────┼──────┼────────┼──────┼───────┼──────┼─────────┼──────┼───────┼──────┼────────┼──────┼───────┼──────┼─────────┤
│    │ 49   │  50   │ 51   │  52    │ 53   │  54   │ 55   │  56     │ 57   │  58   │ 59   │  60    │ 61   │  62   │ 63   │  64     │
└────┴──────┴───────┴──────┴────────┴──────┴───────┴──────┴─────────┴──────┴───────┴──────┴────────┴──────┴───────┴──────┴─────────┘
      ┌──────┬───────┬──────┬────────┬──────┬───────┬──────┬─────────┐
      │ 17   │  18   │ 19   │  20    │ 21   │  22   │ 23   │  24     │
      ├──────┼───────┼──────┼────────┼──────┼───────┼──────┼─────────┤
      │ 41   │  42   │ 43   │  44    │ 45   │  46   │ 47   │  48     │
      ├──────┼───────┼──────┼────────┼──────┼───────┼──────┼─────────┤
      │ 65   │  66   │ 67   │  68    │ 69   │  70   │ 71   │  72     │
      └──────┴───────┴──────┴────────┴──────┴───────┴──────┴─────────┘

      1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24
      25 → 26 → 27 → 28 → 29 → 30 → 31 → 32 → 33 → 34 → 35 → 36 → 37 → 38 → 39 → 40 → 41 → 42 → 43 → 44 → 45 → 46 → 47 → 48
      49 → 50 → 51 → 52 → 53 → 54 → 55 → 56 → 57 → 58 → 59 → 60 → 61 → 62 → 63 → 64 → 65 → 66 → 67 → 68 → 69 → 70 → 71 → 72
```

- **BNGS1S2：**一般为前两种Layout进行矩阵乘的输出，S1S2数据连续存放，一个S1S2为一个Batch的计算数据，Layout格式如下图所示： 

<!-- img2text -->
```
BNGS1S2 ( B = 2 , N = 3 , G = 2 , S1 = 3 , S2 = 2 )

┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                       B                                                       │
├───────────────────────────────────────────────────────┬───────────────────────────────────────────────────────┤
│                           N                           │                           N                           │
├───────────────────────┬───────────────────────┬───────┼───────────────────────┬───────────────────────┬───────┤
│           G           │           G           │   G   │           G           │           G           │   G   │
├───────────┬───────────┬───────────┬───────────┼───────┼───────────┬───────────┬───────────┬───────────┼───────┤
│    S2     │    S2     │    S2     │    S2     │  S2   │    S2     │    S2     │    S2     │    S2     │  S2   │
│    S2     │    S2     │    S2     │    S2     │  S2   │    S2     │    S2     │    S2     │    S2     │  S2   │
└───────────┴───────────┴───────────┴───────────┴───────┴───────────┴───────────┴───────────┴───────────┴───────┘

S1
↓
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│  1  2 │  7  8 │ 13 14 │ 19 20 │ 25 26 │ 31 32 │ 37 38 │ 43 44 │ 49 50 │ 55 56 │ 61 62 │ 67 68 │
│  3  4 │  9 10 │ 15 16 │ 21 22 │ 27 28 │ 33 34 │ 39 40 │ 45 46 │ 51 52 │ 57 58 │ 63 64 │ 69 70 │
│  5  6 │ 11 12 │ 17 18 │ 23 24 │ 29 30 │ 35 36 │ 41 42 │ 47 48 │ 53 54 │ 59 60 │ 65 66 │ 71 72 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
```

说明:
- 顶层按 B 划分，共 2 个 Batch。
- 每个 B 下按 N 划分，共 3 个 N。
- 每个 N 下按 G 划分，共 2 个 G。
- 每个 G 下按 S2 划分，共 2 列数据。
- 纵向 S1 = 3，因此每个最小块在 S1 方向有 3 行。
- 底部 12 个小块对应 `B×N×G = 2×3×2 = 12` 组，每组包含 `S1×S2 = 3×2 = 6` 个元素。
- 元素编号依次为 1~72。

实例化Matmul时，需要通过MatmulType设置输入输出的Layout格式，当前支持4种Layout类型：BSNGD、SBNGD、BNGS1S2、NORMAL（BMNK的数据排布格式使用NORMAL表示）。

对于BSNGD、SBNGD、BNGS1S2 Layout格式，调用该接口之前需要在host Tiling实现中使用SetALayout、SetBLayout、SetCLayout、SetBatchNum设置A/B/C的Layout轴信息和最大BatchNum数；对于NORMAL Layout格式则需使用SetBatchInfoForNormal设置A/B/C的M/N/K轴信息和A/B矩阵的BatchNum数。

单个矩阵乘迭代顺序可通过tiling参数iterateOrder调整。

更多矩阵编程batch场景的相关内容请参考Batch Matmul基础功能。

#### 函数原型

- mix模式 

  - 输出至GM 

```
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
```

  - 输出至VECIN 

```
template <bool sync = true>
__aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, uint32_t batchA, uint32_t batchB, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0)
```

- 纯cube模式 
      使用前需先调用SetBatchNum接口设置batchA和batchB的大小。 

  - 输出至GM 

```
__aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

  - 输出至VECIN 

```
__aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, bool enPartialSum, uint8_t enAtomic, bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0, const uint32_t matrixStrideC = 0)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 获取C矩阵过程分为同步和异步两种模式：                     - **同步：**需要同步等待IterateBatch执行结束。           - **异步：**不需要同步等待IterateBatch执行结束。                    通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false。默认为同步模式。异步场景需要配合WaitIterateBatch接口使用。 |
| waitIterateBatch | 是否需要通过WaitIterateBatch接口等待IterateBatch执行结束，仅在异步场景下使用。默认为false。          true：需要通过WaitIterateBatch接口等待IterateBatch执行结束。          false：不需要通过WaitIterateBatch接口等待IterateBatch执行结束，开发者自行处理等待IterateBatch执行结束的过程。 |

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| gm | 输出 | C矩阵。类型为GlobalTensor。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/bfloat16_t/int32_t/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/bfloat16_t/int32_t/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/bfloat16_t/int32_t/float |
| ubCmatrix | 输出 | C矩阵。类型为LocalTensor。                       Atlas A3 训练系列产品            /             Atlas A3 推理系列产品            ，支持的数据类型为：half/bfloat16_t/int32_t/float                       Atlas A2 训练系列产品            /             Atlas A2 推理系列产品            ，支持的数据类型为：half/bfloat16_t/int32_t/float                       Atlas 推理系列产品            AI Core，支持的数据类型为：half/bfloat16_t/int32_t/float |
| batchA | 输入 | 左矩阵的batch数。 |
| batchB | 输入 | 右矩阵的batch数。在batchA/batchB不相同的情况下，默认做broadcast操作。          多batch计算支持在G轴上做输入broadcast和输出reduce，左矩阵、右矩阵G轴维度必须是整数倍的关系。 |
| enSequentialWrite | 输入 | 输出是否连续存放数据，即是否开启连续写模式（连续写，写入[baseM, baseN]；非连续写，写入[singleCoreM, singleCoreN]中对应的位置）。                     - 左右矩阵和输出矩阵的存储位置为Unified Buffer，则enSequentialWrite参数应配置为true；           - 输出矩阵的存储位置为GM，则enSequentialWrite参数应配置为false。 |
| matrixStrideA | 输入 | A矩阵源操作数相邻nd矩阵起始地址间的偏移，单位是元素，默认值是0。 |
| matrixStrideB | 输入 | B矩阵源操作数相邻nd矩阵起始地址间的偏移，单位是元素，默认值是0。 |
| matrixStrideC | 输入 | 该参数预留，开发者无需关注。 |
| enPartialSum | 输入 | 是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。在L0C累加时，只支持A矩阵和B矩阵相乘的输出C矩阵规格为singleM==baseM &&singleN==baseN。 |
| enAtomic | 输入 | 是否开启Atomic操作，默认值为0**。**          参数取值：          0：不开启Atomic操作          1：开启AtomicAdd累加操作          2：开启AtomicMax求最大值操作          3：开启AtomicMin求最小值操作 |

#### 约束说明

- 该接口只支持Norm模板，即BatchMatmul只支持Norm模板。
- 对于BSNGD、SBNGD、BNGS1S2 Layout格式，输入A、B矩阵按分形对齐后的多Batch数据总和应小于L1 Buffer的大小；对于NORMAL Layout格式没有这种限制，但需通过MatmulConfig配置输入A、B矩阵多Batch数据大小与L1 Buffer的大小关系；
- 对于BSNGD、SBNGD、BNGS1S2 Layout格式，称左矩阵、右矩阵的G轴分别为ALayoutInfoG、BLayoutInfoG，则ALayoutInfoG / batchA = BLayoutInfoG / batchB；对于NORMAL Layout格式，batchA、batchB必须满足倍数关系。
- 如果接口输出到Unified Buffer上，输出C矩阵大小BaseM*BaseN应小于分配的Unified Buffer内存大小。
- 如果接口输出到Unified Buffer上，且单核计算的N方向大小singleCoreN非32字节对齐，C矩阵的CubeFormat仅支持ND_ALIGN格式，输出C矩阵片时，自动将singleCoreN方向上的数据补齐至32字节。
- 对于BSNGD、SBNGD Layout格式，输入输出只支持ND格式数据。对于BNGS1S2、NORMAL Layout格式， 输入支持ND/NZ格式数据。
- 对于BSNGD、SBNGD Layout格式，不支持连续写模式。
- 该接口不支持量化模式，即不支持SetQuantScalar、SetQuantVector接口。
- BSNGD场景，不支持一次计算多行SD，需要算子程序中循环计算，即(ALayoutInfoN * ALayoutInfoG) / batchA、(BLayoutInfoN * BLayoutInfoG) / batchB均为整数。
- **异步模式**不支持IterateBatch搬运到UB上。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 
        Atlas 推理系列产品
       AI Core上，只支持NORMAL Layout格式。
- 
        Atlas 推理系列产品
       AI Core上，不支持A、B矩阵内存逻辑位置为TPosition::TSCM的输入。
- 
        Atlas 推理系列产品
       AI Core上，Bias不支持复用，Bias的shape大小必须为Batch * N。
- 使用该接口时，A矩阵、B矩阵不支持int4b_t类型的输入，即BatchMatmul不支持int4b_t类型的矩阵输入。

#### 调用示例

- 该示例完成aGM、bGM矩阵乘，结果保存到cGm上，其中aGM、bGM、cGM数据的layout格式均为NORMAL，左矩阵每次计算batchA个MK数据，右矩阵每次计算batchB个KN数据。 

```
// 定义matmul type
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::NORMAL> aType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::NORMAL> bType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL> cType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
// 创建Matmul实例
constexpr static MatmulConfig MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);
AscendC::Matmul<aType, bType, cType, biasType, MM_CFG> mm1;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1);
mm1.Init(&tiling);
mm1.SetTensorA(gm_a, isTransposeAIn);
mm1.SetTensorB(gm_b, isTransposeBIn);
if(tiling.isBias) {
    mm1.SetBias(gm_bias);
}
// 多batch Matmul计算
mm1.IterateBatch(gm_c, batchA, batchB, false);
```

- 该示例完成aGM、bGM矩阵乘，结果保存到cGm上，其中aGM数据的layout格式为BSNGD，bGM数据的layout格式为BSNGD，cGM的layout格式为BNGS1S2，左矩阵每次计算batchA个SD数据，右矩阵每次计算batchB个SD数据。aGM、bGM、cGM数据均为BSNDG格式的BatchMatmul完整示例请参考[BatchMatmul样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/batch_matmul)。 

```
// 定义matmul type
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD> aType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, half, true, LayoutMode::BSNGD> bType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BNGS1S2> cType;
typedef AscendC::MatmulType <AscendC::TPosition::GM, CubeFormat::ND, float> biasType;
// 创建Matmul实例
AscendC::Matmul<aType, bType, cType, biasType> mm1;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1);
mm1.Init(&tiling);
int batchC = batchA > batchB ? batchA : batchB;
int g_lay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
// 计算需要多Batch计算循环次数
int for_exent = tiling.ALayoutInfoB * tiling.ALayoutInfoN * g_lay / tiling.BatchNum;
for(int i=0; i<for_exent; ++i) {
    // 计算每次多batch计算A/B矩阵的起始地址
    int batchOffsetA = i * tiling.ALayoutInfoD * batchA;
    int batchOffsetB = i * tiling.BLayoutInfoD * batchB;
    mm1.SetTensorA(gm_a[batchOffsetA], isTransposeAIn);
    mm1.SetTensorB(gm_b[batchOffsetB], isTransposeBIn);
    int idx_c = i * batchC;
    if (tiling.CLayoutInfoG == 1 && (tiling.BLayoutInfoG != 1 || tiling.ALayoutInfoG != 1)) {
        idx_c = idx_c / (tiling.BLayoutInfoG > tiling.ALayoutInfoG ? tiling.BLayoutInfoG : tiling.ALayoutInfoG);
    }
    if(tiling.isBias) {
        int batchOffsetBias = idx_c * tiling.CLayoutInfoS2;
        mm1.SetBias(gm_bias[batchOffsetBias]);
    }
    int batchOffsetC = idx_c * tiling.CLayoutInfoS2;
    if (C_TYPE::layout == LayoutMode::BNGS1S2) {
        batchOffsetC = idx_c * tiling.CLayoutInfoS2 * tiling.CLayoutInfoS1;
    }
    // 多batch Matmul计算
    mm1.IterateBatch(gm_c[batchOffsetC], batchA, batchB, false);
}
```
