# Sort<a name="ZH-CN_TOPIC_0000002554343811"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

-   不带模板参数SortConfig

排序函数，按照数值大小进行降序排序。排序后的数据按照如下排布方式进行保存：

Ascend 950PR/Ascend 950DT采用方式一。

-   排布方式一：

    一次迭代可以完成32个数的排序，排序好的score与其对应的index一起以（score, index）的结构存储在dst中。不论score为half还是float类型，dst中的（score, index）结构总是占据8Bytes空间。如下所示：

    -   当score为float，index为uint32类型时，计算结果中index存储在高4Bytes，score存储在低4Bytes。

        <!-- img2text -->
```
┌──────────────────────┐      ┌──────────────────────┐                          ┌──────────────────────┬──────────────────────┐
│       score[0]       │      │       index[0]       │                          │       Score[5]       │       index[5]       │
├──────────────────────┤      ├──────────────────────┤                          ├──────────────────────┼──────────────────────┤
│       score[1]       │      │       index[1]       │                          │      score[11]       │      index[11]       │
├──────────────────────┤      ├──────────────────────┤                          ├──────────────────────┼──────────────────────┤
│       score[2]       │      │       index[2]       │                          │      score[20]       │      index[20]       │
├──────────────────────┤      ├──────────────────────┤                          ├──────────────────────┼──────────────────────┤
│          ...         │      │          ...         │         ┌────────┐       │          ...         │          ...         │
├──────────────────────┤      ├──────────────────────┤         │        │       ├──────────────────────┼──────────────────────┤
│      score[30]       │      │      index[30]       │   ───→  │        │  ───→ │       score[1]       │       index[1]       │
├──────────────────────┤      ├──────────────────────┤         │        │       ├──────────────────────┼──────────────────────┤
│      score[31]       │      │      index[31]       │         │        │       │       score[8]       │       index[8]       │
└──────────────────────┘      └──────────────────────┘         └────────┘       └──────────────────────┴──────────────────────┘
<──────────────────────>      <──────────────────────>                          <─────────────────────────────────────────────>
            4B                              4B                                                       8B

        ↓                                   ↓                                                       ↑
        │                                   │                                                       │
        │                                   │                                                       │
        └───────────────────────────────────────────────────────────────────────────────────────────────┐
                                                                                                        │
                                                                                         ┌──────────────┘
                                                                                         │
                                                                                         ▼
                                                                                ┌──────────────────────┬──────────────────────┐
                                                                                │       score[8]       │       index[8]       │
                                                                                └──────────────────────┴──────────────────────┘

                                                                 ┌───────────────────────────────────────────────────────────┐
                                                                 │           score[5]                 index[5]              │
                                                                 └───────────────────────────────────────────────────────────┘
```

说明:
- 左侧第一个 4B 列为 score 数组：score[0]、score[1]、score[2]、...、score[30]、score[31]
- 左侧第二个 4B 列为 index 数组：index[0]、index[1]、index[2]、...、index[30]、index[31]
- 经过中间处理后，右侧按 `(score, index)` 结构重排，每组占 `8B`
- 右侧示例中展示的重排结果顺序为：
  - `(Score[5], index[5])`
  - `(score[11], index[11])`
  - `(score[20], index[20])`
  - `...`
  - `(score[1], index[1])`
  - `(score[8], index[8])`
- 红色箭头表示：某个 score 元素与其对应的 index 元素配对后，写入右侧同一条 8B 记录中
- 图中文字保留原样，包含 `Score[5]` 与 `score[...]` 的大小写差异

    -   当score为half，index为uint32类型时，计算结果中index存储在高4Bytes，score存储在低2Bytes， 中间的2Bytes保留。

        <!-- img2text -->
```
┌──────────────┐         ┌──────────────┐                          ┌──────────────┬──────────────┬──────────────┐
│   score[0]   │         │   index[0]   │                          │   Score[5]   │   reserved   │   index[5]   │
├──────────────┤         ├──────────────┤                          ├──────────────┼──────────────┼──────────────┤
│   score[1]   │         │   index[1]   │                          │  score[11]   │   reserved   │  index[11]   │
├──────────────┤         ├──────────────┤                          ├──────────────┼──────────────┼──────────────┤
│   score[2]   │         │   index[2]   │         ┌───────┐        │  score[20]   │   reserved   │  index[20]   │
├──────────────┤         ├──────────────┤         │   ▶   │        ├──────────────┼──────────────┼──────────────┤
│      ...     │         │      ...     │         └───────┘        │      ...     │      ...     │      ...     │
├──────────────┤         ├──────────────┤                          ├──────────────┼──────────────┼──────────────┤
│  score[30]   │         │  index[30]   │                          │   score[1]   │   reserved   │   index[1]   │
├──────────────┤         ├──────────────┤                          ├──────────────┼──────────────┼──────────────┤
│  score[31]   │         │  index[31]   │                          │   score[8]   │   reserved   │   index[8]   │
└──────────────┘         └──────────────┘                          └──────────────┴──────────────┴──────────────┘
<──────────────>         <──────────────>                          <──────────────────────────────────────────────>
      2B                        4B                                                     8B
```

说明:
- 左侧为 score 数据列：score[0]、score[1]、score[2]、...、score[30]、score[31]，单项宽度标注为 2B
- 中间为 index 数据列：index[0]、index[1]、index[2]、...、index[30]、index[31]，单项宽度标注为 4B
- 右侧为合并后的结果排布：每一行依次为 `score[x] | reserved | index[x]`，总宽度为 8B
- 图中示例行依次标注为：`Score[5]`、`score[11]`、`score[20]`、`...`、`score[1]`、`score[8]`，对应右侧的 `index[5]`、`index[11]`、`index[20]`、`...`、`index[1]`、`index[8]`
- reserved 位于 score 和 index 之间
- 结合上下文：当 score 为 half、index 为 uint32 类型时，计算结果中 index 存储在高 4Bytes，score 存储在低 2Bytes，中间的 2Bytes 保留

-   排布方式二：Region Proposal排布

    输入输出数据均为Region Proposal，一次迭代可以完成16个region proposal的排序。每个Region Proposal占用连续8个half/float类型的元素，约定其格式：

    ```
    [x1, y1, x2, y2, score, label, reserved_0, reserved_1]
    ```

    对于数据类型half，每一个Region Proposal占16Bytes，Byte\[15:12\]是无效数据，Byte\[11:0\]包含6个half类型的元素，其中Byte\[11:10\]定义为label，Byte\[9:8\]定义为score，Byte\[7:6\]定义为y2，Byte\[5:4\]定义为x2，Byte\[3:2\]定义为y1，Byte\[1:0\]定义为x1。

    如下图所示，总共包含16个Region Proposals。

    <!-- img2text -->
```text
                    Byte index
┌────────────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬───┐
│            │ 30 │ 28 │ 26 │ 24 │ 22 │ 20 │ 18 │ 16 │ 14 │ 12 │ 10 │  8 │  6 │  4 │  2 │ 0 │
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 0 │ NULL │ NULL │label[1]│score[1]│ y2[1] │ x2[1] │ y1[1] │ x1[1] │ NULL │ NULL │label[0]│score[0]│ y2[0] │ x2[0] │ y1[0] │x1[0]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 1 │ NULL │ NULL │label[3]│score[3]│ y2[3] │ x2[3] │ y1[3] │ x1[3] │ NULL │ NULL │label[2]│score[2]│ y2[2] │ x2[2] │ y1[2] │x1[2]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 2 │ NULL │ NULL │label[5]│score[5]│ y2[5] │ x2[5] │ y1[5] │ x1[5] │ NULL │ NULL │label[4]│score[4]│ y2[4] │ x2[4] │ y1[4] │x1[4]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 3 │ NULL │ NULL │label[7]│score[7]│ y2[7] │ x2[7] │ y1[7] │ x1[7] │ NULL │ NULL │label[6]│score[6]│ y2[6] │ x2[6] │ y1[6] │x1[6]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 4 │ NULL │ NULL │label[9]│score[9]│ y2[9] │ x2[9] │ y1[9] │ x1[9] │ NULL │ NULL │label[8]│score[8]│ y2[8] │ x2[8] │ y1[8] │x1[8]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 5 │ NULL │ NULL │label[11]│score[11]│y2[11]│x2[11]│y1[11]│x1[11]│ NULL │ NULL │label[10]│score[10]│y2[10]│x2[10]│y1[10]│x1[10]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 6 │ NULL │ NULL │label[13]│score[13]│y2[13]│x2[13]│y1[13]│x1[13]│ NULL │ NULL │label[12]│score[12]│y2[12]│x2[12]│y1[12]│x1[12]│
├────────────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤
│ UB addr of 7 │ NULL │ NULL │label[15]│score[15]│y2[15]│x2[15]│y1[15]│x1[15]│ NULL │ NULL │label[14]│score[14]│y2[14]│x2[14]│y1[14]│x1[14]│
└────────────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴───┘
```

    对于数据类型float，每一个Region Proposal占32Bytes，Byte\[31:24\]是无效数据，Byte\[23:0\]包含6个float类型的元素，其中Byte\[23:20\]定义为label，Byte\[19:16\]定义为score，Byte\[15:12\]定义为y2，Byte\[11:8\]定义为x2，Byte\[7:4\]定义为y1，Byte\[3:0\]定义为x1。

    如下图所示，总共包含16个Region Proposals。

    <!-- img2text -->
```text
┌────────────┬────────┬────────┬────────────┬────────────┬────────────┬────────────┬────────────┐
│ Byte index │   28   │   24   │     20     │     16     │     12     │      8     │      4     │ 0
├────────────┼────────┼────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ UB addr off 0  │ NULL   │ NULL   │  label[0]  │  score[0]  │   y2[0]    │   x2[0]    │   y1[0]    │ x1[0]
│ UB addr off 1  │ NULL   │ NULL   │  label[1]  │  score[1]  │   y2[1]    │   x2[1]    │   y1[1]    │ x1[1]
│ UB addr off 2  │ NULL   │ NULL   │  label[2]  │  score[2]  │   y2[2]    │   x2[2]    │   y1[2]    │ x1[2]
│ UB addr off 3  │ NULL   │ NULL   │  label[3]  │  score[3]  │   y2[3]    │   x2[3]    │   y1[3]    │ x1[3]
│ UB addr off 4  │ NULL   │ NULL   │  label[4]  │  score[4]  │   y2[4]    │   x2[4]    │   y1[4]    │ x1[4]
│ UB addr off 5  │ NULL   │ NULL   │  label[5]  │  score[5]  │   y2[5]    │   x2[5]    │   y1[5]    │ x1[5]
│ UB addr off 6  │ NULL   │ NULL   │  label[6]  │  score[6]  │   y2[6]    │   x2[6]    │   y1[6]    │ x1[6]
│ UB addr off 7  │ NULL   │ NULL   │  label[7]  │  score[7]  │   y2[7]    │   x2[7]    │   y1[7]    │ x1[7]
│ UB addr off 8  │ NULL   │ NULL   │  label[8]  │  score[8]  │   y2[8]    │   x2[8]    │   y1[8]    │ x1[8]
│ UB addr off 9  │ NULL   │ NULL   │  label[9]  │  score[9]  │   y2[9]    │   x2[9]    │   y1[9]    │ x1[9]
│ UB addr off 10 │ NULL   │ NULL   │ label[10]  │ score[10]  │  y2[10]    │  x2[10]    │  y1[10]    │ x1[10]
│ UB addr off 11 │ NULL   │ NULL   │ label[11]  │ score[11]  │  y2[11]    │  x2[11]    │  y1[11]    │ x1[11]
│ UB addr off 12 │ NULL   │ NULL   │ label[12]  │ score[12]  │  y2[12]    │  x2[12]    │  y1[12]    │ x1[12]
│ UB addr off 13 │ NULL   │ NULL   │ label[13]  │ score[13]  │  y2[13]    │  x2[13]    │  y1[13]    │ x1[13]
│ UB addr off 14 │ NULL   │ NULL   │ label[14]  │ score[14]  │  y2[14]    │  x2[14]    │  y1[14]    │ x1[14]
│ UB addr off 15 │ NULL   │ NULL   │ label[15]  │ score[15]  │  y2[15]    │  x2[15]    │  y1[15]    │ x1[15]
└────────────┴────────┴────────┴────────────┴────────────┴────────────┴────────────┴────────────┘
```

-   带模板参数SortConfig

    根据模板参数SortConfig，按其中指定的排序算法，对输入数据排序，排序结果可以指定升序或降序排序。

    当函数原型带有输出索引dstIndexTensor参数，需要输出排序结果数据分别对应的索引；若输入带有索引srcIndexTensor参数，则输出索引即为原输入的索引，若输入不带有索引，则对输入数据从0开始生成所需排序数量的索引，最终输出索引即为对应输入数据的索引。如下两幅图，分别为输入带有索引和输入不带索引的数据排序示意图。

    **图 1**  输入带有索引srcIndex的排序样例<a name="fig107141538111418"></a>  
    <!-- img2text -->
```
输入

┌──────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  Value   │  5  │  4  │  2  │  4  │  1  │  6  │  8  │  0  │  3  │
├──────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ srcIndex │  2  │ 15  │  3  │  5  │ 10  │  1  │  7  │  8  │  9  │
└──────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

输出

┌──────────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  Value   │  0  │  1  │  2  │  3  │  4  │  4  │  5  │  6  │  8  │
├──────────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ dstIndex │  8  │ 10  │  3  │  9  │ 15  │  5  │  2  │  1  │  7  │
└──────────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
```

    **图 2**  输入不带索引srcIndex的排序样例<a name="fig138416151180"></a>  
    <!-- img2text -->
```
输入
┌───────┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ Value │ 5 │ 4 │ 2 │ 4 │ 1 │ 6 │ 8 │ 0 │ 3 │
├───────┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ Index │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
└───────┴───┴───┴───┴───┴───┴───┴───┴───┴───┘

输出
┌──────────┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│  Value   │ 0 │ 1 │ 2 │ 3 │ 4 │ 4 │ 5 │ 6 │ 8 │
├──────────┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ dstIndex │ 7 │ 4 │ 2 │ 8 │ 1 │ 3 │ 0 │ 5 │ 6 │
└──────────┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

## 函数原型<a name="section620mcpsimp"></a>

-   不带SortConfig

```
template <typename T, bool isFullSort>
__aicore__ inline void Sort(const LocalTensor<T>& dst, const LocalTensor<T>& concat, const LocalTensor<uint32_t>& index, LocalTensor<T>& tmp, const int32_t repeatTime)
```

-   带SortConfig
    -   接口框架申请临时空间
        -   不带srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
            ```

        -   不带srcIndexTensor参数，带有dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, LocalTensor<uint32_t>& dstIndexTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
            ```

        -   带有srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, typename U, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(const LocalTensor<T>& dstTensor, const LocalTensor<U>& dstIndexTensor, const LocalTensor<T>& srcTensor, const LocalTensor<U>& srcIndexTensor, const uint32_t calCount)
            ```

    -   通过sharedTmpBuffer入参传入临时空间
        -   不带srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
            ```

        -   不带srcIndexTensor参数，带有dstIndexTensor参数

            ```
            template <typename T, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, LocalTensor<uint32_t>& dstIndexTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
            ```

        -   带有srcIndexTensor和dstIndexTensor参数

            ```
            template <typename T, typename U, bool isReuseSource = false, const SortConfig& config = DEFAULT_SORT_CONFIG>
            __aicore__ inline void Sort(LocalTensor<T>& dstTensor, LocalTensor<U>& dstIndexTensor, const LocalTensor<T>& srcTensor, const LocalTensor<U>& srcIndexTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
            ```

            由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

            -   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
            -   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

            通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetSortMaxMinTmpSize](GetSortMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明<a name="section622mcpsimp"></a>

-   不带SortConfig

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p8299155010420"><a name="p8299155010420"></a><a name="p8299155010420"></a>操作数的数据类型。</p>
<p id="p5315184745513"><a name="p5315184745513"></a><a name="p5315184745513"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row1623812985111"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p152385297515"><a name="p152385297515"></a><a name="p152385297515"></a>isFullSort</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p5238529195117"><a name="p5238529195117"></a><a name="p5238529195117"></a>是否开启全排序模式。全排序模式指将全部输入降序排序，非全排序模式下，排序方式请参考<a href="#table62161631132810">表2</a>中的repeatTime说明。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table62161631132810"></a>
<table><thead align="left"><tr id="row12216103118284"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p1421643114288"><a name="p1421643114288"></a><a name="p1421643114288"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p82165310285"><a name="p82165310285"></a><a name="p82165310285"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row82161131182810"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p62165318282"><a name="p62165318282"></a><a name="p62165318282"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p102161931162814"><a name="p102161931162814"></a><a name="p102161931162814"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p3944122817141"><a name="p3944122817141"></a><a name="p3944122817141"></a>目的操作数，shape为[2n]。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p7198164815418"><a name="p7198164815418"></a><a name="p7198164815418"></a><span id="ph1119894813419"><a name="ph1119894813419"></a><a name="ph1119894813419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row5216163192815"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p13216193192813"><a name="p13216193192813"></a><a name="p13216193192813"></a>concat</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p7217031182818"><a name="p7217031182818"></a><a name="p7217031182818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p185486379149"><a name="p185486379149"></a><a name="p185486379149"></a>源操作数，即接口功能说明中的score，shape为[n]。</p>
<p id="p5449124113142"><a name="p5449124113142"></a><a name="p5449124113142"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p3123599392"><a name="p3123599392"></a><a name="p3123599392"></a><span id="ph71345919395"><a name="ph71345919395"></a><a name="ph71345919395"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1521763119281"><a name="p1521763119281"></a><a name="p1521763119281"></a>此源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row88875522820"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p141252118106"><a name="p141252118106"></a><a name="p141252118106"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p105185518102"><a name="p105185518102"></a><a name="p105185518102"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p683812512153"><a name="p683812512153"></a><a name="p683812512153"></a>源操作数，shape为[n]。</p>
<p id="p577151261519"><a name="p577151261519"></a><a name="p577151261519"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p174753218408"><a name="p174753218408"></a><a name="p174753218408"></a><span id="ph247515212407"><a name="ph247515212407"></a><a name="ph247515212407"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p178873523815"><a name="p178873523815"></a><a name="p178873523815"></a>此源操作数固定为uint32_t数据类型。</p>
</td>
</tr>
<tr id="row4809141122410"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1880964192411"><a name="p1880964192411"></a><a name="p1880964192411"></a>tmp</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p9912194814245"><a name="p9912194814245"></a><a name="p9912194814245"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p3809641112411"><a name="p3809641112411"></a><a name="p3809641112411"></a>临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供，临时空间大小BufferSize的获取方式请参考<a href="GetSortTmpSize.md">GetSortTmpSize</a>。数据类型与源操作数保持一致。</p>
<p id="p12523190131819"><a name="p12523190131819"></a><a name="p12523190131819"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1025555124018"><a name="p1025555124018"></a><a name="p1025555124018"></a><span id="ph72551050408"><a name="ph72551050408"></a><a name="ph72551050408"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row521753120287"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1336715511855"><a name="p1336715511855"></a><a name="p1336715511855"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p63676515516"><a name="p63676515516"></a><a name="p63676515516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p636715110511"><a name="p636715110511"></a><a name="p636715110511"></a>重复迭代次数，int32_t类型。</p>
<a name="ul15169323162813"></a><a name="ul15169323162813"></a><ul id="ul15169323162813"><li><span id="ph5485184817117"><a name="ph5485184817117"></a><a name="ph5485184817117"></a>Ascend 950PR/Ascend 950DT</span>：每次迭代完成32个元素的排序，下次迭代concat和index各跳过32个elements，dst跳过32*8 Byte空间。取值范围：repeatTime∈[0,255]。</li></ul>
<a name="ul1879420254213"></a><a name="ul1879420254213"></a>
</td>
</tr>
</tbody>
</table>

-   带SortConfig

    **表 3**  模板参数说明

    <a name="table974693213019"></a>
    <table><thead align="left"><tr id="row1574612327306"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p10746103212308"><a name="p10746103212308"></a><a name="p10746103212308"></a>接口</p>
    </th>
    <th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p3746133293012"><a name="p3746133293012"></a><a name="p3746133293012"></a>功能</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row274693210302"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p167462322301"><a name="p167462322301"></a><a name="p167462322301"></a>T</p>
    </td>
    <td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p97463322306"><a name="p97463322306"></a><a name="p97463322306"></a>操作数srcTensor和dstTensor的数据类型。</p>
    <p id="p12498718201116"><a name="p12498718201116"></a><a name="p12498718201116"></a><span id="ph849841871119"><a name="ph849841871119"></a><a name="ph849841871119"></a>Ascend 950PR/Ascend 950DT</span>，RADIX_SORT排序算法支持的数据类型为：uint8_t、int8_t、uint16_t、int16_t、uint32_t、int32_t、half、bfloat16_t、float、uint64_t、int64_t，MERGE_SORT排序算法支持的数据类型为：half、float。</p>
    </td>
    </tr>
    <tr id="row115431637141113"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p142453412111"><a name="p142453412111"></a><a name="p142453412111"></a>U</p>
    </td>
    <td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p82451418111"><a name="p82451418111"></a><a name="p82451418111"></a>操作数srcIndexTensor和dstIndexTensor的数据类型。</p>
    <p id="p1837712315112"><a name="p1837712315112"></a><a name="p1837712315112"></a><span id="ph637714231118"><a name="ph637714231118"></a><a name="ph637714231118"></a>Ascend 950PR/Ascend 950DT</span>，RADIX_SORT排序算法支持的数据类型为：uint32_t、int32_t、uint64_t、int64_t，MERGE_SORT排序算法支持的数据类型为：uint32_t。</p>
    </td>
    </tr>
    <tr id="row5746532143013"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p19746123214308"><a name="p19746123214308"></a><a name="p19746123214308"></a>isReuseSource</p>
    </td>
    <td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p17461327306"><a name="p17461327306"></a><a name="p17461327306"></a>可选参数。是否可以复用输入的Tensor空间。</p>
    </td>
    </tr>
    <tr id="row16737540173315"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p17390426153110"><a name="p17390426153110"></a><a name="p17390426153110"></a>config</p>
    </td>
    <td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p8322255103120"><a name="p8322255103120"></a><a name="p8322255103120"></a>可选参数。Sort接口的相应配置：选择的排序算法，排序结果的升降序。数据类型SortConfig，定义如下。</p>
    <a name="screen1083884503520"></a><a name="screen1083884503520"></a><pre class="screen" codetype="Cpp" id="screen1083884503520">struct SortConfig {
        SortType type = SortType::RADIX_SORT; // 排序算法
        bool isDescend = false; // 是否降序排序，默认值为false，输出结果升序排序
    };</pre>
    <p id="p24241137113519"><a name="p24241137113519"></a><a name="p24241137113519"></a>其中，排序算法的数据类型SortType取值如下。</p>
    <a name="screen1623155513359"></a><a name="screen1623155513359"></a><pre class="screen" codetype="Cpp" id="screen1623155513359">enum class SortType {
        RADIX_SORT,  // 使用基排序算法实现
        MERGE_SORT   // 使用归并排序算法实现
    };</pre>
    <p id="p139259518411"><a name="p139259518411"></a><a name="p139259518411"></a>Sort提供了两种不同的排序算法，MERGE_SORT归并排序算法和RADIX_SORT基排序算法。两种算法在执行速度、时间复杂度和算法稳定性上表现不同。</p>
    <a name="ul240914179411"></a><a name="ul240914179411"></a><ul id="ul240914179411"><li>MERGE_SORT是一种稳定的排序算法，在所有情况下算法的时间复杂度都是O(nlogn)。</li><li>RADIX_SORT算法的时间复杂度是O(n)，在处理大量数据时，如果最大数字的位数较少，该算法的效率很高，可以接近线性时间复杂度。但是如果最大数字的位数很大，时间复杂度会接近O(n^2)。</li></ul>
    <p id="p89623733410"><a name="p89623733410"></a><a name="p89623733410"></a>config的默认值DEFAULT_SORT_CONFIG取值如下，使用基排序RADIX_SORT，对排序结果升序排序。</p>
    <a name="screen578013207400"></a><a name="screen578013207400"></a><pre class="screen" codetype="Cpp" id="screen578013207400">constexpr SortConfig DEFAULT_SORT_CONFIG = {SortType::RADIX_SORT, false};</pre>
    </td>
    </tr>
    </tbody>
    </table>

    **表 4**  参数说明

    <a name="table12391451143019"></a>
    <table><thead align="left"><tr id="row133965116309"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p20391511301"><a name="p20391511301"></a><a name="p20391511301"></a>参数名称</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.591259125912593%" id="mcps1.2.4.1.2"><p id="p33985118304"><a name="p33985118304"></a><a name="p33985118304"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="73.74737473747375%" id="mcps1.2.4.1.3"><p id="p1339135143010"><a name="p1339135143010"></a><a name="p1339135143010"></a>含义</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row839451113014"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19392518308"><a name="p19392518308"></a><a name="p19392518308"></a>dstTensor</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p143985110303"><a name="p143985110303"></a><a name="p143985110303"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p71935429153"><a name="p71935429153"></a><a name="p71935429153"></a>值目的操作数，shape为[n]。MERGE_SORT算法下输出数据的每个元素需要按8Byte申请空间。</p>
    <p id="p4394515307"><a name="p4394515307"></a><a name="p4394515307"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_4"><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_4"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_4"><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_4"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_4"><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_4"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
    <p id="p2041811134016"><a name="p2041811134016"></a><a name="p2041811134016"></a><span id="ph441151116403"><a name="ph441151116403"></a><a name="ph441151116403"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
    </td>
    </tr>
    <tr id="row339551183018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p123915511308"><a name="p123915511308"></a><a name="p123915511308"></a>dstIndexTensor</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p163945110305"><a name="p163945110305"></a><a name="p163945110305"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p93915193012"><a name="p93915193012"></a><a name="p93915193012"></a>索引目的操作数，shape为[n]。当输入不带srcIndexTensor时，只支持uint32_t类型。</p>
    <p id="p1839165113306"><a name="p1839165113306"></a><a name="p1839165113306"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_5"><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_5"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_5"><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_5"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_5"><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_5"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
    <p id="p260421454010"><a name="p260421454010"></a><a name="p260421454010"></a><span id="ph14604111494010"><a name="ph14604111494010"></a><a name="ph14604111494010"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
    </td>
    </tr>
    <tr id="row64016513304"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p144065153013"><a name="p144065153013"></a><a name="p144065153013"></a>srcTensor</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p6408510302"><a name="p6408510302"></a><a name="p6408510302"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p12498218171119"><a name="p12498218171119"></a><a name="p12498218171119"></a>值源操作数，shape为[n]。</p>
    <p id="p13498141851115"><a name="p13498141851115"></a><a name="p13498141851115"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_6"><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_6"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_6"><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_6"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_6"><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_6"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
    <p id="p7953171617406"><a name="p7953171617406"></a><a name="p7953171617406"></a><span id="ph10953191617402"><a name="ph10953191617402"></a><a name="ph10953191617402"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
    <p id="p0973165215111"><a name="p0973165215111"></a><a name="p0973165215111"></a>此源操作数的数据类型需要与值目的操作数保持一致。</p>
    </td>
    </tr>
    <tr id="row1940155103018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p20401151193012"><a name="p20401151193012"></a><a name="p20401151193012"></a>srcIndexTensor</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p340151193016"><a name="p340151193016"></a><a name="p340151193016"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p63771123171114"><a name="p63771123171114"></a><a name="p63771123171114"></a>索引源操作数，shape为[n]。</p>
    <p id="p19377723151117"><a name="p19377723151117"></a><a name="p19377723151117"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_7"><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_7"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_7"><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_7"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_7"><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_7"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
    <p id="p16903161916406"><a name="p16903161916406"></a><a name="p16903161916406"></a><span id="ph3903191911409"><a name="ph3903191911409"></a><a name="ph3903191911409"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
    <p id="p837792381117"><a name="p837792381117"></a><a name="p837792381117"></a>此源操作数的数据类型需要与索引目的操作数保持一致。</p>
    </td>
    </tr>
    <tr id="row44025115302"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p174075163014"><a name="p174075163014"></a><a name="p174075163014"></a>sharedTmpBuffer</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p184019514305"><a name="p184019514305"></a><a name="p184019514305"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1217519430108"><a name="p1217519430108"></a><a name="p1217519430108"></a>临时空间。接口内部复杂计算时用于存储中间变量，由开发者提供，临时空间大小BufferSize的获取方式请参考<a href="GetSortMaxMinTmpSize.md">GetSortMaxMinTmpSize</a>。数据类型为uint8_t。</p>
    <p id="p161751243181010"><a name="p161751243181010"></a><a name="p161751243181010"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_8"><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_8"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_8"><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_8"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_8"><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_8"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
    <p id="p1298361411"><a name="p1298361411"></a><a name="p1298361411"></a><span id="ph12981563413"><a name="ph12981563413"></a><a name="ph12981563413"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
    </td>
    </tr>
    <tr id="row15132103181011"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p8132123151013"><a name="p8132123151013"></a><a name="p8132123151013"></a>calCount</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.591259125912593%" headers="mcps1.2.4.1.2 "><p id="p201321037106"><a name="p201321037106"></a><a name="p201321037106"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.74737473747375%" headers="mcps1.2.4.1.3 "><p id="p1713219331017"><a name="p1713219331017"></a><a name="p1713219331017"></a>需要进行排序的数据元素个数。uint32_t类型。</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值说明<a name="section91032023123812"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   不带SortConfig：

-   当存在score\[i\]与score\[j\]相同时，如果i\>j，则score\[j\]将首先被选出来，排在前面，即index的顺序与输入顺序一致。
-   非全排序模式下，每次迭代内的数据会进行排序，不同迭代间的数据不会进行排序。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   带SortConfig：
    -   基排序RadixSort和归并排序MergeSort都为稳定排序，即相同值在排序后的先后顺序保持不变。
    -   值目的操作数、值源操作数、索引目的操作数、索引源操作数的元素个数相同，且calCount参数值不能超过元素个数。
    -   不支持源操作数与目的操作数地址重叠。
    -   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
    -   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
    -   使用MERGE\_SORT算法排序时，待排序的元素个数必须是32的倍数。若不是32的倍数，用户需要手动将数据量补齐到32的倍数。

## 调用示例<a name="section642mcpsimp"></a>

算子样例工程请通过[sort样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/06_sort/sort)链接获取。

-   处理128个half类型数据。

    该样例适用于：

    Ascend 950PR/Ascend 950DT

    ```
    uint32_t elementCount = 128;
    uint32_t m_sortRepeatTimes = m_elementCount / 32;
    uint32_t m_extractRepeatTimes = m_elementCount / 32;
    AscendC::Concat(concatLocal, valueLocal, concatTmpLocal, m_concatRepeatTimes);
    AscendC::Sort<T, isFullSort>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, m_sortRepeatTimes);
    AscendC::Extract(dstValueLocal, dstIndexLocal, sortedLocal, m_extractRepeatTimes);
    ```

    ```
    示例结果
    输入数据（srcValueGm）: 128个half类型数据
    [31 30 29 ... 2 1 0
     63 62 61 ... 34 33 32
     95 94 93 ... 66 65 64
     127 126 125 ... 98 97 96]
    输入数据（srcIndexGm）:
    [31 30 29 ... 2 1 0
     63 62 61 ... 34 33 32
     95 94 93 ... 66 65 64
     127 126 125 ... 98 97 96]
    输出数据（dstValueGm）:
    [127 126 125 ... 2 1 0]
    输出数据（dstIndexGm）:
    [127 126 125 ... 2 1 0]
    ```

-   带SortConfig
    -   处理1024个half类型数据，输入索引和输出索引为1024个uint32\_t类型数据。

        该样例适用于：

        Ascend 950PR/Ascend 950DT

        ```
        static constexpr AscendC::SortConfig config = {AscendC::SortType::RADIX_SORT, false};
        Sort<T, false, config>(dstLocal, dstIndexLocal, srcLocal, 1024);
        ```

        ```
        示例结果
        输入数据（srcGm）: 1024个half类型数据
        [1023 1022 ... 2 1 0]
        输入数据（srcIndexGm）: 1024个uint32_t类型数据
        [0 1 2 ... 1022 1023]
        输出数据（dstGm）:
        [0 1 2 ... 1022 1023]
        输出数据（dstIndexGm）:
        [1023 1022 ... 2 1 0]
        ```

