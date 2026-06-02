# Kernel侧算子实现<a name="ZH-CN_TOPIC_0000002523311550"></a>

在[核函数](核函数.md)章节已经介绍了kernel侧算子核心的实现方法，本章节侧重于介绍接入CANN框架时编程模式和API的使用。

## 自动生成kernel侧算子实现模板<a name="section282118105913"></a>

在算子工程目录下的“op\_kernel/xxx.cpp”文件中实现算子的核函数。核函数的定义模板已通过msOpGen工具自动生成，样例如下所示**。注意这里参数的顺序按照“输入、输出、workspace、tiling”的顺序排布，开发者不要调整其顺序。**

```
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);// 获取Tiling参数，详见下文介绍
    // TODO: user kernel impl
}
```

> **说明：** 
>算子原型定义中的输入和输出同名的情况下，自动生成的核函数中，输出参数增加ref后缀予以区分。示例如下：
>```
>extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR x_ref, GM_ADDR workspace, GM_ADDR tiling) {
>    ...
>}
>```

## GET\_TILING\_DATA获取Tiling参数<a name="section141061949201"></a>

提供[GET\_TILING\_DATA](GET_TILING_DATA.md)，用于获取算子kernel入口函数传入的tiling信息，并填入注册的Tiling结构体中，此函数会以宏展开的方式进行编译。注意，对应的算子host实现中需要定义TilingData结构体，实现并注册计算TilingData的Tiling函数。具体请参考[Host侧Tiling实现](Host侧Tiling实现.md)。

核函数中调用[GET\_TILING\_DATA](GET_TILING_DATA.md)获取TilingData的样例如下：

```
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
```

## 核函数内获取算子输入输出的数据类型和格式<a name="section194927685210"></a>

算子工程在核函数内提供了DTYPE\_<Arg\>、ORIG\_DTYPE\_<Arg\>、FORMAT\_<Arg\>三种宏用于表示核函数入参（算子的输入输出）的数据类型、原始数据类型和数据格式。其中<Arg\>为入参的大写格式。

-   DTYPE\_<Arg\>，入参的数据类型。是指在Device侧实际可使用的数据类型，比如half、float等。
-   ORIG\_DTYPE\_<Arg\>，入参的原始数据类型。是指在Host侧进行原型定义时，指定的数据类型（不包含命名空间），比如DT\_FLOAT16、DT\_FLOAT等。
-   FORMAT\_<Arg\>，入参的数据格式。是指在Host侧进行原型定义时，指定的数据格式（不包含命名空间），比如FORMAT\_ND、FORMAT\_NZ等。

样例如下：

```
template<class T> func() {}
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    DTYPE_X temp;
    func<DTYPE_Z>();
    if (FORMAT_Y == FORMAT_ND) {
        ...
    }
    if (ORIG_DTYPE_Y == DT_FLOAT) {
        ...
    }
}
```

## 输出shape依赖计算的算子kernel实现<a name="section1961152716108"></a>

某些算子，比如NonZero（统计tensor中非零值的个数），计算完成前无法得知算子输出的shape信息，算子计算完成后才能获取。该类算子在原型定义时，需要使用[OutputShapeDependOnCompute](OutputShapeDependOnCompute.md)接口进行标识，同时在算子核函数中将实际输出shape写入到出参中，便于框架侧基于该信息进行输出内存的管理。

在核函数所有输出的最后增加一个GM\_ADDR类型的输出参数，并在核函数计算完成后，将输出shape信息写入到该出参中。shape信息的排布格式如下，大小为**n \* \(8 + 1\)**，每个元素的数据类型为**uint64\_t**。其中n表示待刷新shape信息的输出个数，每个输出的shape信息都通过第1个元素来保存实际的shape维度（dim），后续的8个元素来保存具体每个维度的shape信息。

<!-- img2text -->
```text
第1个输出（shape依赖计算的输出）维度      第1个输出（shape依赖计算的输出）shape      第2个输出（shape依赖计算的输出）维度      第2个输出（shape依赖计算的输出）shape      第n个输出（shape依赖计算的输出）维度
                                                                                                                                                ...

┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ dim │     │     │     │     │     │     │     │     │ dim │     │     │     │     │     │     │     │     │ dim │ ... │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

<─────>
  1

      <───────────────────────────────────────>
                      8

                                                <─────>
                                                  1

                                                      <───────────────────────────────────────>
                                                                      8

                                                                                                        <─────>
                                                                                                          1

                                    共n个输出（shape依赖计算的输出）
```

- 1: 第1个输出的维度信息，占第1块(dim)
- 8: 第1个输出的shape信息，占第2-9块
- 1: 第2个输出的维度信息，占第10块(dim)
- 8: 第2个输出的shape信息，占第11-18块
- 1: 第n个输出的维度信息，占最后一个可见dim块
- 共n个输出（shape依赖计算的输出）: 整体按“1个dim + 8个shape”作为一组重复排列

> **说明：** 
>-   输出的顺序和原型定义中输出的顺序保持一致。
>-   对于uint64\_t的输出数据类型（对于tensor而言），需要将dim的uint32\_t的高位设置为1，表示以uint64\_t类型解析该tensor。

-   如下示例中，算子中有一个输出依赖计算得出，输出tensor的数据类型为uint32\_t，计算完成后，得到输出的shape为（32, 64），出参shape\_out用于存放该shape信息，值为（2, 32, 64）。代码示例如下：

    ```
    extern "C" __global__ __aicore__ void xxx_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling) {
    ...
        constexpr uint32_t SHAPEOUT_SIZE = 9;
        // 输出数据为2维([32, 64])，tensor类型为uint32_t
        // shapeoutGlobal_uint32用于存放输出Shape信息，数据类型固定为uint64_t
        GlobalTensor<uint64_t> shapeoutGlobal_uint32;
        shapeoutGlobal_uint32.SetGlobalBuffer((__gm__ uint64_t*)shape_out, SHAPEOUT_SIZE);
        shapeoutGlobal_uint32.SetValue(0, 2);
        shapeoutGlobal_uint32.SetValue(1, 32);
        shapeoutGlobal_uint32.SetValue(2, 64);
    ...
    }
    ```

-   如下示例中，算子中有一个输出依赖计算得出，输出tensor的数据类型为uint64\_t，计算完成后，得到输出的shape为（1, 64, 128, 128），出参shape\_out用于存放该shape信息，值为（0x0000000080000000 | 4 , 1, 64, 128, 128）。代码示例如下：

    ```
    extern "C" __global__ __aicore__ void xxx_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling) {
    ...
        constexpr uint32_t SHAPEOUT_SIZE = 9;
        // 输出数据为4维([1, 64, 128, 128])，tensor类型为uint64_t
        // shapeoutGlobal_uint64用于存放输出Shape信息，数据类型固定为uint64_t
        GlobalTensor<uint64_t> shapeoutGlobal_uint64;
        shapeoutGlobal_uint64.SetGlobalBuffer((__gm__ uint64_t*)shape_out, SHAPEOUT_SIZE);
        shapeoutGlobal_uint64.SetValue(0, 0x0000000080000000 | 4);
        shapeoutGlobal_uint64.SetValue(1, 1);
        shapeoutGlobal_uint64.SetValue(2, 64);
        shapeoutGlobal_uint64.SetValue(3, 128);
        shapeoutGlobal_uint64.SetValue(4, 128);
    ...
    }
    ```

-   如下示例中，算子中有两个输出依赖计算得出，输出tensor的数据类型为uint64\_t，计算完成后，得到输出的shape为（16, 32）和 （1, 16, 16, 32），出参shape\_out用于存放该shape信息。示例如下：

    ```
    extern "C" __global__ __aicore__ void xxx_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling) {
        ...
        // 有两个输出需要刷新shape，一个维度为2维[16, 32]，一个维度为4维[1, 16, 16, 32]
        // 输出tensor类型为uint64_t
        constexpr uint32_t SHAPEOUT_SIZE_2 = 18;
        // shapeoutGlobal_uint64_2用于存放输出Shape信息，数据类型固定为uint64_t
        GlobalTensor<uint64_t> shapeoutGlobal_uint64_2;
        shapeoutGlobal_uint64_2.SetGlobalBuffer((__gm__ uint64_t*)shape_out, SHAPEOUT_SIZE_2 );
        shapeoutGlobal_uint64_2.SetValue(0, 0x0000000080000000 | 2);
        shapeoutGlobal_uint64_2.SetValue(1, 16);
        shapeoutGlobal_uint64_2.SetValue(2, 32);
        // index[3]~index[8]数据为占位
        shapeoutGlobal_uint64_2.SetValue(9, 0x0000000080000000 | 4);
        shapeoutGlobal_uint64_2.SetValue(10, 1);
        shapeoutGlobal_uint64_2.SetValue(11, 16);
        shapeoutGlobal_uint64_2.SetValue(12, 16);
        shapeoutGlobal_uint64_2.SetValue(13, 32);
        ...
    }
    ```

