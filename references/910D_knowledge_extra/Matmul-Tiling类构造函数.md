# Matmul Tiling类构造函数<a name="ZH-CN_TOPIC_0000002554344235"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于创建一个Matmul单核Tiling对象，或者多核Tiling对象，或者BatchMatmul Tiling对象。

## 函数原型<a name="section620mcpsimp"></a>

-   带参构造函数，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性。
    -   使用PlatformAscendC类传入信息

        ```
        explicit MatmulApiTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
        ```

        ```
        explicit MultiCoreMatmulTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
        ```

        ```
        explicit BatchMatmulTiling(const platform_ascendc::PlatformAscendC &ascendcPlatform)
        ```

    -   使用PlatformInfo传入信息

        当platform\_ascendc::PlatformAscendC无法在Tiling运行时获取时，需要用户自行构造PlatformInfo结构体，透传给MatmulApiTiling构造函数。

        ```
        explicit MatmulApiTiling(const PlatformInfo& platform)
        ```

        ```
        explicit MultiCoreMatmulTiling(const PlatformInfo &platform)
        ```

-   基类构造函数

    MatmulApiTiling、MultiCoreMatmulTiling和BatchMatmulTiling都继承自基类MatmulApiTilingBase，其构造函数如下：

    ```
    explicit MatmulApiTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
    ```

    ```
    explicit MatmulApiTilingBase(const PlatformInfo& platform)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p143864062514"><a name="p143864062514"></a><a name="p143864062514"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p340161012618"><a name="p340161012618"></a><a name="p340161012618"></a>传入硬件平台的信息，PlatformAscendC定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row091441113482"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p991531119484"><a name="p991531119484"></a><a name="p991531119484"></a>platform</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p1291571113485"><a name="p1291571113485"></a><a name="p1291571113485"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1391514115488"><a name="p1391514115488"></a><a name="p1391514115488"></a>传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过<a href="构造及析构函数.md">构造及析构函数</a>获取。</p>
<p id="p554716466506"><a name="p554716466506"></a><a name="p554716466506"></a>PlatformInfo结构定义如下，socVersion通过<a href="GetSocVersion.md">GetSocVersion</a>获取并透传，各类硬件存储空间大小通过<a href="GetCoreMemSize.md">GetCoreMemSize</a>获取并透传。</p>
<a name="screen99307017314"></a><a name="screen99307017314"></a><pre class="screen" codetype="Cpp" id="screen99307017314">struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
};</pre>
<p id="p129117471829"><a name="p129117471829"></a><a name="p129117471829"></a>不推荐通过直接填值构造PlatformInfo的方式调用构造函数，例如PlatformInfo(socVersion, 1024, 1024, ..);</p>
</td>
</tr>
</tbody>
</table>

在实现Host侧的Tiling函数时，platform\_ascendc::PlatformAscendC用于获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

和platform\_ascendc::PlatformAscendC不同的是，PlatformInfo则用于获取芯片版本、AI Core中各个硬件单元提供的内存大小等只针对单个AI Core的信息。

## 约束说明<a name="section633mcpsimp"></a>

无

## 使用样例<a name="section43651661979"></a>

-   无参构造函数

    ```
    // 单核Tiling
    matmul_tiling::MatmulApiTiling tiling;
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    
    // 多核Tiling
    matmul_tiling::MultiCoreMatmulTiling tiling;   
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    
    // BatchMatmul Tiling
    matmul_tiling::BatchMatmulTiling bmmTiling;
    bmmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    ```

-   带参构造函数

    ```
    // 单核Tiling
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    
    // 多核Tiling
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform); 
    tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    
    // BatchMatmul Tiling
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::BatchMatmulTiling bmmTiling(ascendcPlatform); 
    bmmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
    ```

