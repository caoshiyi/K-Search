# Conv3DBackpropFilter Tiling构造函数<a name="ZH-CN_TOPIC_0000002523304842"></a>

## 功能说明<a name="section618mcpsimp"></a>

用于创建一个Conv3DBackpropFilter 单核Tiling对象。

## 函数原型<a name="section620mcpsimp"></a>

-   带参构造函数，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性。
    -   使用PlatformAscendC类传入信息

        ```
        explicit Conv3dBpFilterTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
        ```

    -   使用PlatformInfo传入信息

        当platform\_ascendc::PlatformAscendC无法在Tiling运行时获取时，需要用户自己构造PlatformInfo结构体，透传给Conv3dBpFilterTiling构造函数。

        ```
        explicit Conv3dBpFilterTiling(const PlatformInfo& platform)
        ```

-   无参构造函数

    ```
    Conv3dBpFilterTiling()
    ```

-   基类构造函数

    Conv3dBpFilterTiling继承自基类Conv3dBpFilterTilingBase，其构造函数如下：

    ```
    Conv3dBpFilterTilingBase()
    ```

    ```
    explicit Conv3dBpFilterTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
    ```

    ```
    explicit Conv3dBpFilterTilingBase(const PlatformInfo& platform)
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
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1078117517"><a name="p1078117517"></a><a name="p1078117517"></a>传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过<a href="构造及析构函数.md">构造及析构函数</a>获取。</p>
<p id="p197827317"><a name="p197827317"></a><a name="p197827317"></a>PlatformInfo结构定义如下，socVersion通过<a href="GetSocVersion.md">GetSocVersion</a>获取并透传，各类硬件存储空间大小通过<a href="GetCoreMemSize.md">GetCoreMemSize</a>获取并透传。</p>
<a name="screen99307017314"></a><a name="screen99307017314"></a><pre class="screen" codetype="Cpp" id="screen99307017314">struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
};</pre>
<p id="p129117471829"><a name="p129117471829"></a><a name="p129117471829"></a>不推荐通过直接填值构造PlatformInfo的方式调用构造函数，例如PlatformInfo(, 1024, 1024, ..);</p>
</td>
</tr>
</tbody>
</table>

在实现Host侧的Tiling函数时，platform\_ascendc::PlatformAscendC用于获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。PlatformAscendC类提供获取这些平台信息的功能。

和platform\_ascendc::PlatformAscendC不同的是，PlatformInfo则用于获取芯片版本以及AI Core中各个硬件单元提供的内存大小等只针对单个AI Core的信息。

## 约束说明<a name="section633mcpsimp"></a>

无

## 使用样例<a name="section14792223131314"></a>

-   无参构造函数

    ```
    Convolution3DBackprop::Conv3dBpFilterTiling tiling;
    tiling.SetWeightType(ConvCommonApi::TPosition::GM,ConvCommonApi::ConvFormat::FRACTAL_Z_3D,ConvCommonApi::ConvDtype::FLOAT32);
    ...
    optiling::Conv3DBackpropFilterTilingData tilingData; 
    int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed
    ...
    ```

-   带参构造函数

    ```
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ConvBackpropApi::Conv3dBpFilterTiling tiling(ascendcPlatform); 
    tiling.SetWeightType(ConvCommonApi::TPosition::GM,Convolution3DBackprop::ConvFormat::FRACTAL_Z_3D,ConvCommonApi::ConvDtype::FLOAT32);
    ...
    optiling::Conv3DBackpropFilterTilingData tilingData; 
    int ret = tiling.GetTiling(tilingData);    // if ret = -1, gen tiling failed
    ```

