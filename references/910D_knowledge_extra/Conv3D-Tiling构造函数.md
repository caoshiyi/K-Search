# Conv3D Tiling构造函数<a name="ZH-CN_TOPIC_0000002554343555"></a>

## 功能说明<a name="section72951211115"></a>

用于创建一个Conv3D单核Tiling对象。

## 函数原型<a name="section1281382381113"></a>

-   带参构造函数，需要传入硬件平台信息，推荐使用这类构造函数来获得更好的兼容性。
    -   使用PlatformAscendC类传入信息

        ```
        explicit Conv3dTiling(const platform_ascendc::PlatformAscendC& ascendcPlatform)
        ```

    -   使用PlatformInfo传入信息

        当platform\_ascendc::PlatformAscendC无法在Tiling运行时获取时，需要用户自己构造PlatformInfo结构体，透传给Conv3dTiling构造函数。

        ```
        explicit Conv3dTiling(const PlatformInfo& platform)
        ```

-   基类构造函数

    Conv3dTiling继承自基类Conv3dTilingBase，其构造函数如下：

    ```
    explicit Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform)
    ```

    ```
    explicit Conv3dTilingBase(const PlatformInfo& platform)
    ```

## 参数说明<a name="section134342991215"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="17.24%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.77%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="17.24%" headers="mcps1.2.4.1.1 "><p id="p143864062514"><a name="p143864062514"></a><a name="p143864062514"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="9.77%" headers="mcps1.2.4.1.2 "><p id="p893343112595"><a name="p893343112595"></a><a name="p893343112595"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1351741205512"><a name="p1351741205512"></a><a name="p1351741205512"></a>传入硬件平台的信息，PlatformAscendC定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row091441113482"><td class="cellrowborder" valign="top" width="17.24%" headers="mcps1.2.4.1.1 "><p id="p991531119484"><a name="p991531119484"></a><a name="p991531119484"></a>platform</p>
</td>
<td class="cellrowborder" valign="top" width="9.77%" headers="mcps1.2.4.1.2 "><p id="p1291571113485"><a name="p1291571113485"></a><a name="p1291571113485"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p3351541175513"><a name="p3351541175513"></a><a name="p3351541175513"></a>传入硬件版本以及AI Core中各个硬件单元提供的内存大小。PlatformInfo构造时通过<a href="构造及析构函数.md">构造及析构函数</a>获取。</p>
<p id="p1035144114559"><a name="p1035144114559"></a><a name="p1035144114559"></a>PlatformInfo结构定义如下，socVersion通过<a href="GetSocVersion.md">GetSocVersion</a>获取并透传，各类硬件存储空间大小通过<a href="GetCoreMemSize.md">GetCoreMemSize</a>获取并透传。</p>
<a name="screen99307017314"></a><a name="screen99307017314"></a><pre class="screen" codetype="Cpp" id="screen99307017314">struct PlatformInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t l1Size = 0;
    uint64_t l0CSize = 0;
    uint64_t ubSize = 0;
    uint64_t l0ASize = 0;
    uint64_t l0BSize = 0;
    uint64_t btSize = 0;
    uint64_t fbSize = 0;
};</pre>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section102515497127"></a>

无

## 调用示例<a name="section152210580125"></a>

```
// 实例化Conv3d Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetGroups(groups);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
...
conv3dApiTiling.GetTiling(conv3dCustomTilingData.conv3dApiTilingData);
```

