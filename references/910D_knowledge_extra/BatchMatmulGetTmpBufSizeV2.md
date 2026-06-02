# BatchMatmulGetTmpBufSizeV2<a name="ZH-CN_TOPIC_0000002523343580"></a>

## 功能说明<a name="section618mcpsimp"></a>

[BatchMatmul](Batch-Matmul基础功能.md)  Tiling调用[GetTiling](GetTiling.md)接口获取Tiling参数后，根据Tiling结构体信息获取L1 Buffer/Unified Buffer/L0C Buffer的使用大小。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t BatchMatmulGetTmpBufSizeV2(AscendC::tiling::TCubeTiling &tiling, matmul_tiling::SysTilingTempBufSize &bufSize)
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
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1050655718146"><a name="p1050655718146"></a><a name="p1050655718146"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p4506145791414"><a name="p4506145791414"></a><a name="p4506145791414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1959473193716"><a name="p1959473193716"></a><a name="p1959473193716"></a>BatchMatmul Tiling的结构体，即BatchMatmulTiling对象得到的TCubeTiling结构体。</p>
<p id="p350610573140"><a name="p350610573140"></a><a name="p350610573140"></a>TCubeTiling为Kernel侧定义的Matmul TilingData，与入参为带AscendC::tiling命名空间的TCubeTiling结构体的<a href="GetTiling.md">GetTiling</a>接口配合使用。</p>
</td>
</tr>
<tr id="row132515237117"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p19507257101413"><a name="p19507257101413"></a><a name="p19507257101413"></a>bufSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p250755718145"><a name="p250755718145"></a><a name="p250755718145"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p1150725719147"><a name="p1150725719147"></a><a name="p1150725719147"></a>根据TCubeTiling结构体信息获取<span id="ph17268645162612"><a name="ph17268645162612"></a><a name="ph17268645162612"></a>L1 Buffer</span>/<span id="ph102685458265"><a name="ph102685458265"></a><a name="ph102685458265"></a>Unified Buffer</span>/<span id="ph026844513269"><a name="ph026844513269"></a><a name="ph026844513269"></a>L0C Buffer</span>的使用大小。</p>
<p id="p1662334214145"><a name="p1662334214145"></a><a name="p1662334214145"></a>SysTilingTempBufSize结构定义如下方代码所示。</p>
</td>
</tr>
</tbody>
</table>

```
struct SysTilingTempBufSize {
    int32_t ubSize = 0; // Unified Buffer大小
    int32_t l1Size = 0; // L1 Buffer大小
    int32_t l0cSize = 0; // L0C Buffer大小
};
```

## 返回值说明<a name="section640mcpsimp"></a>

-1表示获取失败； 0表示获取成功。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::BatchMatmulTiling tiling(ascendcPlatform); 
AscendC::tiling::TCubeTiling tilingData;
...  // 初始化tilingData，详见MatmulTiling类使用说明
int ret = tiling.GetTiling(tilingData);    // 获取Tiling参数
SysTilingTempBufSize bufSize;
BatchMatmulGetTmpBufSizeV2(tilingData, bufSize);
```

