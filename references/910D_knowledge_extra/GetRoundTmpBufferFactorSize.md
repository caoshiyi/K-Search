# GetRoundTmpBufferFactorSize<a name="ZH-CN_TOPIC_0000002523344444"></a>

## 功能说明<a name="section4566122063415"></a>

该接口用于获取maxLiveNodeCount和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuf可以推算算子单次最大计算元素数量。maxLiveNodeCount表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

-   算子实现需要调用Round接口，开发者为其预留currBuff大小的空间，利用GetRoundTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuf输出值，可推导算子单次最大计算元素数量为：

    **currentShapeSize = \(currBuff - extraBuf\) / maxLiveNodeCount / typeSize**

-   算子实现需要调用两个kernel侧API KernelIntf1、KernelIntf2，利用两个GetXxxTmpBufferFactorSize（其中Xxx为需要调用的两个高阶API）接口的两组输出值\(maxLiveNodeCount、extraBuf\)以及当前现有的临时空间，推导单次最大计算元素数量currentShapeSize为：

    **currentShapeSize1 = \(currBuff - extraBuf1\) / maxLiveNodeCount1 / typeSize**

    **currentShapeSize2 = \(currBuff - extraBuf2\) / maxLiveNodeCount2 / typeSize**

    **currentShapeSize = min\(currentShapeSize1, currentShapeSize2\)**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLiveNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

## 函数原型<a name="section175663209344"></a>

```
void GetRoundTmpBufferFactorSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
```

## 参数说明<a name="section8566182019343"></a>

**表 1**  参数列表

<a name="table197951710478"></a>
<table><thead align="left"><tr id="row199799171475"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="p1597991712477"><a name="p1597991712477"></a><a name="p1597991712477"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.19%" id="mcps1.2.4.1.2"><p id="p1497910174477"><a name="p1497910174477"></a><a name="p1497910174477"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="69.92%" id="mcps1.2.4.1.3"><p id="p2979181714715"><a name="p2979181714715"></a><a name="p2979181714715"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row1474835716465"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p5757154817317"><a name="p5757154817317"></a><a name="p5757154817317"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p59310339311"><a name="p59310339311"></a><a name="p59310339311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p1937336310"><a name="p1937336310"></a><a name="p1937336310"></a>输入的平台信息。</p>
</td>
</tr>
<tr id="row139791917184711"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p697915173474"><a name="p697915173474"></a><a name="p697915173474"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p169793179474"><a name="p169793179474"></a><a name="p169793179474"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p6979417154719"><a name="p6979417154719"></a><a name="p6979417154719"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row9979161711479"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p169791217174714"><a name="p169791217174714"></a><a name="p169791217174714"></a>maxLiveNodeCount</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p1897971710477"><a name="p1897971710477"></a><a name="p1897971710477"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p162773507585"><a name="p162773507585"></a><a name="p162773507585"></a>最大存活节点数，表示临时空间是单次计算数据量所占空间的多少倍。</p>
</td>
</tr>
<tr id="row597941784718"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p49794171475"><a name="p49794171475"></a><a name="p49794171475"></a>extraBuf</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p12979151712476"><a name="p12979151712476"></a><a name="p12979151712476"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p2097941714474"><a name="p2097941714474"></a><a name="p2097941714474"></a>使用的额外临时空间大小，单位为字节。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section13567620163418"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

当利用maxLiveNodeCount，extraBuf反推出的currentShapeSize  \* typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

## 调用示例<a name="section85671420193420"></a>

完整的调用样例请参考[更多样例](更多样例-104.md)。

```
auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;
AscendC::GetRoundTmpBufferFactorSize(plat, typeSize, maxLiveNodeCount, extraBuf);
```

