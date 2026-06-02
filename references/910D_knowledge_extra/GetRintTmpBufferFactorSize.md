# GetRintTmpBufferFactorSize<a name="ZH-CN_TOPIC_0000002554344921"></a>

## 功能说明<a name="section203656409452"></a>

该接口用于获取maxLivedNodeCount和extraBuf，在固定空间大小的情况下，通过maxLivedNodeCount和extraBuf可以推算算子单次最大计算元素数量。maxLivedNodeCount表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

-   算子实现需要调用Rint接口，开发者为其预留currBuff大小的空间，利用GetRintTmpBufferFactorSize接口得到maxLivedNodeCount、extraBuf输出值，可推导算子单次最大计算元素数量为：

    **currentShapeSize = \(currBuff - extraBuf\) / maxLivedNodeCount / typeSize**

-   算子实现需要调用两个kernel侧API KernelIntf1、KernelIntf2，利用两个GetXxxTmpBufferFactorSize（其中Xxx为需要调用的两个高阶API）接口的两组输出值\(maxLivedNodeCount、extraBuf\)以及当前现有的临时空间，推导单次最大计算元素数量currentShapeSize为：

    **currentShapeSize1 = \(currBuff - extraBuf1\) / maxLivedNodeCount1 / typeSize**

    **currentShapeSize2 = \(currBuff - extraBuf2\) / maxLivedNodeCount2 / typeSize**

    **currentShapeSize = min\(currentShapeSize1, currentShapeSize2\)**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLivedNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

## 函数原型<a name="section072684304519"></a>

```
void GetRintTmpBufferFactorSize(const platform_ascendc::PlatformAscendC& ascendcPlatform, const uint32_t typeSize, uint32_t& maxLivedNodeCount, uint32_t& extraBuf)
```

## 参数说明<a name="section114736468451"></a>

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
<tbody><tr id="row12550757181313"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p109583119141"><a name="p109583119141"></a><a name="p109583119141"></a>ascendcPlatform</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p13958161181417"><a name="p13958161181417"></a><a name="p13958161181417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p129581515143"><a name="p129581515143"></a><a name="p129581515143"></a>输入的平台信息。PlatformAscendC的定义请参见<a href="构造及析构函数.md">构造及析构函数</a>。</p>
</td>
</tr>
<tr id="row139791917184711"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p697915173474"><a name="p697915173474"></a><a name="p697915173474"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p169793179474"><a name="p169793179474"></a><a name="p169793179474"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p6979417154719"><a name="p6979417154719"></a><a name="p6979417154719"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row9979161711479"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p169791217174714"><a name="p169791217174714"></a><a name="p169791217174714"></a>maxLivedNodeCount</p>
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

## 返回值说明<a name="section12487115194518"></a>

无

## 约束说明<a name="section3909755194514"></a>

当利用maxLivedNodeCount、extraBuf反推出的currentShapeSize  \* typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

## 调用示例<a name="section9561108464"></a>

完整的调用样例请参考[更多样例](更多样例-104.md)。

```
uint32_t maxLivedNodeCount = 0;
uint32_t extraBuf = 0;
auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
AscendC::GetRintTmpBufferFactorSize(plat, typeSize, maxLivedNodeCount, extraBuf);
```

