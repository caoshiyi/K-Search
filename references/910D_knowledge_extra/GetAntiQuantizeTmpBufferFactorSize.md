# GetAntiQuantizeTmpBufferFactorSize<a name="ZH-CN_TOPIC_0000002523304264"></a>

## 功能说明<a name="section4566122063415"></a>

该接口用于获取maxLiveNodeCount和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuf可以推算算子单次最大计算元素数量。maxLiveNodeCount表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

-   算子实现需要调用AntiQuantize接口，开发者为其预留currBuff大小的空间，利用GetAntiQuantizeTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuf输出值，可推导算子单次最大计算元素数量为：

    **currentShapeSize = \(currBuff - extraBuf\) / maxLiveNodeCount / typeSize**

-   算子实现需要调用两个kernel侧API KernelIntf1、KernelIntf2，利用两个GetXxxTmpBufferFactorSize（其中Xxx为需要调用的两个高阶API）接口的两组输出值\(maxLiveNodeCount、extraBuf\)以及当前现有的临时空间，推导单次最大计算元素数量currentShapeSize为：

    **currentShapeSize1 = \(currBuff - extraBuf1\) / maxLiveNodeCount1 / typeSize**

    **currentShapeSize2 = \(currBuff - extraBuf2\) / maxLiveNodeCount2 / typeSize**

    **currentShapeSize = min\(currentShapeSize1, currentShapeSize2\)**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLiveNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

## 函数原型<a name="section175663209344"></a>

```
void GetAntiQuantizeTmpBufferFactorSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, ge::DataType inputDataType, ge::DataType outputDataType, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
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
<tbody><tr id="row139791917184711"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p189045212538"><a name="p189045212538"></a><a name="p189045212538"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p490205216537"><a name="p490205216537"></a><a name="p490205216537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p290155216536"><a name="p290155216536"></a><a name="p290155216536"></a>输入srcTensor的shape信息。</p>
</td>
</tr>
<tr id="row16808162515611"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p56271151414"><a name="p56271151414"></a><a name="p56271151414"></a>scaleShape</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p12627451545"><a name="p12627451545"></a><a name="p12627451545"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p19627757411"><a name="p19627757411"></a><a name="p19627757411"></a>输入scale的shape信息。</p>
</td>
</tr>
<tr id="row1441130769"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p20497734152111"><a name="p20497734152111"></a><a name="p20497734152111"></a>inputDataType</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p196161348132115"><a name="p196161348132115"></a><a name="p196161348132115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p249716347211"><a name="p249716347211"></a><a name="p249716347211"></a>输入数据类型。ge::DataType类型。</p>
</td>
</tr>
<tr id="row18411933668"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p6258163792118"><a name="p6258163792118"></a><a name="p6258163792118"></a>outputDataType</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p134074493218"><a name="p134074493218"></a><a name="p134074493218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p172581337142117"><a name="p172581337142117"></a><a name="p172581337142117"></a>输出数据类型。ge::DataType类型。</p>
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

```
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;
std::vector<int64_t> srcDims = { 64, 512 };
auto srcShape = ge::Shape(srcDims);
std::vector<int64_t> scaleDims = { 1, 512 };
auto scaleShape = ge::Shape(scaleDims);
bool isTranspose = false;
AscendC::GetAntiQuantizeTmpBufferFactorSize(srcShape, scaleShape, ge::DT_INT8, ge::DT_BF16, maxLiveNodeCount, extraBuf);
```

