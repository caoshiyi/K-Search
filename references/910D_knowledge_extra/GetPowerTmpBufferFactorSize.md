# GetPowerTmpBufferFactorSize<a name="ZH-CN_TOPIC_0000002554423897"></a>

## 功能说明<a name="section4566122063415"></a>

该接口用于获取maxLiveNodeCount和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuf可以推算算子单次最大计算元素数量。maxLiveNodeCount表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

推算示例如下：

算子实现需要调用Power接口，开发者为其预留currBuff大小的空间，利用GetPowerTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuf输出值，可推导算子单次最大计算元素数量为：

**currentShapeSize = \(currBuff - extraBuf\) / maxLiveNodeCount / typeSize**

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间；另外，接口获取的maxLiveNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

## 函数原型<a name="section175663209344"></a>

```
void GetPowerTmpBufferFactorSize(const bool baseIsTensor, const bool expIsTensor, const bool typeIsInt, const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
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
<tbody><tr id="row126631624155020"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p36631024175017"><a name="p36631024175017"></a><a name="p36631024175017"></a>baseIsTensor</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p666317244506"><a name="p666317244506"></a><a name="p666317244506"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p4664102495014"><a name="p4664102495014"></a><a name="p4664102495014"></a>底数是Tensor输入true，否则输入false</p>
</td>
</tr>
<tr id="row830311227507"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p3303622125018"><a name="p3303622125018"></a><a name="p3303622125018"></a>expIsTensor</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p15303142210509"><a name="p15303142210509"></a><a name="p15303142210509"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p230332217508"><a name="p230332217508"></a><a name="p230332217508"></a>指数是Tensor输入true，否则输入false</p>
</td>
</tr>
<tr id="row186822018105017"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p18682191819504"><a name="p18682191819504"></a><a name="p18682191819504"></a>typeIsInt</p>
</td>
<td class="cellrowborder" valign="top" width="13.19%" headers="mcps1.2.4.1.2 "><p id="p186827186505"><a name="p186827186505"></a><a name="p186827186505"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="69.92%" headers="mcps1.2.4.1.3 "><p id="p8682111885018"><a name="p8682111885018"></a><a name="p8682111885018"></a>数据类型是int时输入true，否则输入false</p>
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
<tr id="row597941784718"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p49794171475"><a name="p49794171475"></a><a name="p49794171475"></a>extraBuffer</p>
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
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;
AscendC::GetPowerTmpBufferFactorSize(true, true, true, 4, maxLiveNodeCount, extraBuf);
```

