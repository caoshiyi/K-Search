# GetDigammaTmpBufferFactorSize<a name="ZH-CN_TOPIC_0000002523304814"></a>

## 功能说明<a name="section4566122063415"></a>

kernel侧Digamma接口的计算需要开发者预留/申请临时空间，最大临时空间（maxTmpBuffer）和输入所占空间（inputSize \* typeSize）存在以下关系：

**maxTmpBuffer = maxLiveNodeCount  \* inputSize \* typeSize + extraBuffer**

其中maxLiveNodeCount表示最大临时空间是输入所占空间的多少倍；extraBuffer表示使用的额外临时空间大小。

该接口用于获取maxLiveNodeCount和extraBuffer，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuffer可以推算算子单次最大计算元素数量；另外，接口获取的maxLiveNodeCount值可能为0，计算时需要判断该值非0，避免除零错误。

示例如下：

算子实现需要调用Digamma接口，开发者为其预留currBuff大小的空间，利用GetDigammaTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuffer输出值，反推Digamma算子单次最大计算元素数量为：

**currentShapeSize = \(currBuff - extraBuffer\) / maxLiveNodeCount  / typeSize**

## 函数原型<a name="section175663209344"></a>

```
void GetDigammaTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuffer)
```

## 参数说明<a name="section8566182019343"></a>

**表 1**  参数列表

<a name="table197951710478"></a>
<table><thead align="left"><tr id="row199799171475"><th class="cellrowborder" valign="top" width="18.94%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.08%" id="mcps1.2.4.1.2"><p id="p1497910174477"><a name="p1497910174477"></a><a name="p1497910174477"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.98%" id="mcps1.2.4.1.3"><p id="p2979181714715"><a name="p2979181714715"></a><a name="p2979181714715"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row139791917184711"><td class="cellrowborder" valign="top" width="18.94%" headers="mcps1.2.4.1.1 "><p id="p697915173474"><a name="p697915173474"></a><a name="p697915173474"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.2 "><p id="p169793179474"><a name="p169793179474"></a><a name="p169793179474"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.98%" headers="mcps1.2.4.1.3 "><p id="p6979417154719"><a name="p6979417154719"></a><a name="p6979417154719"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row9979161711479"><td class="cellrowborder" valign="top" width="18.94%" headers="mcps1.2.4.1.1 "><p id="p192743363407"><a name="p192743363407"></a><a name="p192743363407"></a>maxLiveNodeCount</p>
</td>
<td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.2 "><p id="p1897971710477"><a name="p1897971710477"></a><a name="p1897971710477"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.98%" headers="mcps1.2.4.1.3 "><p id="p162773507585"><a name="p162773507585"></a><a name="p162773507585"></a>最大存活节点数，最大临时空间是输入所占空间的多少倍。</p>
</td>
</tr>
<tr id="row597941784718"><td class="cellrowborder" valign="top" width="18.94%" headers="mcps1.2.4.1.1 "><p id="p59491058114013"><a name="p59491058114013"></a><a name="p59491058114013"></a>extraBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.2 "><p id="p12979151712476"><a name="p12979151712476"></a><a name="p12979151712476"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.98%" headers="mcps1.2.4.1.3 "><p id="p2097941714474"><a name="p2097941714474"></a><a name="p2097941714474"></a>使用的额外临时空间大小，单位为字节。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section13567620163418"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

当利用maxLiveNodeCount，extraBuffer反推出的currentShapeSize  \* typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

## 调用示例<a name="section85671420193420"></a>

```
// 获取输入类型为half的digamma操作的maxLiveNodeCount和extraBuffer
uint32_t typeSize = sizeof(half);
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuffer = 0;

AscendC::GetDigammaTmpBufferFactorSize(typeSize, maxLiveNodeCount, extraBuffer);
```

