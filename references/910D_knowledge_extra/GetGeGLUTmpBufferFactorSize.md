# GetGeGLUTmpBufferFactorSize<a name="ZH-CN_TOPIC_0000002554424147"></a>

## 功能说明<a name="section943714520116"></a>

kernel侧GeGLU接口的计算需要开发者预留/申请临时空间，最大临时空间（maxTmpBuffer）和输入所占空间（inputSize \* typeSize）存在以下关系：

**maxTmpBuffer = maxLiveNodeCount \* inputSize \* typeSize + extraBuf**

其中maxLiveNodeCount表示最大临时空间是输入所占空间的多少倍；extraBuf表示使用的额外临时空间大小。

该接口用于获取maxLiveNodeCount和extraBuf，在固定空间大小的情况下，通过maxLiveNodeCount和extraBuf可以推算算子单次最大计算元素数量。

示例如下：

算子实现需要调用GeGLU接口，开发者为其预留currBuff大小的空间，利用GetGeGLUTmpBufferFactorSize接口得到maxLiveNodeCount、extraBuf输出值，反推GeGLU算子单次最大计算元素数量为：

**currentShapeSize = \(currBuff - extraBuf\) / maxLiveNodeCount / typeSize**

## 函数原型<a name="section1572681115152"></a>

```
void GetGeGLUTmpBufferFactorSize(const uint32_t typeSize, uint32_t& maxLiveNodeCount, uint32_t& extraBuf)
```

## 参数说明<a name="section1472612116152"></a>

**表 1**  参数列表

<a name="table1472691151513"></a>
<table><thead align="left"><tr id="row157261611131512"><th class="cellrowborder" valign="top" width="18.94%" id="mcps1.2.4.1.1"><p id="p1572617118152"><a name="p1572617118152"></a><a name="p1572617118152"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="10.08%" id="mcps1.2.4.1.2"><p id="p1072741131515"><a name="p1072741131515"></a><a name="p1072741131515"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.98%" id="mcps1.2.4.1.3"><p id="p14727171161511"><a name="p14727171161511"></a><a name="p14727171161511"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row27271111101513"><td class="cellrowborder" valign="top" width="18.94%" headers="mcps1.2.4.1.1 "><p id="p97276114156"><a name="p97276114156"></a><a name="p97276114156"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.2 "><p id="p1772716114154"><a name="p1772716114154"></a><a name="p1772716114154"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.98%" headers="mcps1.2.4.1.3 "><p id="p5727191114153"><a name="p5727191114153"></a><a name="p5727191114153"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为half，此处应传入2。</p>
</td>
</tr>
<tr id="row1672721118156"><td class="cellrowborder" valign="top" width="18.94%" headers="mcps1.2.4.1.1 "><p id="p192743363407"><a name="p192743363407"></a><a name="p192743363407"></a>maxLiveNodeCount</p>
</td>
<td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.2 "><p id="p87277117158"><a name="p87277117158"></a><a name="p87277117158"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.98%" headers="mcps1.2.4.1.3 "><p id="p6727191121511"><a name="p6727191121511"></a><a name="p6727191121511"></a>最大存活节点数，最大临时空间是输入所占空间的多少倍。</p>
</td>
</tr>
<tr id="row20727411201510"><td class="cellrowborder" valign="top" width="18.94%" headers="mcps1.2.4.1.1 "><p id="p59491058114013"><a name="p59491058114013"></a><a name="p59491058114013"></a>extraBuf</p>
</td>
<td class="cellrowborder" valign="top" width="10.08%" headers="mcps1.2.4.1.2 "><p id="p187271711151518"><a name="p187271711151518"></a><a name="p187271711151518"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="70.98%" headers="mcps1.2.4.1.3 "><p id="p107271211121520"><a name="p107271211121520"></a><a name="p107271211121520"></a>使用的额外临时空间大小，单位为字节。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section272791113150"></a>

无

## 约束说明<a name="section197271711131518"></a>

当利用maxLiveNodeCount，extraBuf反推出的currentShapeSize  \* typeSize < 256B时，currentShapeSize按照256B/typeSize的值向上取整。

## 调用示例<a name="section87271911141513"></a>

```
uint32_t typeSize = sizeof(half);
uint32_t maxLiveNodeCount = 0;
uint32_t extraBuf = 0;

AscendC::GetGeGLUTmpBufferFactorSize(typeSize, maxLiveNodeCount, extraBuf);
```

