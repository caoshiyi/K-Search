# GetAscendAntiQuantMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002554423821"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧AscendAntiQuant接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

## 函数原型<a name="section620mcpsimp"></a>

```
void GetAscendAntiQuantMaxMinTmpSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose, ge::DataType inputDataType, ge::DataType outputDataType, uint32_t& maxValue, uint32_t& minValue)
```

```
uint32_t GetAscendAntiQuantMaxTmpSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose, ge::DataType inputDataType, ge::DataType outputDataType)
```

```
uint32_t GetAscendAntiQuantMinTmpSize(const ge::Shape& srcShape, const ge::Shape& scaleShape, bool isTranspose, ge::DataType inputDataType, ge::DataType outputDataType)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数列表

<a name="table16492526102317"></a>
<table><thead align="left"><tr id="row1349217266238"><th class="cellrowborder" valign="top" width="16.89%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.5200000000000005%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.59%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row6492192612318"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p189045212538"><a name="p189045212538"></a><a name="p189045212538"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p490205216537"><a name="p490205216537"></a><a name="p490205216537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p290155216536"><a name="p290155216536"></a><a name="p290155216536"></a>输入src的shape信息。</p>
</td>
</tr>
<tr id="row218374242010"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1718316426202"><a name="p1718316426202"></a><a name="p1718316426202"></a>scaleShape</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p618384252020"><a name="p618384252020"></a><a name="p618384252020"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p318394252012"><a name="p318394252012"></a><a name="p318394252012"></a>输入scale的shape信息。</p>
</td>
</tr>
<tr id="row4492162642313"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p690155215538"><a name="p690155215538"></a><a name="p690155215538"></a>isTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p790195215536"><a name="p790195215536"></a><a name="p790195215536"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1492826172315"><a name="p1492826172315"></a><a name="p1492826172315"></a>是否转置。</p>
</td>
</tr>
<tr id="row1449703411211"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p20497734152111"><a name="p20497734152111"></a><a name="p20497734152111"></a>inputDataType</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p196161348132115"><a name="p196161348132115"></a><a name="p196161348132115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p249716347211"><a name="p249716347211"></a><a name="p249716347211"></a>输入数据类型。ge::DataType类型。</p>
</td>
</tr>
<tr id="row22571337192110"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p6258163792118"><a name="p6258163792118"></a><a name="p6258163792118"></a>outputDataType</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p134074493218"><a name="p134074493218"></a><a name="p134074493218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p172581337142117"><a name="p172581337142117"></a><a name="p172581337142117"></a>输出数据类型。ge::DataType类型。</p>
</td>
</tr>
<tr id="row6777152811436"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p590155210531"><a name="p590155210531"></a><a name="p590155210531"></a>输出</p>
<p id="p13901052205318"><a name="p13901052205318"></a><a name="p13901052205318"></a></p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>AscendAntiQuant接口能完成计算所需的最大临时空间大小，超出该值的空间不会被该接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。最大空间大小为0表示计算不需要临时空间。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row66944015289"><td class="cellrowborder" valign="top" width="16.89%" headers="mcps1.2.4.1.1 "><p id="p1369580162813"><a name="p1369580162813"></a><a name="p1369580162813"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="7.5200000000000005%" headers="mcps1.2.4.1.2 "><p id="p176953052810"><a name="p176953052810"></a><a name="p176953052810"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.59%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>AscendAntiQuant接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

**GetAscendAntiQuantMaxMinTmpSize**：无。

**GetAscendAntiQuantMaxTmpSize**：AscendAntiQuant接口能完成计算所需的最大临时空间大小。

**GetAscendAntiQuantMinTmpSize**：AscendAntiQuant接口能完成计算所需的最小临时空间大小。

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
uint32_t maxValue = 0;
uint32_t minValue = 0;
std::vector<int64_t> srcDims = { 64, 512 };
auto srcShape = ge::Shape(srcDims);
std::vector<int64_t> scaleDims = { 1, 512 };
auto scaleShape = ge::Shape(scaleDims);
bool isTranspose = false;
AscendC::GetAscendAntiQuantMaxMinTmpSize(srcShape, scaleShape, isTranspose, ge::DT_INT8, ge::DT_BF16, maxValue, minValue);
```

