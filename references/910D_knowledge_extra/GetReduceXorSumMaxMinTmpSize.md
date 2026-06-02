# GetReduceXorSumMaxMinTmpSize<a name="ZH-CN_TOPIC_0000002523304352"></a>

## 功能说明<a name="section618mcpsimp"></a>

kernel侧ReduceXorSum接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大最小临时空间大小，开发者基于此范围选择合适的空间大小作为Tiling参数传递到kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小。
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。该接口**最大临时空间当前等于最小临时空间**。

## 函数原型<a name="section620mcpsimp"></a>

```
void GetReduceXorSumMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数列表

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="14.27%" id="mcps1.2.4.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>接口</p>
</th>
<th class="cellrowborder" valign="top" width="9.98%" id="mcps1.2.4.1.2"><p id="p13216203215467"><a name="p13216203215467"></a><a name="p13216203215467"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.75%" id="mcps1.2.4.1.3"><p id="p1629911506421"><a name="p1629911506421"></a><a name="p1629911506421"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="14.27%" headers="mcps1.2.4.1.1 "><p id="p14566220143420"><a name="p14566220143420"></a><a name="p14566220143420"></a>srcShape</p>
</td>
<td class="cellrowborder" valign="top" width="9.98%" headers="mcps1.2.4.1.2 "><p id="p35665201343"><a name="p35665201343"></a><a name="p35665201343"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.75%" headers="mcps1.2.4.1.3 "><p id="p756672010345"><a name="p756672010345"></a><a name="p756672010345"></a>输入的shape信息。</p>
</td>
</tr>
<tr id="row19299125011422"><td class="cellrowborder" valign="top" width="14.27%" headers="mcps1.2.4.1.1 "><p id="p1929916502421"><a name="p1929916502421"></a><a name="p1929916502421"></a>typeSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.98%" headers="mcps1.2.4.1.2 "><p id="p172166321461"><a name="p172166321461"></a><a name="p172166321461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.75%" headers="mcps1.2.4.1.3 "><p id="p8299150154218"><a name="p8299150154218"></a><a name="p8299150154218"></a>输入的数据类型大小，单位为字节。比如输入的数据类型为int16_t，此处应传入2。</p>
</td>
</tr>
<tr id="row202622311190"><td class="cellrowborder" valign="top" width="14.27%" headers="mcps1.2.4.1.1 "><p id="p12634311919"><a name="p12634311919"></a><a name="p12634311919"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="9.98%" headers="mcps1.2.4.1.2 "><p id="p1726311311495"><a name="p1726311311495"></a><a name="p1726311311495"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.75%" headers="mcps1.2.4.1.3 "><p id="p681410174421"><a name="p681410174421"></a><a name="p681410174421"></a>是否复用源操作数输入的空间，与ReduceXorSum接口一致。</p>
</td>
</tr>
<tr id="row20919154124316"><td class="cellrowborder" valign="top" width="14.27%" headers="mcps1.2.4.1.1 "><p id="p119016528533"><a name="p119016528533"></a><a name="p119016528533"></a>maxValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.98%" headers="mcps1.2.4.1.2 "><p id="p590155210531"><a name="p590155210531"></a><a name="p590155210531"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.75%" headers="mcps1.2.4.1.3 "><p id="p1890195216531"><a name="p1890195216531"></a><a name="p1890195216531"></a>ReduceXorSum接口能完成计算所需最大临时空间大小，超出该值的空间不会被该接口使用。</p>
<div class="note" id="note1275197121212"><a name="note1275197121212"></a><a name="note1275197121212"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1040915016493"><a name="p1040915016493"></a><a name="p1040915016493"></a>maxValue仅作为参考值，有可能大于<span id="ph1088254310583"><a name="ph1088254310583"></a><a name="ph1088254310583"></a>Unified Buffer</span>剩余空间的大小，该场景下，开发者需要根据<span id="ph20796650203718"><a name="ph20796650203718"></a><a name="ph20796650203718"></a>Unified Buffer</span>剩余空间的大小来选取合适的临时空间大小。</p>
</div></div>
</td>
</tr>
<tr id="row124201637134315"><td class="cellrowborder" valign="top" width="14.27%" headers="mcps1.2.4.1.1 "><p id="p1369580162813"><a name="p1369580162813"></a><a name="p1369580162813"></a>minValue</p>
</td>
<td class="cellrowborder" valign="top" width="9.98%" headers="mcps1.2.4.1.2 "><p id="p176953052810"><a name="p176953052810"></a><a name="p176953052810"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.75%" headers="mcps1.2.4.1.3 "><p id="p66955018287"><a name="p66955018287"></a><a name="p66955018287"></a>ReduceXorSum接口能完成计算所需最小临时空间大小。为保证功能正确，接口计算时预留/申请的临时空间不能小于该数值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section92611953111217"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
// 输入shape信息为1024;算子输入的数据类型为int16_t;不允许修改源操作数
std::vector<int64_t> shape_vec = {1024};
ge::Shape shape(shape_vec);
uint32_t maxValue = 0;
uint32_t minValue = 0;
uint32_t typeSize = sizeof(int16_t);
AscendC::GetReduceXorSumMaxMinTmpSize(shape, typeSize, false, maxValue, minValue);
```

