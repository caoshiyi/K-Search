# WriteSpmBuffer<a name="ZH-CN_TOPIC_0000002523344322"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554424215_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424215_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554424215_p1883113061818"><a name="zh-cn_topic_0000002554424215_p1883113061818"></a><a name="zh-cn_topic_0000002554424215_p1883113061818"></a><span id="zh-cn_topic_0000002554424215_ph20833205312295"><a name="zh-cn_topic_0000002554424215_ph20833205312295"></a><a name="zh-cn_topic_0000002554424215_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554424215_p783113012187"><a name="zh-cn_topic_0000002554424215_p783113012187"></a><a name="zh-cn_topic_0000002554424215_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424215_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554424215_p17301775812"><a name="zh-cn_topic_0000002554424215_p17301775812"></a><a name="zh-cn_topic_0000002554424215_p17301775812"></a><span id="zh-cn_topic_0000002554424215_ph2272194216543"><a name="zh-cn_topic_0000002554424215_ph2272194216543"></a><a name="zh-cn_topic_0000002554424215_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554424215_p37256491200"><a name="zh-cn_topic_0000002554424215_p37256491200"></a><a name="zh-cn_topic_0000002554424215_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将需要溢出暂存的数据拷贝到SPM Buffer中。

## 函数原型<a name="section620mcpsimp"></a>

-   适用于连续和不连续的数据暂存：

    ```
    template <typename T>
    __aicore__ inline void WriteSpmBuffer(const LocalTensor<T>& writeBuffer, const DataCopyParams& copyParams, int32_t writeOffset = 0)
    ```

-   适用于连续的数据暂存：

    ```
    template <typename T>
    __aicore__ inline void WriteSpmBuffer(const LocalTensor<T>& writeBuffer, const int32_t writeSize, int32_t writeOffset = 0)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table1794522316251"></a>
<table><thead align="left"><tr id="row19456238252"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p119458239258"><a name="p119458239258"></a><a name="p119458239258"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.2"><p id="p9945152332514"><a name="p9945152332514"></a><a name="p9945152332514"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p1594552312513"><a name="p1594552312513"></a><a name="p1594552312513"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1694552372511"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1094516239250"><a name="p1094516239250"></a><a name="p1094516239250"></a>writeBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p8945112312514"><a name="p8945112312514"></a><a name="p8945112312514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p169454239253"><a name="p169454239253"></a><a name="p169454239253"></a>需要溢出暂存的Local内存。</p>
</td>
</tr>
<tr id="row524916295111"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p92491829141111"><a name="p92491829141111"></a><a name="p92491829141111"></a>copyParams</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p6249142931114"><a name="p6249142931114"></a><a name="p6249142931114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p18111121482617"><a name="p18111121482617"></a><a name="p18111121482617"></a>搬运参数，DataCopyParams类型，DataCopyParams结构定义请参考<a href="#table6679011144413">表2</a>。</p>
</td>
</tr>
<tr id="row138643616158"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1986436201513"><a name="p1986436201513"></a><a name="p1986436201513"></a>writeSize</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p208648611152"><a name="p208648611152"></a><a name="p208648611152"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p198121859191513"><a name="p198121859191513"></a><a name="p198121859191513"></a>拷贝的元素个数。</p>
</td>
</tr>
<tr id="row126421762566"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p2642463560"><a name="p2642463560"></a><a name="p2642463560"></a>writeOffset</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p18642364562"><a name="p18642364562"></a><a name="p18642364562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p146421267562"><a name="p146421267562"></a><a name="p146421267562"></a>拷贝到SPM Buffer的偏移，单位为字节。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  DataCopyParams结构体参数定义

<a name="table6679011144413"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554344227_row151816516917"><th class="cellrowborder" valign="top" width="15%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000002554344227_p18182513916"><a name="zh-cn_topic_0000002554344227_p18182513916"></a><a name="zh-cn_topic_0000002554344227_p18182513916"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000002554344227_p41815515920"><a name="zh-cn_topic_0000002554344227_p41815515920"></a><a name="zh-cn_topic_0000002554344227_p41815515920"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554344227_row1818105113916"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002554344227_p17780347142614"><a name="zh-cn_topic_0000002554344227_p17780347142614"></a><a name="zh-cn_topic_0000002554344227_p17780347142614"></a>blockCount</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554344227_p478014752618"><a name="zh-cn_topic_0000002554344227_p478014752618"></a><a name="zh-cn_topic_0000002554344227_p478014752618"></a>待搬运的连续传输数据块个数。uint16_t类型，取值范围：blockCount∈[1, 4095]。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002554344227_row2968131992515"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002554344227_p878011470264"><a name="zh-cn_topic_0000002554344227_p878011470264"></a><a name="zh-cn_topic_0000002554344227_p878011470264"></a>blockLen</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554344227_p11780174752617"><a name="zh-cn_topic_0000002554344227_p11780174752617"></a><a name="zh-cn_topic_0000002554344227_p11780174752617"></a>待搬运的每个连续传输数据块长度，单位为DataBlock（32字节）。uint16_t类型，取值范围：blockLen∈[1, 65535]。</p>
<p id="zh-cn_topic_0000002554344227_p187308239357"><a name="zh-cn_topic_0000002554344227_p187308239357"></a><a name="zh-cn_topic_0000002554344227_p187308239357"></a>特殊情况：</p>
<a name="zh-cn_topic_0000002554344227_ul171217289354"></a><a name="zh-cn_topic_0000002554344227_ul171217289354"></a><ul id="zh-cn_topic_0000002554344227_ul171217289354"><li>当dst位于C2PIPE2GM时，单位为128B。</li><li>一般情况：当dst位于C2时，表示源操作数的连续传输数据块长度，单位为64B。针对<span id="zh-cn_topic_0000002554344227_ph172191120153415"><a name="zh-cn_topic_0000002554344227_ph172191120153415"></a><a name="zh-cn_topic_0000002554344227_ph172191120153415"></a>Ascend 950PR/Ascend 950DT</span>，当dst位于C2时，表示源操作数的连续传输数据长度，单位为32B。</li></ul>
</td>
</tr>
<tr id="zh-cn_topic_0000002554344227_row1589112062510"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002554344227_p378018478265"><a name="zh-cn_topic_0000002554344227_p378018478265"></a><a name="zh-cn_topic_0000002554344227_p378018478265"></a>srcGap</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554344227_p19780547162614"><a name="zh-cn_topic_0000002554344227_p19780547162614"></a><a name="zh-cn_topic_0000002554344227_p19780547162614"></a>源操作数相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，srcGap不要超出该数据类型的取值范围。</p>
<p id="zh-cn_topic_0000002554344227_p14646122813194"><a name="zh-cn_topic_0000002554344227_p14646122813194"></a><a name="zh-cn_topic_0000002554344227_p14646122813194"></a>在L1 Buffer -&gt; Fixpipe Buffer场景中，srcGap特指源操作数相邻连续数据块的间隔（前面一个数据块的头与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，srcGap不要超出该数据类型的取值范围。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000002554344227_row3593192082512"><td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000002554344227_p18780347152610"><a name="zh-cn_topic_0000002554344227_p18780347152610"></a><a name="zh-cn_topic_0000002554344227_p18780347152610"></a>dstGap</p>
</td>
<td class="cellrowborder" valign="top" width="85%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554344227_p18780947162613"><a name="zh-cn_topic_0000002554344227_p18780947162613"></a><a name="zh-cn_topic_0000002554344227_p18780947162613"></a>目的操作数相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，dstGap不要超出该数据类型的取值范围。</p>
<p id="zh-cn_topic_0000002554344227_p67331658103720"><a name="zh-cn_topic_0000002554344227_p67331658103720"></a><a name="zh-cn_topic_0000002554344227_p67331658103720"></a>特殊情况：</p>
<a name="zh-cn_topic_0000002554344227_ul1973345893717"></a><a name="zh-cn_topic_0000002554344227_ul1973345893717"></a><ul id="zh-cn_topic_0000002554344227_ul1973345893717"><li>当dst位于C2PIPE2GM时，单位为128B。</li><li>一般情况：当dst位于C2时，表示源操作数的连续传输数据块长度，单位为64B。针对<span id="zh-cn_topic_0000002554344227_ph107331558103717"><a name="zh-cn_topic_0000002554344227_ph107331558103717"></a><a name="zh-cn_topic_0000002554344227_ph107331558103717"></a>Ascend 950PR/Ascend 950DT</span>，当dst位于C2时，表示源操作数的连续传输数据长度，单位为32B。</li></ul>
<p id="zh-cn_topic_0000002554344227_p1764792281920"><a name="zh-cn_topic_0000002554344227_p1764792281920"></a><a name="zh-cn_topic_0000002554344227_p1764792281920"></a>在L1 Buffer -&gt; Fixpipe Buffer场景中，dstGap特指源操作数相邻连续数据块的间隔（前面一个数据块的头与后面数据块的头的间隔），单位为DataBlock（32字节）。uint16_t类型，dstGap不要超出该数据类型的取值范围。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   暂存拷贝到L1时注意writeSize和writeOffset保证32字节对齐
-   拷贝的内存不要超出初始化的SPM Buffer大小，否则会存在溢出踩踏等问题。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
int dataSize = 32; // 假设T为half类型，从ub上申请一块内存32 * sizeof(half)字节
int offset = 32; // 拷贝到spmBuffer时偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize * sizeof(half)); // 分配SPM Buffer空间，大小为32 * sizeof(half)字节
AscendC::LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
AscendC::DataCopyParams copyParams{1, 2, 0, 0}; // 从ub上搬运一个连续传输数据块，一个数据块的长度为2个datablock，一个datablock为32bytes
pipe.WriteSpmBuffer(writeLocal, copyParams, offset); // 将ub上的连续传输数据块搬运到SPM Buffer
pipe.ReadSpmBuffer(writeLocal, copyParams, offset); // 将暂存在SPM Buffer的数据读回到local数据
...
// 当ub内存足够时，将暂存在SPM Buffer的数据块搬运回GM上，dstGlobal为half类型的GlobalTensor
AscendC::DataCopy(dstGlobal, writeLocal, copyParams);
```

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
int dataSize = 32; // 假设T为half类型，从ub上申请一块内存32 * sizeof(half)字节
int offset = 32; // 拷贝到spmBuffer时偏移32字节
pipe.InitBuffer(inQueueSrcVecIn, 1, dataSize * sizeof(half)); // 分配SPM Buffer空间，大小为32 * sizeof(half)字节
AscendC::LocalTensor<half> writeLocal = inQueueSrcVecIn.AllocTensor<half>();
pipe.WriteSpmBuffer(writeLocal, dataSize, offset); // 将ub上的连续传输数据块搬运到SPM Buffer
pipe.ReadSpmBuffer(writeLocal, dataSize, offset); // 将暂存在SPM Buffer的数据读回到local数据
...
// 当ub内存足够时，将暂存在SPM Buffer的数据块搬运回GM上，dstGlobal为half类型的GlobalTensor
AscendC::DataCopy(dstGlobal, writeLocal, dataSize);
```

