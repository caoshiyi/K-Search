# GetWithOffset<a name="ZH-CN_TOPIC_0000002523343820"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002523343584_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002523343584_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002523343584_p1883113061818"><a name="zh-cn_topic_0000002523343584_p1883113061818"></a><a name="zh-cn_topic_0000002523343584_p1883113061818"></a><span id="zh-cn_topic_0000002523343584_ph20833205312295"><a name="zh-cn_topic_0000002523343584_ph20833205312295"></a><a name="zh-cn_topic_0000002523343584_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002523343584_p783113012187"><a name="zh-cn_topic_0000002523343584_p783113012187"></a><a name="zh-cn_topic_0000002523343584_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002523343584_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002523343584_p17301775812"><a name="zh-cn_topic_0000002523343584_p17301775812"></a><a name="zh-cn_topic_0000002523343584_p17301775812"></a><span id="zh-cn_topic_0000002523343584_ph2272194216543"><a name="zh-cn_topic_0000002523343584_ph2272194216543"></a><a name="zh-cn_topic_0000002523343584_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002523343584_p37256491200"><a name="zh-cn_topic_0000002523343584_p37256491200"></a><a name="zh-cn_topic_0000002523343584_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section954881431816"></a>

以TBuf为基地址，向后偏移指定长度，将偏移后的地址作为起始地址，提取长度为指定值的Tensor。

## 函数原型<a name="section1449617323189"></a>

```
template <typename T>
__aicore__ inline LocalTensor<T> GetWithOffset(uint32_t size, uint32_t bufOffset)
```

## 参数说明<a name="section85121215142514"></a>

**表 1**  模板参数说明

<a name="table012895562310"></a>
<table><thead align="left"><tr id="row2128195532318"><th class="cellrowborder" valign="top" width="12.65%" id="mcps1.2.3.1.1"><p id="p1212885512232"><a name="p1212885512232"></a><a name="p1212885512232"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="87.35000000000001%" id="mcps1.2.3.1.2"><p id="p16129155552315"><a name="p16129155552315"></a><a name="p16129155552315"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row6129355182310"><td class="cellrowborder" valign="top" width="12.65%" headers="mcps1.2.3.1.1 "><p id="p81291855102318"><a name="p81291855102318"></a><a name="p81291855102318"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="87.35000000000001%" headers="mcps1.2.3.1.2 "><p id="p14983161812418"><a name="p14983161812418"></a><a name="p14983161812418"></a>待获取Tensor的数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table1794522316251"></a>
<table><thead align="left"><tr id="row19456238252"><th class="cellrowborder" valign="top" width="12.36%" id="mcps1.2.4.1.1"><p id="p119458239258"><a name="p119458239258"></a><a name="p119458239258"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.379999999999999%" id="mcps1.2.4.1.2"><p id="p9945152332514"><a name="p9945152332514"></a><a name="p9945152332514"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.26%" id="mcps1.2.4.1.3"><p id="p1594552312513"><a name="p1594552312513"></a><a name="p1594552312513"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1694552372511"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p1094516239250"><a name="p1094516239250"></a><a name="p1094516239250"></a>size</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p8945112312514"><a name="p8945112312514"></a><a name="p8945112312514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p169454239253"><a name="p169454239253"></a><a name="p169454239253"></a>需要获取的Tensor元素个数。</p>
</td>
</tr>
<tr id="row165981554174"><td class="cellrowborder" valign="top" width="12.36%" headers="mcps1.2.4.1.1 "><p id="p959810553176"><a name="p959810553176"></a><a name="p959810553176"></a>bufOffset</p>
</td>
<td class="cellrowborder" valign="top" width="12.379999999999999%" headers="mcps1.2.4.1.2 "><p id="p1659820553174"><a name="p1659820553174"></a><a name="p1659820553174"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.26%" headers="mcps1.2.4.1.3 "><p id="p125983554175"><a name="p125983554175"></a><a name="p125983554175"></a>从起始位置的偏移长度，单位是字节，且需32字节对齐。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

size的数值是Tensor中元素的个数，size\*sizeof\(T\) + bufOffset不能超过TBuf初始化时的长度。

bufOffset需满足32字节对齐的要求。

## 返回值说明<a name="section640mcpsimp"></a>

获取到的[LocalTensor](LocalTensor.md)。

## 调用示例<a name="section5725818154718"></a>

```
// 为TBuf初始化分配内存，分配内存长度为1024字节
AscendC::TPipe pipe;
AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf; // 模板参数为TPosition中的VECCALC类型
uint32_t byteLen = 1024;
pipe.InitBuffer(calcBuf, byteLen);
// 从calcBuf偏移64字节获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
AscendC::LocalTensor<int32_t> tempTensor1 = calcBuf.GetWithOffset<int32_t>(128, 64);
```

