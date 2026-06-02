# SetAddrWithOffset<a name="ZH-CN_TOPIC_0000002523343614"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="zh-cn_topic_0000002554424439_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002554424439_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002554424439_p1883113061818"><a name="zh-cn_topic_0000002554424439_p1883113061818"></a><a name="zh-cn_topic_0000002554424439_p1883113061818"></a><span id="zh-cn_topic_0000002554424439_ph20833205312295"><a name="zh-cn_topic_0000002554424439_ph20833205312295"></a><a name="zh-cn_topic_0000002554424439_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002554424439_p783113012187"><a name="zh-cn_topic_0000002554424439_p783113012187"></a><a name="zh-cn_topic_0000002554424439_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002554424439_row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002554424439_p17301775812"><a name="zh-cn_topic_0000002554424439_p17301775812"></a><a name="zh-cn_topic_0000002554424439_p17301775812"></a><span id="zh-cn_topic_0000002554424439_ph2272194216543"><a name="zh-cn_topic_0000002554424439_ph2272194216543"></a><a name="zh-cn_topic_0000002554424439_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002554424439_p37256491200"><a name="zh-cn_topic_0000002554424439_p37256491200"></a><a name="zh-cn_topic_0000002554424439_p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

设置带有偏移的Tensor地址。用于快速获取定义一个Tensor，同时指定新Tensor相对于旧Tensor首地址的偏移。偏移的长度为旧Tensor的元素个数。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename S>
__aicore__ inline void SetAddrWithOffset(LocalTensor<S> &src, uint32_t offset)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="13.94%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.98%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.08%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p1143210495175"><a name="p1143210495175"></a><a name="p1143210495175"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p13205111913115"><a name="p13205111913115"></a><a name="p13205111913115"></a>基础地址的Tensor，将该Tensor的地址作为基础地址，设置偏移后的Tensor地址。</p>
</td>
</tr>
<tr id="row184841037192110"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="p4484123719216"><a name="p4484123719216"></a><a name="p4484123719216"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p18484133715217"><a name="p18484133715217"></a><a name="p18484133715217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p1461645141810"><a name="p1461645141810"></a><a name="p1461645141810"></a>偏移的长度，单位为元素。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```
// 示例 SetAddrWithOffset，用于快速获取定义一个Tensor，同时指定新Tensor相对于旧Tensor首地址的偏移
// 需要注意，偏移的长度为旧Tensor的元素个数
AscendC::LocalTensor<float> tmpBuffer1 = tempBmm2Queue.AllocTensor<float>();
AscendC::LocalTensor<half> tmpHalfBuffer;
tmpHalfBuffer.SetAddrWithOffset(tmpBuffer1, calcSize * 2);
```

