# GetLength<a name="ZH-CN_TOPIC_0000002554344301"></a>

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

获取LocalTensor数据长度。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline uint32_t GetLength() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor数据长度，单位为字节。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```
// 获取localTensor的长度(单位为Byte)，数据类型为int32_t，所以是16*sizeof(int32_t)
uint32_t len = inputLocal.GetLength();
// inputLocal:0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// len: 64
```

