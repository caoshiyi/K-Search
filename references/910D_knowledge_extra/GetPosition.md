# GetPosition<a name="ZH-CN_TOPIC_0000002523303618"></a>

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

获取LocalTensor所在的TPosition逻辑位置，支持TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline int32_t GetPosition() const
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

LocalTensor所在的[TPosition](TPosition.md)逻辑位置。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```
// 示例
AscendC::TPosition srcPos = (AscendC::TPosition)inputLocal.GetPosition();
if (srcPos == AscendC::TPosition::VECCALC) {
    // 处理逻辑1
} else if (srcPos == AscendC::TPosition::A1) {
    // 处理逻辑2
} else {
    // 处理逻辑3
}
```

