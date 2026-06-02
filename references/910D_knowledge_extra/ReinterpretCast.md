# ReinterpretCast<a name="ZH-CN_TOPIC_0000002554423465"></a>

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

将当前Tensor重解释为用户指定的新类型，转换后的Tensor与原Tensor地址及内容完全相同，Tensor的大小（字节数）保持不变。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename CAST_T> 
__aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="p2177206172414"><a name="p2177206172414"></a><a name="p2177206172414"></a>CAST_T</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><p id="p106801314162417"><a name="p106801314162417"></a><a name="p106801314162417"></a>用户指定的新类型。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

重解释转换后的Tensor。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

```
// 示例
// input_local为int32_t 类型，包含16个元素(64字节)
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对inputLocal中第i个位置进行赋值为i
}


// 调用ReinterpretCast将input_local重解释为int16_t类型
AscendC::LocalTensor<int16_t> interpreTensor = inputLocal.ReinterpretCast<int16_t>();
// 示例结果如下，二者数据完全一致，在物理内存上也是同一地址，仅根据不同类型进行了重解释
// inputLocal:0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// interpreTensor:0 0 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 9 0 10 0 11 0 12 0 13 0 14 0 15 0
```

