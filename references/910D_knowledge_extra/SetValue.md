# SetValue<a name="ZH-CN_TOPIC_0000002523344698"></a>

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

设置LocalTensor中的某个值。

**该接口仅在LocalTensor的TPosition为VECIN/VECCALC/VECOUT时支持。**

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T1> __aicore__ inline __inout_pipe__(S)
void SetValue(const uint32_t index, const T1 value) const
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001441184464_p5938528141020"><a name="zh-cn_topic_0000001441184464_p5938528141020"></a><a name="zh-cn_topic_0000001441184464_p5938528141020"></a>index</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p14360822193719"><a name="p14360822193719"></a><a name="p14360822193719"></a>LocalTensor索引，单位为元素。</p>
</td>
</tr>
<tr id="row184841037192110"><td class="cellrowborder" valign="top" width="13.94%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001441184464_p188644116443"><a name="zh-cn_topic_0000001441184464_p188644116443"></a><a name="zh-cn_topic_0000001441184464_p188644116443"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="12.98%" headers="mcps1.2.4.1.2 "><p id="p18484133715217"><a name="p18484133715217"></a><a name="p18484133715217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.08%" headers="mcps1.2.4.1.3 "><p id="p2411756104110"><a name="p2411756104110"></a><a name="p2411756104110"></a>待设置的数值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

不要大量使用SetValue对LocalTensor进行赋值，会使性能下降。若需要大批量赋值，请根据实际场景选择[数据填充基础API接口](数据填充.md)或数据填充高阶API接口（[Pad](Pad.md)、[Broadcast](Broadcast.md)），以及在需要生成递增数列的场景，选择[Arange](Arange-115.md)。

## 调用示例<a name="section17531157161314"></a>

```
// srcLen = 256, num = 100, M=50
// 示例1
for (int32_t i = 0; i < srcLen; ++i) {
    inputLocal.SetValue(i, num); // 对inputLocal中第i个位置进行赋值为num
}
// 示例1结果如下：
// 数据(inputLocal): [100 100 100  ... 100]


// srcLen = 256, num = 99, M=50
// 示例2
for (int32_t i = 0; i < srcLen; ++i) {
    inputLocal(i) = num; // 对inputLocal中第i个位置进行赋值为num
}
// 示例2结果如下：
// 数据(inputLocal): [99 99 99  ... 99]
```

