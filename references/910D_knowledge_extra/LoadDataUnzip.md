# LoadDataUnzip<a name="ZH-CN_TOPIC_0000002554423701"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将GM上的数据解压并搬运到A1/B1/B2上。执行该API前需要执行[LoadUnzipIndex](LoadUnzipIndex.md)加载压缩索引表。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void LoadDataUnzip(const LocalTensor<T>& dst, const GlobalTensor<T>& src)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.371637163716375%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.341134113411341%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.28722872287229%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p18889201617312"><a name="p18889201617312"></a><a name="p18889201617312"></a>目的操作数，类型为LocalTensor，支持的TPosition为A1/B1/B2。</p>
<p id="p3844132913311"><a name="p3844132913311"></a><a name="p3844132913311"></a>LocalTensor的起始地址需要保证：TPosition为A1/B1时，32字节对齐；TPosition为B2时，512B对齐。</p>
<p id="p66101129164712"><a name="p66101129164712"></a><a name="p66101129164712"></a>支持的数据类型为：int8_t。</p>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p96501710201711"><a name="p96501710201711"></a><a name="p96501710201711"></a>源操作数，类型为GlobalTensor。数据类型需要与dst保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 返回值说明<a name="section640mcpsimp"></a>

无

