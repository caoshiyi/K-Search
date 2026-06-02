# LoadDataWithSparse<a name="ZH-CN_TOPIC_0000002523344308"></a>

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

用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化。索引矩阵的数据类型为int2，需要拼成int8的数据类型，再传入接口。

索引矩阵在一个int8的地址中的排布是逆序排布的，例如：索引矩阵1 2 0 1 0 2 1 0，在地址中的排布为1 0 2 1 0 1 2 0，其中1 0 2 1（对应索引矩阵前四位1 2 0 1）为一个int8，0 1 2 0（对应索引矩阵后四位0 2 1 0）为一个int8。

索引矩阵的功能说明参考[MmadWithSparse](MmadWithSparse.md)。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = int8_t, typename U = uint8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int8_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, uint8_t>::value, bool>::type = true>
__aicore__ inline void LoadDataWithSparse(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<U>& idx, const LoadData2dParams& loadDataParam)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="17.43%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.57%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>dst、src的数据类型。</p>
</td>
</tr>
<tr id="row14306152011913"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="p1030652081918"><a name="p1030652081918"></a><a name="p1030652081918"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="p14306152041916"><a name="p14306152041916"></a><a name="p14306152041916"></a>idx的数据类型。</p>
<a name="ul3987732122120"></a><a name="ul3987732122120"></a><ul id="ul3987732122120"><li>当dst、src、idx为基础数据类型时，T和U必须为uint8_t类型，否则编译失败。</li></ul>
<a name="ul17416131483320"></a><a name="ul17416131483320"></a><ul id="ul17416131483320"><li>当dst、src、idx为<a href="TensorTrait.md">TensorTrait</a>类型时，T和U的LiteType必须为int8_t类型，否则编译失败。</li></ul>
<p id="p882014184212"><a name="p882014184212"></a><a name="p882014184212"></a>最后两个模板参数仅用于上述数据类型检查，用户无需关注。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

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
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p56610562610"><a name="p56610562610"></a><a name="p56610562610"></a>目的操作数，类型为LocalTensor，支持的TPosition为B2，<span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要512字节对齐。</span></p>
<p id="p66101129164712"><a name="p66101129164712"></a><a name="p66101129164712"></a>支持的数据类型为int8_t。</p>
<p id="p3610102994715"><a name="p3610102994715"></a><a name="p3610102994715"></a>数据连续排列顺序要求为小N大Z格式。</p>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p134685815130"><a name="p134685815130"></a><a name="p134685815130"></a>源操作数，类型为LocalTensor，支持的TPosition为B1，<span id="ph1158135141316"><a name="ph1158135141316"></a><a name="ph1158135141316"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p96501710201711"><a name="p96501710201711"></a><a name="p96501710201711"></a>支持的数据类型为int8_t。</p>
</td>
</tr>
<tr id="row1116635674818"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p8166195611488"><a name="p8166195611488"></a><a name="p8166195611488"></a>idx</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p10166195620486"><a name="p10166195620486"></a><a name="p10166195620486"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p1251653316137"><a name="p1251653316137"></a><a name="p1251653316137"></a>源操作数，类型为LocalTensor，支持的TPosition为B1，<span id="ph20802937111310"><a name="ph20802937111310"></a><a name="ph20802937111310"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p41661156204812"><a name="p41661156204812"></a><a name="p41661156204812"></a>支持的数据类型为int8_t。</p>
</td>
</tr>
<tr id="row1767431631917"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p667418162198"><a name="p667418162198"></a><a name="p667418162198"></a>loadDataParam</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p11675191610195"><a name="p11675191610195"></a><a name="p11675191610195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p1667541617193"><a name="p1667541617193"></a><a name="p1667541617193"></a>LoadData参数结构体，LoadData2DParams类型，详细说明参考<a href="Load2D.md#table8955841508">LoadData2DParams结构体内参数说明</a>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   repeat=0表示不执行。
-   每次迭代中的startIndex不能小于零。
-   不支持转置功能。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section6461234123118"></a>

详细用例请参考[MmadWithSparse](MmadWithSparse.md)。

