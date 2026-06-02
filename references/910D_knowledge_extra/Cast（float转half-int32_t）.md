# Cast（float转half、int32\_t）<a name="ZH-CN_TOPIC_0000002523304854"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

对标量的数据类型进行转换。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, typename U, RoundMode roundMode>
__aicore__ inline U Cast(T valueIn)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1017514222109"></a>
<table><thead align="left"><tr id="row717622218103"><th class="cellrowborder" valign="top" width="18.59%" id="mcps1.2.3.1.1"><p id="p1117617228103"><a name="p1117617228103"></a><a name="p1117617228103"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.41000000000001%" id="mcps1.2.3.1.2"><p id="p14176192219101"><a name="p14176192219101"></a><a name="p14176192219101"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row171761322131011"><td class="cellrowborder" valign="top" width="18.59%" headers="mcps1.2.3.1.1 "><p id="p1270210223183"><a name="p1270210223183"></a><a name="p1270210223183"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.41000000000001%" headers="mcps1.2.3.1.2 "><p id="p1986921081815"><a name="p1986921081815"></a><a name="p1986921081815"></a>valueIn的数据类型，支持float。</p>
</td>
</tr>
<tr id="row1418573831814"><td class="cellrowborder" valign="top" width="18.59%" headers="mcps1.2.3.1.1 "><p id="p198026488183"><a name="p198026488183"></a><a name="p198026488183"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="81.41000000000001%" headers="mcps1.2.3.1.2 "><p id="p02031955111811"><a name="p02031955111811"></a><a name="p02031955111811"></a>转换后的数据类型，支持half、int32_t。</p>
</td>
</tr>
<tr id="row125538426187"><td class="cellrowborder" valign="top" width="18.59%" headers="mcps1.2.3.1.1 "><p id="p727181791914"><a name="p727181791914"></a><a name="p727181791914"></a>roundMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.41000000000001%" headers="mcps1.2.3.1.2 "><p id="p885382310199"><a name="p885382310199"></a><a name="p885382310199"></a>精度转换处理模式，类型是RoundMode。</p>
<p id="p1885316233193"><a name="p1885316233193"></a><a name="p1885316233193"></a>RoundMode为枚举类型，用以控制精度转换处理模式，具体定义为：</p>
<a name="screen1085382310197"></a><a name="screen1085382310197"></a><pre class="screen" codetype="Cpp" id="screen1085382310197">enum class RoundMode {
    CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不取整
    CAST_RINT,      // rint，四舍六入五成双取整
    CAST_FLOOR,     // floor，向负无穷取整
    CAST_CEIL,      // ceil，向正无穷取整
    CAST_ROUND,     // round，四舍五入取整
    CAST_TRUNC,     // trunc，向零取整
    CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入
};</pre>
<p id="p16853152316196"><a name="p16853152316196"></a><a name="p16853152316196"></a>对于Cast，转换类型仅支持float转half(f322f16)与float转int32_t(f322s32)，相应支持的RoundMode如下：</p>
<a name="ul155971413112618"></a><a name="ul155971413112618"></a><ul id="ul155971413112618"><li>f322f16：CAST_ODD；</li><li>f322s32：CAST_ROUND、CAST_CEIL、CAST_FLOOR、CAST_RINT。</li></ul>
<p id="p685318232191"><a name="p685318232191"></a><a name="p685318232191"></a>Cast的精度转换规则具体可参考<a href="Cast.md#table235404962912">表1</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>valueIn</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>被转换数据类型的标量。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

U类型的valueIn。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```
float valueIn = 2.5;
// 输出数据valueOut：3， 2.5向上取整为3
int32_t valueOut = AscendC::Cast<float, int32_t, AscendC::RoundMode::CAST_ROUND>(valueIn);
```

