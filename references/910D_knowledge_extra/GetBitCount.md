# GetBitCount<a name="ZH-CN_TOPIC_0000002523303836"></a>

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

获取一个uint64\_t类型数字的二进制中0或者1的个数。

## 函数原型<a name="section620mcpsimp"></a>

```
template <int countValue> 
__aicore__ inline int64_t GetBitCount(uint64_t valueIn)
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
<tbody><tr id="row171761322131011"><td class="cellrowborder" valign="top" width="18.59%" headers="mcps1.2.3.1.1 "><p id="p217692213105"><a name="p217692213105"></a><a name="p217692213105"></a>countValue</p>
</td>
<td class="cellrowborder" valign="top" width="81.41000000000001%" headers="mcps1.2.3.1.2 "><p id="p1717692211109"><a name="p1717692211109"></a><a name="p1717692211109"></a>指定统计0还是统计1的个数。</p>
<p id="p5176102210107"><a name="p5176102210107"></a><a name="p5176102210107"></a>只能输入0或1。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>valueIn</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p57631244105711"><a name="p57631244105711"></a><a name="p57631244105711"></a>被统计的二进制数字。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

valueIn中0或者1的个数。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

```
uint64_t valueIn = 0xffff;    // 二进制格式中有16个1
constexpr int countValue = 1;    // 统计valueIn二进制格式中1的个数
// 输出数据oneCount: 16
int64_t oneCount = AscendC::GetBitCount<countValue>(valueIn);
```

