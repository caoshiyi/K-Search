# min<a name="ZH-CN_TOPIC_0000002523344960"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section7376114729"></a>

比较相同数据类型的两个数，返回最小值。

## 函数原型<a name="section126881859101617"></a>

```
template <typename T, typename U>
__aicore__ inline T min(const T src0, const U src1)
```

## 参数说明<a name="section121562129312"></a>

**表 1**  模板参数说明

<a name="table729818506422"></a>
<table><thead align="left"><tr id="row11299950204217"><th class="cellrowborder" valign="top" width="19.18%" id="mcps1.2.3.1.1"><p id="p1029955044218"><a name="p1029955044218"></a><a name="p1029955044218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.82000000000001%" id="mcps1.2.3.1.2"><p id="p1121663111288"><a name="p1121663111288"></a><a name="p1121663111288"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12299165018421"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p1329915004219"><a name="p1329915004219"></a><a name="p1329915004219"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p154563172254"><a name="p154563172254"></a><a name="p154563172254"></a>输入数据src0的数据类型。当前支持的数据类型为</p>
<p id="p1398105017233"><a name="p1398105017233"></a><a name="p1398105017233"></a>bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
</td>
</tr>
<tr id="row5299125054217"><td class="cellrowborder" valign="top" width="19.18%" headers="mcps1.2.3.1.1 "><p id="p9777142884312"><a name="p9777142884312"></a><a name="p9777142884312"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p38871730112511"><a name="p38871730112511"></a><a name="p38871730112511"></a>输入数据src1的数据类型。当前支持的数据类型为</p>
<p id="p88871130192513"><a name="p88871130192513"></a><a name="p88871130192513"></a>bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t、float、int64_t、uint64_t。</p>
<p id="p10308155243615"><a name="p10308155243615"></a><a name="p10308155243615"></a>预留类型，当前必须与T保持一致。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.69%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.54%" id="mcps1.2.4.1.3"><p id="p1583482513439"><a name="p1583482513439"></a><a name="p1583482513439"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row2654147154519"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1557920894512"><a name="p1557920894512"></a><a name="p1557920894512"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p55792824511"><a name="p55792824511"></a><a name="p55792824511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p9579480457"><a name="p9579480457"></a><a name="p9579480457"></a>源操作数。参与比较的输入。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p7123131611117"><a name="p7123131611117"></a><a name="p7123131611117"></a>src1</p>
</td>
<td class="cellrowborder" valign="top" width="9.69%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.54%" headers="mcps1.2.4.1.3 "><p id="p963863814519"><a name="p963863814519"></a><a name="p963863814519"></a>源操作数。参与比较的输入。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1564510486314"></a>

两个源操作数的数据类型必须相同。

## 返回值说明<a name="section62431148556"></a>

两个输入数据中的最小值。

## 调用示例<a name="section1193764916212"></a>

```
int64_t src0 = 1;
int64_t src1 = 2;
 
int64_t result = AscendC::Std::min(src0, src1); 
// result: 1
```

