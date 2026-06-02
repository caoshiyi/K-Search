# Cast（float转bfloat16\_t）<a name="ZH-CN_TOPIC_0000002554423751"></a>

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

float类型标量数据转换成bfloat16\_t类型标量数据。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline bfloat16_t Cast(const float& fVal)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.38163816381638%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.861086108610861%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.75727572757276%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row23264162912"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p941862411595"><a name="p941862411595"></a><a name="p941862411595"></a>fVal</p>
</td>
<td class="cellrowborder" valign="top" width="10.861086108610861%" headers="mcps1.2.4.1.2 "><p id="p941792465918"><a name="p941792465918"></a><a name="p941792465918"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.75727572757276%" headers="mcps1.2.4.1.3 "><p id="p950834622215"><a name="p950834622215"></a><a name="p950834622215"></a>float类型标量数据。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

转换后的bfloat16\_t类型标量数据。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section19372434133520"></a>

```
float m = 3.0f;
bfloat16_t n = AscendC::Cast(m);  // n = 3.0  bfloat16_t和 float指数和尾数表达不同，可以通过截断进行转换
```

