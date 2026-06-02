# remquof<a name="ZH-CN_TOPIC_0000002523304680"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取输入数据x除以y的余数。求余数时，商取最接近x除以y浮点数结果的整数，当x除以y的浮点数结果与左右最接近的整数距离相等时，商取偶数，同时将商赋值给指针变量quo。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline float remquof(float x, float y, int *quo)
```

## 参数说明<a name="section0866173114710"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="14.97%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.62%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="14.97%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="13.62%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>源操作数。</p>
</td>
</tr>
<tr id="row201811549422"><td class="cellrowborder" valign="top" width="14.97%" headers="mcps1.2.4.1.1 "><p id="p13716205910427"><a name="p13716205910427"></a><a name="p13716205910427"></a>y</p>
</td>
<td class="cellrowborder" valign="top" width="13.62%" headers="mcps1.2.4.1.2 "><p id="p11716165913428"><a name="p11716165913428"></a><a name="p11716165913428"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p9716125974211"><a name="p9716125974211"></a><a name="p9716125974211"></a>源操作数。</p>
</td>
</tr>
<tr id="row137311457164215"><td class="cellrowborder" valign="top" width="14.97%" headers="mcps1.2.4.1.1 "><p id="p755911014437"><a name="p755911014437"></a><a name="p755911014437"></a>quo</p>
</td>
<td class="cellrowborder" valign="top" width="13.62%" headers="mcps1.2.4.1.2 "><p id="p1055914010431"><a name="p1055914010431"></a><a name="p1055914010431"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p85594010433"><a name="p85594010433"></a><a name="p85594010433"></a>除法运算的商。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

输入数据x除以y的余数。

-   x，y任意一个为inf、-inf、nan时，返回值为nan。

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"simt\_api/math\_functions.h"头文件。

```
#include "simt_api/math_functions.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelRemQuo(__gm__ float* dst, __gm__ float* x, __gm__ float* y, __gm__ int* quo)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int q = -1;
    dst[idx] = remquof(x[idx], y[idx], &q);
    quo[idx] = q;
}
```

