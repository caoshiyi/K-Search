# make\_longlong3<a name="ZH-CN_TOPIC_0000002523343890"></a>

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

使用给定的三个long long int类型的数据创建一个longlong3类型的向量。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline longlong3 make_longlong3(long long int x, long long int y, long long int z)
```

## 参数说明<a name="section0866173114710"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p53271211078"><a name="p53271211078"></a><a name="p53271211078"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p19327311572"><a name="p19327311572"></a><a name="p19327311572"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1332751473"><a name="p1332751473"></a><a name="p1332751473"></a>源操作数。</p>
</td>
</tr>
<tr id="row99291135121715"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6327617718"><a name="p6327617718"></a><a name="p6327617718"></a>y</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p932781271"><a name="p932781271"></a><a name="p932781271"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p143271811177"><a name="p143271811177"></a><a name="p143271811177"></a>源操作数。</p>
</td>
</tr>
<tr id="row152711101173"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p428251115313"><a name="p428251115313"></a><a name="p428251115313"></a>z</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1928271173110"><a name="p1928271173110"></a><a name="p1928271173110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p12282911193111"><a name="p12282911193111"></a><a name="p12282911193111"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

由三个long long int类型数字构成的新longlong3向量。

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section7131139151810"></a>

使用该接口需要包含"simt\_api/vector\_functions.h"头文件。

```
#include "simt_api/vector_functions.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void kernel_make_longlong3(__gm__ longlong3* dst, __gm__ long long int* x, __gm__ long long int* y, __gm__ long long int* z)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = make_longlong3(x[idx], y[idx], z[idx]);
}
```

