# cbrtf<a name="ZH-CN_TOPIC_0000002523303900"></a>

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

## 功能说明<a name="section1304164365714"></a>

获取输入数据x的立方根。

<!-- img2text -->
$$\operatorname{cbrtf}(x)=\sqrt[3]{x}$$

## 函数原型<a name="section349316241116"></a>

```
__simt_callee__ inline float cbrtf(float x)
```

## 参数说明<a name="section188691427232"></a>

**表 1**  函数形参说明

<a name="table2341146641"></a>
<table><thead align="left"><tr id="row10341946942"><th class="cellrowborder" valign="top" width="11.42%" id="mcps1.2.4.1.1"><p id="p135154617419"><a name="p135154617419"></a><a name="p135154617419"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.540000000000001%" id="mcps1.2.4.1.2"><p id="p6351146341"><a name="p6351146341"></a><a name="p6351146341"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76.03999999999999%" id="mcps1.2.4.1.3"><p id="p1891841715512"><a name="p1891841715512"></a><a name="p1891841715512"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row735134619415"><td class="cellrowborder" valign="top" width="11.42%" headers="mcps1.2.4.1.1 "><p id="p6351246544"><a name="p6351246544"></a><a name="p6351246544"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="12.540000000000001%" headers="mcps1.2.4.1.2 "><p id="p43520462411"><a name="p43520462411"></a><a name="p43520462411"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.03999999999999%" headers="mcps1.2.4.1.3 "><p id="p89187171352"><a name="p89187171352"></a><a name="p89187171352"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section192141211712"></a>

x的立方根。

-   当x为0时，返回值为0。
-   当x为nan时，返回值为nan。
-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为-inf。

## 约束说明<a name="section97781637398"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"simt\_api/math\_functions.h"头文件。

```
#include "simt_api/math_functions.h"
```

## 调用示例<a name="section105610512156"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelCbrt(__gm__ float* x, __gm__ float* y)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    y[idx] = cbrtf(x[idx]);
}
```

