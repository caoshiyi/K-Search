# \_\_habs<a name="ZH-CN_TOPIC_0000002554344147"></a>

## 产品支持情况<a name="section17122773610"></a>

<a name="table81257143614"></a>
<table><thead align="left"><tr id="row12121733616"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p91214711361"><a name="p91214711361"></a><a name="p91214711361"></a><span id="ph91211716367"><a name="ph91211716367"></a><a name="ph91211716367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p16121673367"><a name="p16121673367"></a><a name="p16121673367"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1912676362"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p41218723611"><a name="p41218723611"></a><a name="p41218723611"></a><span id="ph21212712360"><a name="ph21212712360"></a><a name="ph21212712360"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1112167143614"><a name="p1112167143614"></a><a name="p1112167143614"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section20136719367"></a>

获取输入数据的绝对值。

<!-- img2text -->
$$
|x| =
\begin{cases}
x, & x \geq 0 \\
-x, & x < 0
\end{cases}
$$

## 函数原型<a name="section111357153618"></a>

```
__simt_callee__ inline half __habs(half x)
```

```
__simt_callee__ inline bfloat16_t __habs(bfloat16_t x)
```

## 参数说明<a name="section4131179368"></a>

**表 1**  参数说明

<a name="table14135710361"></a>
<table><thead align="left"><tr id="row141312715366"><th class="cellrowborder" valign="top" width="16.32%" id="mcps1.2.4.1.1"><p id="p01347113610"><a name="p01347113610"></a><a name="p01347113610"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.27%" id="mcps1.2.4.1.2"><p id="p131314783615"><a name="p131314783615"></a><a name="p131314783615"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p181377103615"><a name="p181377103615"></a><a name="p181377103615"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1113187113613"><td class="cellrowborder" valign="top" width="16.32%" headers="mcps1.2.4.1.1 "><p id="p19131279363"><a name="p19131279363"></a><a name="p19131279363"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="12.27%" headers="mcps1.2.4.1.2 "><p id="p31387103619"><a name="p31387103619"></a><a name="p31387103619"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p12131970365"><a name="p12131970365"></a><a name="p12131970365"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section2131271361"></a>

输入数据的绝对值。

-   当x为±0时，返回值为+0。
-   当x为nan时，返回值为nan。
-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为inf。

## 约束说明<a name="section201318743619"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section61316733615"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelAbs(__gm__ half* dst, __gm__ half* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = __habs(x[idx]);
}
```

