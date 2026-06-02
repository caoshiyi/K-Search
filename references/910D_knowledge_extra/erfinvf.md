# erfinvf<a name="ZH-CN_TOPIC_0000002554423823"></a>

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

## 功能说明<a name="section0572318193011"></a>

获取输入数据的逆误差函数值。

<!-- img2text -->
$$
\operatorname{erf}^{-1}(x)=
\begin{cases}
w-\dfrac{2.5}{0.0891314744949340820313+0.00660224142593383789063w+0.00059066917339563369751w^{2}+0.00281690806580870032311w^{3}+0.00292105684854653835252w^{4}+0.00983774803400749444914w^{5}}, & w<5 \\
\sqrt{w}-\dfrac{3}{2.71046216948973947962+0.00567095089082665473866w+0.00028981997697683918617w^{2}+0.0012426609473880784386w^{3}+0.000026532189526576123093w^{4}+0.00000301581553508235416007w^{5}}, & w\geq 5
\end{cases}
$$

$$
\text{其中 } w=-\ln\bigl((1-x)(1+x)\bigr)
$$

## 函数原型<a name="section1188715215306"></a>

```
__simt_callee__ inline float erfinvf(float x)
```

## 参数说明<a name="section19580242193016"></a>

**表 1**  函数形参说明

<a name="table168351351182212"></a>
<table><thead align="left"><tr id="row1483655182212"><th class="cellrowborder" valign="top" width="11.49114911491149%" id="mcps1.2.4.1.1"><p id="p8836351122218"><a name="p8836351122218"></a><a name="p8836351122218"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.31123112311231%" id="mcps1.2.4.1.2"><p id="p5836195114227"><a name="p5836195114227"></a><a name="p5836195114227"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="76.1976197619762%" id="mcps1.2.4.1.3"><p id="p18836125142213"><a name="p18836125142213"></a><a name="p18836125142213"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row6836145162220"><td class="cellrowborder" valign="top" width="11.49114911491149%" headers="mcps1.2.4.1.1 "><p id="p14836951102214"><a name="p14836951102214"></a><a name="p14836951102214"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="12.31123112311231%" headers="mcps1.2.4.1.2 "><p id="p7836851132212"><a name="p7836851132212"></a><a name="p7836851132212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.1976197619762%" headers="mcps1.2.4.1.3 "><p id="p7836115172214"><a name="p7836115172214"></a><a name="p7836115172214"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section63891446143019"></a>

-   当x为0时，返回值为0。
-   当x为nan时，返回值为nan。
-   当x为1时，返回值为inf。
-   当x为-1时，返回值为-inf。
-   当x∉\[-1,1\]时，返回nan。

## 约束说明<a name="section48951449183010"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"simt\_api/math\_functions.h"头文件。

```
#include "simt_api/math_functions.h"
```

## 调用示例<a name="section1598053163014"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelErfc(__gm__ float* x, __gm__ float* y)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    y[idx] = erfinvf(x[idx]);
}
```

