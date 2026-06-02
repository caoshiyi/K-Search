# htrunc<a name="ZH-CN_TOPIC_0000002523304926"></a>

## 产品支持情况<a name="section44291952112710"></a>

<a name="table642935232714"></a>
<table><thead align="left"><tr id="row144291352202714"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p14429145242716"><a name="p14429145242716"></a><a name="p14429145242716"></a><span id="ph742915524275"><a name="ph742915524275"></a><a name="ph742915524275"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p4429115214273"><a name="p4429115214273"></a><a name="p4429115214273"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row842913521275"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p142919527279"><a name="p142919527279"></a><a name="p142919527279"></a><span id="ph104291352102712"><a name="ph104291352102712"></a><a name="ph104291352102712"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p17429165252719"><a name="p17429165252719"></a><a name="p17429165252719"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section1443095242715"></a>

获取对输入数据的浮点数截断后的整数。

## 函数原型<a name="section1343020529278"></a>

```
__simt_callee__ inline half htrunc(half x)
```

```
__simt_callee__ inline bfloat16_t htrunc(bfloat16_t x)
```

## 参数说明<a name="section13430165219276"></a>

**表 1**  参数说明

<a name="table843010521273"></a>
<table><thead align="left"><tr id="row143018525278"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p10430165282716"><a name="p10430165282716"></a><a name="p10430165282716"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p17430652112718"><a name="p17430652112718"></a><a name="p17430652112718"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p1343075292711"><a name="p1343075292711"></a><a name="p1343075292711"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row13430752112718"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p54301952132710"><a name="p54301952132710"></a><a name="p54301952132710"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p0430952122719"><a name="p0430952122719"></a><a name="p0430952122719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p104308521274"><a name="p104308521274"></a><a name="p104308521274"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section143016527279"></a>

输入数据浮点数截断后的整数。特别场景说明如下：

-   当x为nan时，返回值为nan。
-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为-inf。

## 约束说明<a name="section144307526277"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section543014526279"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelTrunc(__gm__ half* dst, __gm__ half* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = htrunc(x[idx]);
}
```

