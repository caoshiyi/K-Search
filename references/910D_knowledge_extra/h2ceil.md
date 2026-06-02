# h2ceil<a name="ZH-CN_TOPIC_0000002523304734"></a>

## 产品支持情况<a name="section1063765432310"></a>

<a name="table0637165419234"></a>
<table><thead align="left"><tr id="row1363725422312"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p12637454152311"><a name="p12637454152311"></a><a name="p12637454152311"></a><span id="ph19637105482311"><a name="ph19637105482311"></a><a name="ph19637105482311"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p11637754162312"><a name="p11637754162312"></a><a name="p11637754162312"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row126372549231"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1563715417234"><a name="p1563715417234"></a><a name="p1563715417234"></a><span id="ph1663735419238"><a name="ph1663735419238"></a><a name="ph1663735419238"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p156375548230"><a name="p156375548230"></a><a name="p156375548230"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section196387547239"></a>

获取大于或等于输入数据各元素的最小整数值。

## 函数原型<a name="section2638105482318"></a>

```
__simt_callee__ inline half2 h2ceil(half2 x)
```

```
__simt_callee__ inline bfloat16x2_t h2ceil(bfloat16x2_t x)
```

## 参数说明<a name="section1263815548236"></a>

**表 1**  参数说明

<a name="table14638125414231"></a>
<table><thead align="left"><tr id="row126390542238"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p11639854162319"><a name="p11639854162319"></a><a name="p11639854162319"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p3639105418234"><a name="p3639105418234"></a><a name="p3639105418234"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p136391454162314"><a name="p136391454162314"></a><a name="p136391454162314"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row163918541235"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p13639145412319"><a name="p13639145412319"></a><a name="p13639145412319"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p9639105417232"><a name="p9639105417232"></a><a name="p9639105417232"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p363915410234"><a name="p363915410234"></a><a name="p363915410234"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section106391654182320"></a>

大于或等于输入数据各元素的最小整数值。特别场景说明如下：

-   当输入元素为nan时，返回值为nan。
-   当输入元素为inf时，返回值为inf。
-   当输入元素为-inf时，返回值为-inf。

## 约束说明<a name="section5639554162310"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section17639115402315"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelCeil(__gm__ half2* dst, __gm__ half2* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = h2ceil(x[idx]);
}
```

