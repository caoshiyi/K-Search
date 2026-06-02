# h2rint<a name="ZH-CN_TOPIC_0000002554424201"></a>

## 产品支持情况<a name="section9366205831018"></a>

<a name="table10366125861020"></a>
<table><thead align="left"><tr id="row236685851016"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1136675820109"><a name="p1136675820109"></a><a name="p1136675820109"></a><span id="ph036625812108"><a name="ph036625812108"></a><a name="ph036625812108"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p1436620581108"><a name="p1436620581108"></a><a name="p1436620581108"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row11366958101020"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p153663581108"><a name="p153663581108"></a><a name="p153663581108"></a><span id="ph1366258171013"><a name="ph1366258171013"></a><a name="ph1366258171013"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p5366458181018"><a name="p5366458181018"></a><a name="p5366458181018"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section2367358151015"></a>

获取与输入数据各元素最接近的整数，若存在两个同样接近的整数，则取其中的偶数。

## 函数原型<a name="section836717581104"></a>

```
__simt_callee__ inline half2 h2rint(half2 x)
```

```
__simt_callee__ inline bfloat16x2_t h2rint(bfloat16x2_t x)
```

## 参数说明<a name="section83671458111014"></a>

**表 1**  参数说明

<a name="table5367185818103"></a>
<table><thead align="left"><tr id="row20367125810108"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p1136711581100"><a name="p1136711581100"></a><a name="p1136711581100"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p036710583105"><a name="p036710583105"></a><a name="p036710583105"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p133676583101"><a name="p133676583101"></a><a name="p133676583101"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row183671758101017"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p193677589101"><a name="p193677589101"></a><a name="p193677589101"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p15367358121012"><a name="p15367358121012"></a><a name="p15367358121012"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1636714586104"><a name="p1636714586104"></a><a name="p1636714586104"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section136795841013"></a>

与输入各元素最接近的整数值。特别场景说明如下：

-   当输入元素为0时，返回值为0。
-   当输入元素为0.5时，返回值为0。
-   当输入元素为1.5时，返回值为2。
-   当输入元素为nan时，返回值为nan。
-   当输入元素为inf时，返回值为inf。
-   当输入元素为-inf时，返回值为-inf。

## 约束说明<a name="section1036795816103"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section8367165841015"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelRint(__gm__ half2* dst, __gm__ half2* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = h2rint(x[idx]);
}
```

