# hfloor<a name="ZH-CN_TOPIC_0000002554343593"></a>

## 产品支持情况<a name="section930515579174"></a>

<a name="table11305257201711"></a>
<table><thead align="left"><tr id="row23051357151719"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p14305135711176"><a name="p14305135711176"></a><a name="p14305135711176"></a><span id="ph1830585711719"><a name="ph1830585711719"></a><a name="ph1830585711719"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p130515715178"><a name="p130515715178"></a><a name="p130515715178"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row143051357191716"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17305957141710"><a name="p17305957141710"></a><a name="p17305957141710"></a><span id="ph1530517579171"><a name="ph1530517579171"></a><a name="ph1530517579171"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1630535714172"><a name="p1630535714172"></a><a name="p1630535714172"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section14306957131718"></a>

获取小于或等于输入数据的最大整数值。

## 函数原型<a name="section1530635717178"></a>

```
__simt_callee__ inline half hfloor(half x)
```

```
__simt_callee__ inline bfloat16_t hfloor(bfloat16_t x)
```

## 参数说明<a name="section1630615571172"></a>

**表 1**  参数说明

<a name="table5306195721712"></a>
<table><thead align="left"><tr id="row1930665791710"><th class="cellrowborder" valign="top" width="16.009999999999998%" id="mcps1.2.4.1.1"><p id="p1230695771712"><a name="p1230695771712"></a><a name="p1230695771712"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.58%" id="mcps1.2.4.1.2"><p id="p153063574176"><a name="p153063574176"></a><a name="p153063574176"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p9306957131716"><a name="p9306957131716"></a><a name="p9306957131716"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row163060577171"><td class="cellrowborder" valign="top" width="16.009999999999998%" headers="mcps1.2.4.1.1 "><p id="p163061657181719"><a name="p163061657181719"></a><a name="p163061657181719"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="12.58%" headers="mcps1.2.4.1.2 "><p id="p12307105731719"><a name="p12307105731719"></a><a name="p12307105731719"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p130735741710"><a name="p130735741710"></a><a name="p130735741710"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section203077579176"></a>

小于或等于输入数据的最大整数值。特别场景说明如下：

-   当x为nan时，返回值为nan。
-   当x为inf时，返回值为inf。
-   当x为-inf时，返回值为-inf。

## 约束说明<a name="section9307205718178"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section1830745716173"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelIsFinite(__gm__ half* dst, __gm__ half* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = hfloor(x[idx]);
}
```

