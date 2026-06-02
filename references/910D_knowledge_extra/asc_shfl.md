# asc\_shfl<a name="ZH-CN_TOPIC_0000002523344216"></a>

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

获取Warp内指定线程src\_lane输入的用于交换的var值；如果目标线程是非活跃状态，获取到寄存器中未初始化的值。其中，参数width用于划分Warp内线程的分组。参数width设置参与交换的32个线程的分组宽度，默认值为32，即所有线程分为1组。

在多个分组场景（width小于32）下，每个分组交换操作是独立的，当前线程获取本组内线程编号为src\_lane的var值。如果src\_lane小于分组宽度width，src\_lane是相对组内起始线程位置的逻辑LaneId；如果src\_lane大于等于width，src\_lane%width的结果是相对组内起始线程位置的逻辑LaneId。

例如，Warp内32个活跃线程调用asc\_shfl\(LaneId, 1, 16\)接口，每个线程的返回值为当前线程所在分组内线程编号为1的var值。

**图 1**  asc\_shfl结果示意图<a name="fig6403174235518"></a>  
<!-- img2text -->
```
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │16 │17 │18 │19 │20 │21 │22 │23 │24 │25 │26 │27 │28 │29 │30 │31 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
      └──── width = 16 ────┘      └──── width = 16 ────┘

┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │ 1 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │17 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
      ↑ 组0内取 src_lane=1 的 var=1                ↑ 组1内取 src_lane=1 的 var=17
```

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline int32_t asc_shfl(int32_t var, int32_t src_lane, int32_t width = warpSize)
```

```
__simt_callee__ inline uint32_t asc_shfl(uint32_t var, int32_t src_lane, int32_t width = warpSize)
```

```
__simt_callee__ inline float asc_shfl(float var, int32_t src_lane, int32_t width = warpSize)
```

```
__simt_callee__ inline int64_t asc_shfl(int64_t var, int32_t src_lane, int32_t width = warpSize)
```

```
__simt_callee__ inline uint64_t asc_shfl(uint64_t var, int32_t src_lane, int32_t width = warpSize)
```

```
__simt_callee__ inline half asc_shfl(half var, int32_t src_lane, int32_t width = warpSize)
```

```
__simt_callee__ inline half2 asc_shfl(half2 var, int32_t src_lane, int32_t width = warpSize)
```

## 参数说明<a name="section0866173114710"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="11.91%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.04%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="77.05%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>var</p>
</td>
<td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.05%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>线程用于交换的输入操作数。</p>
</td>
</tr>
<tr id="row1678544713120"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="p9785247183110"><a name="p9785247183110"></a><a name="p9785247183110"></a>src_lane</p>
</td>
<td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.2 "><p id="p7785547123115"><a name="p7785547123115"></a><a name="p7785547123115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.05%" headers="mcps1.2.4.1.3 "><p id="p27852472313"><a name="p27852472313"></a><a name="p27852472313"></a>期望获取的var值所在线程的LaneId。范围为[0, 32)。</p>
</td>
</tr>
<tr id="row1766035143117"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="p1466013510318"><a name="p1466013510318"></a><a name="p1466013510318"></a>width</p>
</td>
<td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.2 "><p id="p16601451183116"><a name="p16601451183116"></a><a name="p16601451183116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.05%" headers="mcps1.2.4.1.3 "><p id="p2660155123119"><a name="p2660155123119"></a><a name="p2660155123119"></a>Warp内参与交换的线程的分组宽度，默认值为32。width的取值范围为(0, 32]，width必须是2的倍数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

-   Warp内指定线程的var值
-   未初始化undefined的值

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用除half、half2类型之外的接口需要包含"simt\_api/device\_warp\_functions.h"头文件，使用half和half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件。

```
#include "simt_api/device_warp_functions.h" 
```

```
#include "simt_api/asc_fp16.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelShfl(__gm__ int32_t* dst)
{
    // asc_vf_call参数：dim3{1024, 1, 1}
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int32_t laneId = idx % 32;
    // 0-15线程返回值为1，16-31线程返回值为17
    int32_t result = asc_shfl(laneId, 1, 16);
    dst[idx] = result;
}
```

