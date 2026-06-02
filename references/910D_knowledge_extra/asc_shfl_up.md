# asc\_shfl\_up<a name="ZH-CN_TOPIC_0000002523343860"></a>

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

获取Warp内当前线程向前偏移delta（当前线程LaneId-delta）的线程输入的用于交换的var值；如果目标线程是非活跃状态，获取到寄存器中未初始化的值。其中，参数width用于划分Warp内线程的分组。参数width设置参与交换的32个线程的分组宽度，默认值为32，即所有线程分为1组。

在多个分组场景（width小于32）下，每个分组交换操作是独立的，每个线程获取本组内当前线程向前偏移delta的线程的var值。如果当前线程向前偏移delta的线程编号，即LaneId-delta，小于所在分组的起始LaneId，则返回当前线程的var值。

例如，Warp内32个活跃线程调用asc\_shfl\_up\(LaneId, 2, 16\)接口，每个线程的返回值为当前线程LaneId-2对应线程的var值，或者当前线程的var值。

**图 1**  asc\_shfl\_up结果示意图<a name="fig147375361415"></a>  
<!-- img2text -->
```text
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │16 │17 │18 │19 │20 │21 │22 │23 │24 │25 │26 │27 │28 │29 │30 │31 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ↓   ↓   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙    ↓   ↓   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙   ↙
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │16 │17 │16 │17 │18 │19 │20 │21 │22 │23 │24 │25 │26 │27 │28 │29 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
```

说明:
- 上排表示输入 LaneId：0~31
- 下排表示 `asc_shfl_up(LaneId, 2, 16)` 的返回结果
- 分组宽度 `width=16`，因此分为两组：0~15、16~31
- 每个线程获取本组内 `LaneId-2` 对应线程的值；若 `LaneId-2` 小于本组起始 LaneId，则返回当前线程自己的值
- 对应关系可写为：
  - 第1组(0~15): `0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13`
  - 第2组(16~31): `16, 17, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29`

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline int32_t asc_shfl_up(int32_t var, uint32_t delta, int32_t width = warpSize)
```

```
__simt_callee__ inline uint32_t asc_shfl_up(uint32_t var, uint32_t delta, int32_t width = warpSize)
```

```
__simt_callee__ inline float asc_shfl_up(float var, uint32_t delta, int32_t width = warpSize)
```

```
__simt_callee__ inline int64_t asc_shfl_up(int64_t var, uint32_t delta, int32_t width = warpSize)
```

```
__simt_callee__ inline uint64_t asc_shfl_up(uint64_t var, uint32_t delta, int32_t width = warpSize)
```

```
__simt_callee__ inline half asc_shfl_up(half var, uint32_t delta, int32_t width = warpSize)
```

```
__simt_callee__ inline half2 asc_shfl_up(half2 var, uint32_t delta, int32_t width = warpSize)
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
<tr id="row1678544713120"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="p9785247183110"><a name="p9785247183110"></a><a name="p9785247183110"></a>delta</p>
</td>
<td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.2 "><p id="p7785547123115"><a name="p7785547123115"></a><a name="p7785547123115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.05%" headers="mcps1.2.4.1.3 "><p id="p27852472313"><a name="p27852472313"></a><a name="p27852472313"></a>期望获取的var值所在线程相对当前线程的向前偏移值。范围为[0, 32)，且小于width。</p>
</td>
</tr>
<tr id="row1766035143117"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="p1466013510318"><a name="p1466013510318"></a><a name="p1466013510318"></a>width</p>
</td>
<td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.2.4.1.2 "><p id="p16601451183116"><a name="p16601451183116"></a><a name="p16601451183116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="77.05%" headers="mcps1.2.4.1.3 "><p id="p917212139619"><a name="p917212139619"></a><a name="p917212139619"></a>Warp内参与交换的线程的分组宽度，默认值为32。width的取值范围为(0, 32]，width必须是2的倍数。</p>
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
__simt_vf__ __launch_bounds__(1024) inline void KernelShflUp(__gm__ int32_t* dst)
{
     // asc_vf_call参数：dim3{1024, 1, 1}
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     int32_t laneId = idx % 32;
     // 0-15线程返回值分别为{0,1,0,1,2,3,4,5,6,7,8,9,10,11,12,13}
     // 16-31线程返回值为{16,17,16,17,18,19,20,21,22,23,24,25,26,27,28,29}

     int32_t result = asc_shfl_up(laneId, 2, 16);
     dst[idx] = result;
} 

// asc_shfl_up实现reducesum
__simt_vf__ __launch_bounds__(1024) inline void KernelShflUpReduceSum(__gm__ int32_t* dst)
{
     // asc_vf_call参数：dim3{1024, 1, 1}
     int idx = threadIdx.x + blockIdx.x * blockDim.x;
     int32_t laneId = idx % 32;
     int32_t value = laneId;

     value += asc_shfl_up(value, 16, 31); // 16
     value += asc_shfl_up(value, 8, 31);  // 8
     value += asc_shfl_up(value, 4, 31);  // 4
     value += asc_shfl_up(value, 2, 31);  // 2
     value += asc_shfl_up(value, 1, 31);  // 1

     dst[idx] = value;
}
```

