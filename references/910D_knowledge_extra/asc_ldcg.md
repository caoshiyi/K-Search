# asc\_ldcg<a name="ZH-CN_TOPIC_0000002523344790"></a>

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

从L2 Cache加载缓存的数据，如果缓存命中，则直接返回数据。若未命中，则从Global Memory地址预加载数据缓存至L2 Cache，并返回数据。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline long int asc_ldcg(__gm__ long int* address)
__simt_callee__ inline unsigned long int asc_ldcg(__gm__ unsigned long int* address)
__simt_callee__ inline long long int asc_ldcg(__gm__ long long int* address)
__simt_callee__ inline unsigned long long int asc_ldcg(__gm__ unsigned long long int* address)
__simt_callee__ inline long2 asc_ldcg(__gm__ long2* address)
__simt_callee__ inline ulong2 asc_ldcg(__gm__ ulong2* address)
__simt_callee__ inline long4 asc_ldcg(__gm__ long4* address)
__simt_callee__ inline ulong4 asc_ldcg(__gm__ ulong4* address)
__simt_callee__ inline longlong2 asc_ldcg(__gm__ longlong2* address)
__simt_callee__ inline ulonglong2 asc_ldcg(__gm__ ulonglong2* address)
__simt_callee__ inline longlong4 asc_ldcg(__gm__ longlong4* address)
__simt_callee__ inline ulonglong4 asc_ldcg(__gm__ ulonglong4* address)
__simt_callee__ inline signed char asc_ldcg(__gm__ signed char* address)
__simt_callee__ inline unsigned char asc_ldcg(__gm__ unsigned char* address)
__simt_callee__ inline char2 asc_ldcg(__gm__ char2* address)
__simt_callee__ inline uchar2 asc_ldcg(__gm__ uchar2* address)
__simt_callee__ inline char4 asc_ldcg(__gm__ char4* address)
__simt_callee__ inline uchar4 asc_ldcg(__gm__ uchar4* address)
__simt_callee__ inline short asc_ldcg(__gm__ short* address)
__simt_callee__ inline unsigned short asc_ldcg(__gm__ unsigned short* address)
__simt_callee__ inline short2 asc_ldcg(__gm__ short2* address)
__simt_callee__ inline ushort2 asc_ldcg(__gm__ ushort2* address)
__simt_callee__ inline short4 asc_ldcg(__gm__ short4* address)
__simt_callee__ inline ushort4 asc_ldcg(__gm__ ushort4* address)
__simt_callee__ inline int asc_ldcg(__gm__ int* address)
__simt_callee__ inline unsigned int asc_ldcg(__gm__ unsigned int* address)
__simt_callee__ inline int2 asc_ldcg(__gm__ int2* address)
__simt_callee__ inline uint2 asc_ldcg(__gm__ uint2* address)
__simt_callee__ inline int4 asc_ldcg(__gm__ int4* address)
__simt_callee__ inline uint4 asc_ldcg(__gm__ uint4* address)
__simt_callee__ inline float asc_ldcg(__gm__ float* address)
__simt_callee__ inline float2 asc_ldcg(__gm__ float2* address)
__simt_callee__ inline float4 asc_ldcg(__gm__ float4* address)
__simt_callee__ inline bfloat16_t asc_ldcg(__gm__ bfloat16_t* address)
__simt_callee__ inline bfloat16x2_t asc_ldcg(__gm__ bfloat16x2_t* address)
__simt_callee__ inline half asc_ldcg(__gm__ half* address)
__simt_callee__ inline half2 asc_ldcg(__gm__ half2* address)
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>address</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a><span id="ph128774221307"><a name="ph128774221307"></a><a name="ph128774221307"></a>Global Memory</span>的地址。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

L2 Cache中的缓存数据或输入的Global Memory地址上的数据。

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section7131139151810"></a>

使用除half、half2、bfloat16\_t、bfloat16x2\_t类型之外的接口需要包含"simt\_api/device\_functions.h"头文件，使用half和half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t和bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/device_functions.h"
```

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void kernel_asc_ldcg(__gm__ float* dst, __gm__ float* src)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = asc_ldcg(src + idx);
}
```

