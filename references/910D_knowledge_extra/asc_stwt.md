# asc\_stwt<a name="ZH-CN_TOPIC_0000002554424661"></a>

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

将指定数据存储到Global Memory的地址address中，并缓存至Data Cache和L2 Cache。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline void asc_stwt(__gm__ long int* address, long int val)
__simt_callee__ inline void asc_stwt(__gm__ unsigned long int* address, unsigned long int val)
__simt_callee__ inline void asc_stwt(__gm__ long long int* address, long long int val)
__simt_callee__ inline void asc_stwt(__gm__ unsigned long long int* address, unsigned long long int val)
__simt_callee__ inline void asc_stwt(__gm__ long2* address, long2 val)
__simt_callee__ inline void asc_stwt(__gm__ ulong2* address, ulong2 val)
__simt_callee__ inline void asc_stwt(__gm__ long4* address, long4 val)
__simt_callee__ inline void asc_stwt(__gm__ ulong4* address, ulong4 val)
__simt_callee__ inline void asc_stwt(__gm__ longlong2* address, longlong2 val)
__simt_callee__ inline void asc_stwt(__gm__ ulonglong2* address, ulonglong2 val)
__simt_callee__ inline void asc_stwt(__gm__ longlong4* address, longlong4 val)
__simt_callee__ inline void asc_stwt(__gm__ ulonglong4* address, ulonglong4 val)
__simt_callee__ inline void asc_stwt(__gm__ signed char* address, signed char val)
__simt_callee__ inline void asc_stwt(__gm__ unsigned char* address, unsigned char val)
__simt_callee__ inline void asc_stwt(__gm__ char2* address, char2 val)
__simt_callee__ inline void asc_stwt(__gm__ uchar2* address, uchar2 val)
__simt_callee__ inline void asc_stwt(__gm__ char4* address, char4 val)
__simt_callee__ inline void asc_stwt(__gm__ uchar4* address, uchar4 val)
__simt_callee__ inline void asc_stwt(__gm__ short* address, short val)
__simt_callee__ inline void asc_stwt(__gm__ unsigned short* address, unsigned short val)
__simt_callee__ inline void asc_stwt(__gm__ short2* address, short2 val)
__simt_callee__ inline void asc_stwt(__gm__ ushort2* address, ushort2 val)
__simt_callee__ inline void asc_stwt(__gm__ short4* address, short4 val)
__simt_callee__ inline void asc_stwt(__gm__ ushort4* address, ushort4 val)
__simt_callee__ inline void asc_stwt(__gm__ int* address, int val)
__simt_callee__ inline void asc_stwt(__gm__ unsigned int* address, unsigned int val)
__simt_callee__ inline void asc_stwt(__gm__ int2* address, int2 val)
__simt_callee__ inline void asc_stwt(__gm__ uint2* address, uint2 val)
__simt_callee__ inline void asc_stwt(__gm__ int4* address, int4 val)
__simt_callee__ inline void asc_stwt(__gm__ uint4* address, uint4 val)
__simt_callee__ inline void asc_stwt(__gm__ float* address, float val)
__simt_callee__ inline void asc_stwt(__gm__ float2* address, float2 val)
__simt_callee__ inline void asc_stwt(__gm__ float4* address, float4 val)
__simt_callee__ inline void asc_stwt(__gm__ bfloat16_t* address, bfloat16_t val)
__simt_callee__ inline void asc_stwt(__gm__ bfloat16x2_t* address, bfloat16x2_t val)
__simt_callee__ inline void asc_stwt(__gm__ half* address, half2 val)
__simt_callee__ inline void asc_stwt(__gm__ half2* address, half2 val)
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
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p935122215820"><a name="p935122215820"></a><a name="p935122215820"></a>address</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p135122217814"><a name="p135122217814"></a><a name="p135122217814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p03511122582"><a name="p03511122582"></a><a name="p03511122582"></a><span id="ph133519224816"><a name="ph133519224816"></a><a name="ph133519224816"></a>Global Memory</span>的地址。</p>
</td>
</tr>
<tr id="row1346541814815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6255648487"><a name="p6255648487"></a><a name="p6255648487"></a>val</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p182559481483"><a name="p182559481483"></a><a name="p182559481483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p62551148687"><a name="p62551148687"></a><a name="p62551148687"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

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
__simt_vf__ __launch_bounds__(1024) inline void kernel_asc_stwt(__gm__ float* src, __gm__ float* val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    asc_stwt(src + idx, val[idx]);
}
```

