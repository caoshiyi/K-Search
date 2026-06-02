# asc\_atomic\_cas<a name="ZH-CN_TOPIC_0000002554344459"></a>

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

对Unified Buffer或Global Memory上address的数值进行原子比较赋值操作，如果address上的数值等于指定数值compare，则对address赋值为指定数值val，否则address的数值不变。

## 函数原型<a name="section620mcpsimp"></a>

-   Unified Buffer数据的原子比较赋值操作

    ```
    __simt_callee__ inline float asc_atomic_cas(__ubuf__ float *address, float compare, float val)
    ```

    ```
    __simt_callee__ inline int32_t asc_atomic_cas(__ubuf__ int32_t *address, int32_t compare, int32_t val)
    ```

    ```
    __simt_callee__ inline uint32_t asc_atomic_cas(__ubuf__ uint32_t *address, uint32_t compare, uint32_t val)
    ```

    ```
    __simt_callee__ inline half2 asc_atomic_cas(__ubuf__ half2 *address, half2 compare, half2 val)
    ```

    ```
    __simt_callee__ inline bfloat16x2_t asc_atomic_cas(__ubuf__ bfloat16x2_t *address, bfloat16x2_t compare, bfloat16x2_t val)
    ```

-   Global Memory数据的原子比较赋值操作

    ```
    __simt_callee__ inline float asc_atomic_cas(__gm__ float *address, float compare, float val)
    ```

    ```
    __simt_callee__ inline int32_t asc_atomic_cas(__gm__ int32_t *address, int32_t compare, int32_t val)
    ```

    ```
    __simt_callee__ inline uint32_t asc_atomic_cas(__gm__ uint32_t *address, uint32_t compare, uint32_t val)
    ```

    ```
    __simt_callee__ inline int64_t asc_atomic_cas(__gm__ int64_t *address, int64_t compare, int64_t val)
    ```

    ```
    __simt_callee__ inline uint64_t asc_atomic_cas(__gm__ uint64_t *address, uint64_t compare, uint64_t val)
    ```

    ```
    __simt_callee__ inline half2 asc_atomic_cas(__gm__ half2 *address, half2 compare, half2 val)
    ```

    ```
    __simt_callee__ inline bfloat16x2_t asc_atomic_cas(__gm__ bfloat16x2_t *address, bfloat16x2_t compare, bfloat16x2_t val)
    ```

## 参数说明<a name="section0866173114710"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="15.09%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="13.5%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="15.09%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>address</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a><span id="ph10389112463510"><a name="ph10389112463510"></a><a name="ph10389112463510"></a>Unified Buffer</span>或<span id="ph1744110199399"><a name="ph1744110199399"></a><a name="ph1744110199399"></a>Global Memory</span>的地址。</p>
</td>
</tr>
<tr id="row96761233103319"><td class="cellrowborder" valign="top" width="15.09%" headers="mcps1.2.4.1.1 "><p id="p6255648487"><a name="p6255648487"></a><a name="p6255648487"></a>compare</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p182559481483"><a name="p182559481483"></a><a name="p182559481483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p53501143123819"><a name="p53501143123819"></a><a name="p53501143123819"></a>源操作数，做比较的值。</p>
</td>
</tr>
<tr id="row1380616275388"><td class="cellrowborder" valign="top" width="15.09%" headers="mcps1.2.4.1.1 "><p id="p6807927193813"><a name="p6807927193813"></a><a name="p6807927193813"></a>val</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p48071627173819"><a name="p48071627173819"></a><a name="p48071627173819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p08071327163815"><a name="p08071327163815"></a><a name="p08071327163815"></a>源操作数，用于赋值的值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

Unified Buffer或Global Memory上的初始数据。

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section7131139151810"></a>

使用除half2、bfloat16x2\_t类型之外的接口需要包含"simt\_api/device\_atomic\_functions.h"头文件，使用half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/device_atomic_functions.h"
```

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelAtomicCas(__gm__ int32_t* dst, __gm__ int32_t* src0, __gm__ int32_t* src1)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    asc_atomic_cas(dst + idx, src0[idx], src1[idx]);
}
```

