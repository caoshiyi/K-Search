# asc\_atomic\_or<a name="ZH-CN_TOPIC_0000002523343900"></a>

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

对Unified Buffer或Global Memory上address的数值与指定数值val进行原子或（|）操作，即将address数值或（|）val的结果赋值到Unified Buffer或Global Memory上。

## 函数原型<a name="section620mcpsimp"></a>

-   Unified Buffer数据的原子或操作

    ```
    __simt_callee__ inline int32_t asc_atomic_or(__ubuf__ int32_t *address, int32_t val)
    ```

    ```
    __simt_callee__ inline uint32_t asc_atomic_or(__ubuf__ uint32_t *address, uint32_t val)
    ```

-   Global Memory数据的原子或操作

    ```
    __simt_callee__ inline int32_t asc_atomic_or(__gm__ int32_t *address, int32_t val)
    ```

    ```
    __simt_callee__ inline uint32_t asc_atomic_or(__gm__ uint32_t *address, uint32_t val)
    ```

    ```
    __simt_callee__ inline int64_t asc_atomic_or(__gm__ int64_t *address, int64_t val)
    ```

    ```
    __simt_callee__ inline uint64_t asc_atomic_or(__gm__ uint64_t *address, uint64_t val)
    ```

## 参数说明<a name="section0866173114710"></a>

**表 1**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="12.889999999999999%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="15.7%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="12.889999999999999%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>address</p>
</td>
<td class="cellrowborder" valign="top" width="15.7%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p430979104011"><a name="p430979104011"></a><a name="p430979104011"></a><span id="ph10389112463510"><a name="ph10389112463510"></a><a name="ph10389112463510"></a>Unified Buffer</span>或<span id="ph1744110199399"><a name="ph1744110199399"></a><a name="ph1744110199399"></a>Global Memory</span>的地址。</p>
</td>
</tr>
<tr id="row10532134133310"><td class="cellrowborder" valign="top" width="12.889999999999999%" headers="mcps1.2.4.1.1 "><p id="p6255648487"><a name="p6255648487"></a><a name="p6255648487"></a>val</p>
</td>
<td class="cellrowborder" valign="top" width="15.7%" headers="mcps1.2.4.1.2 "><p id="p182559481483"><a name="p182559481483"></a><a name="p182559481483"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p62551148687"><a name="p62551148687"></a><a name="p62551148687"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

Unified Buffer或Global Memory上的初始数据。

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section7131139151810"></a>

使用该接口需要包含"simt\_api/device\_atomic\_functions.h"头文件。

```
#include "simt_api/device_atomic_functions.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelAtomicOr(__gm__ int32_t* dst, __gm__ int32_t* src)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    asc_atomic_or(dst + idx, src[idx]);
}
```

