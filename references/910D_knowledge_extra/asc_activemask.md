# asc\_activemask<a name="ZH-CN_TOPIC_0000002523344526"></a>

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

查看Warp内所有线程是否为活跃状态。

返回一个32bit的无符号整数，若Warp内某个线程是活跃（已结束线程是非活跃状态）的，则返回值中与线程LaneId对应的bit位为1，否则为0。Warp内所有活跃线程返回相同的结果。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline uint32_t asc_activemask()
```

## 参数说明<a name="section0866173114710"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

32bit的无符号整数：若Warp内某个线程是活跃的，则返回值中与线程LaneId对应的bit位为1，否则为0。

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"simt\_api/device\_warp\_functions.h"头文件。

```
#include "simt_api/device_warp_functions.h"
```

## 调用示例<a name="section134121647154719"></a>

```
__simt_vf____launch_bounds__(1024) inline void KernelActiveMask(__gm__ uint32_t* dst)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // asc_vf_call参数：dim3{1024, 1, 1}
    uint32_t result = asc_activemask(); // 返回值为0xffffffff
    dst[idx] = result;
}
```

