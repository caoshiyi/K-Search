# asc\_threadfence\_block<a name="ZH-CN_TOPIC_0000002554343617"></a>

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

用于协调同一个线程块（Thread Block）内的线程之间的内存操作顺序。确保某一线程在asc\_threadfence\_block\(\) 之前的所有内存操作（读写），对同一线程块内的其他线程是可见的。

## 函数原型<a name="section620mcpsimp"></a>

```
__simt_callee__ inline void asc_threadfence_block()
```

## 参数说明<a name="section0866173114710"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 需要包含的头文件<a name="section7131139151810"></a>

使用该接口需要包含"simt\_api/device\_sync\_functions.h"头文件。

```
#include "simt_api/device_sync_functions.h"
```

## 调用示例<a name="section8586632183917"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelThreadFenceBlock(__gm__ float* dst, __gm__ float* src)
{
    src[0] = src[0] + 1;
    asc_threadfence_block();
    dst[0] = src[0];
}
```

