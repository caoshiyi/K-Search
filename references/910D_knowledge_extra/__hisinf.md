# \_\_hisinf<a name="ZH-CN_TOPIC_0000002523344802"></a>

## 产品支持情况<a name="section65472564339"></a>

<a name="table35471056133316"></a>
<table><thead align="left"><tr id="row1254775613337"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p115470567338"><a name="p115470567338"></a><a name="p115470567338"></a><span id="ph754765619335"><a name="ph754765619335"></a><a name="ph754765619335"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p105472564333"><a name="p105472564333"></a><a name="p105472564333"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1754713568333"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p145471156133318"><a name="p145471156133318"></a><a name="p145471156133318"></a><span id="ph5547195620331"><a name="ph5547195620331"></a><a name="ph5547195620331"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p154714569332"><a name="p154714569332"></a><a name="p154714569332"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section19548756173310"></a>

判断浮点数是否为无穷。

## 函数原型<a name="section13548205617332"></a>

```
__simt_callee__ inline bool __hisinf(half x)
```

```
__simt_callee__ inline bool __hisinf(bfloat16_t x)
```

## 参数说明<a name="section20548135663313"></a>

**表 1**  参数说明

<a name="table1454895614332"></a>
<table><thead align="left"><tr id="row165489569331"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p1654895643314"><a name="p1654895643314"></a><a name="p1654895643314"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p654865618337"><a name="p654865618337"></a><a name="p654865618337"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p205487566333"><a name="p205487566333"></a><a name="p205487566333"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row17548175619336"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1054865603311"><a name="p1054865603311"></a><a name="p1054865603311"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p16548115653318"><a name="p16548115653318"></a><a name="p16548115653318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p55481156103319"><a name="p55481156103319"></a><a name="p55481156103319"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1654815561336"></a>

-   false：输入不是无穷。
-   true：输入为inf、-inf。

## 约束说明<a name="section165481356163312"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section14548125615335"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelIsInf(__gm__ bool* dst, __gm__ half* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = __hisinf(x[idx]);
}
```

