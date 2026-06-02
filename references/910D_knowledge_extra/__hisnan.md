# \_\_hisnan<a name="ZH-CN_TOPIC_0000002523303672"></a>

## 产品支持情况<a name="section187361215133116"></a>

<a name="table97361315153115"></a>
<table><thead align="left"><tr id="row1873681512311"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p8736151519312"><a name="p8736151519312"></a><a name="p8736151519312"></a><span id="ph8736161563114"><a name="ph8736161563114"></a><a name="ph8736161563114"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p12736201553116"><a name="p12736201553116"></a><a name="p12736201553116"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row13736141573115"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p5736161514313"><a name="p5736161514313"></a><a name="p5736161514313"></a><span id="ph207361215103118"><a name="ph207361215103118"></a><a name="ph207361215103118"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1573641512315"><a name="p1573641512315"></a><a name="p1573641512315"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section7738151563114"></a>

判断浮点数是否为nan。

## 函数原型<a name="section187386155313"></a>

```
__simt_callee__ inline bool __hisnan(half x)
```

```
__simt_callee__ inline bool __hisnan(bfloat16_t x)
```

## 参数说明<a name="section57383151318"></a>

**表 1**  参数说明

<a name="table15738015103113"></a>
<table><thead align="left"><tr id="row157381615143115"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p15738315193114"><a name="p15738315193114"></a><a name="p15738315193114"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p37381151312"><a name="p37381151312"></a><a name="p37381151312"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p17381615203112"><a name="p17381615203112"></a><a name="p17381615203112"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12738131511318"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1573831563119"><a name="p1573831563119"></a><a name="p1573831563119"></a>x</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p9738315113115"><a name="p9738315113115"></a><a name="p9738315113115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1573871516313"><a name="p1573871516313"></a><a name="p1573871516313"></a>源操作数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section127389152311"></a>

-   false：输入非nan。
-   true：输入为nan。

## 约束说明<a name="section573851518317"></a>

无

## 需要包含的头文件<a name="section10354115115916"></a>

使用half类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例<a name="section4738015203110"></a>

```
__simt_vf__ __launch_bounds__(1024) inline void KernelIsNan(__gm__ bool* dst, __gm__ half* x)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    dst[idx] = __hisnan(x[idx]);
}
```

