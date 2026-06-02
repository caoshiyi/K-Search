# printf<a name="ZH-CN_TOPIC_0000002554424113"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

本接口提供SIMT VF调试场景下的格式化输出功能。在算子Kernel侧的SIMT VF实现代码中，需要输出日志信息时，调用printf接口打印相关内容。

## 函数原型<a name="section2067518173415"></a>

```
template <class... Args>
__attribute__((always_inline)) inline __simt_callee__ void printf(const __gm__ char* fmt, Args&&... args)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p45208478318"><a name="p45208478318"></a><a name="p45208478318"></a>fmt</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p135196472314"><a name="p135196472314"></a><a name="p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1965816506312"><a name="p1965816506312"></a><a name="p1965816506312"></a>格式控制字符串，包含两种类型的对象：普通字符和转换说明。</p>
<a name="ul419411543310"></a><a name="ul419411543310"></a><ul id="ul419411543310"><li>普通字符将原样不动地打印输出。</li><li>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。<div class="p" id="p158820115597"><a name="p158820115597"></a><a name="p158820115597"></a>支持的转换类型包括：<a name="ul541124915329"></a><a name="ul541124915329"></a><ul id="ul541124915329"><li>%d / %ld / %lld / %i / %li / %lli：输出十进制数，支持打印的数据类型：int8_t、int16_t、int32_t、int64_t</li><li>%f / %F：输出浮点数，支持打印的数据类型：float、half、bfloat16_t</li><li>%x / %lx / %llx：输出十六进制整数，支持打印的数据类型：int8_t、int16_t、int32_t、int64_t、uint8_t、uint16_t、uint32_t、uint64_t</li><li>%s：输出字符串</li><li>%u / %lu /%llu：输出unsigned类型数据，支持打印的数据类型：uint8_t、uint16_t、uint32_t、uint64_t</li><li>%p：输出指针地址</li></ul>
</div>
<p id="p1133101265017"><a name="p1133101265017"></a><a name="p1133101265017"></a><strong id="b164621818125011"><a name="b164621818125011"></a><a name="b164621818125011"></a>注意</strong>：上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。</p>
</li></ul>
</td>
</tr>
<tr id="row163919564263"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p1563916565265"><a name="p1563916565265"></a><a name="p1563916565265"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p59396564285"><a name="p59396564285"></a><a name="p59396564285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1257115311337"><a name="p1257115311337"></a><a name="p1257115311337"></a>附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

-   printf在SIMT VF中调用，会占用SIMT栈空间，请合理控制调用printf次数，防止SIMT栈溢出。
-   printf功能需要占用额外的Global Memory空间用于数据缓存，缓存空间大小默认为2MB。您可以通过acl.json中的"simt\_printf\_fifo\_size"字段进行配置，配置范围最小为1MB，最大为64MB。当打印数据量较大时，建议增加缓存空间。
-   在仿真环境下，使用printf接口会增加算子运行时间，通过在VF代码中判断线程ID，可以仅在部分线程中打印调试信息，减少重复内容的打印，更有利于调试。

## 需要包含的头文件<a name="section10354115115916"></a>

使用该接口需要包含"utils/debug/asc\_printf.h"头文件。

```
#include "utils/debug/asc_printf.h"
```

## 调用示例<a name="section82241477610"></a>

```
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "utils/debug/asc_printf.h"

// asc_vf_call调用时dim3参数：dim3(8, 2, 8)
__simt_vf__ __launch_bounds__(128) inline void SimtCompute()
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;
    printf("simt vf: d: (%d, %d, %d), f: %f, s: %s\n", x, y, z, 3.14f, "pass");
}
```

NPU模式下，程序运行时打印效果如下：

```
simt vf: d: (0, 0, 0), f: 3.140000, s: pass
simt vf: d: (0, 0, 1), f: 3.140000, s: pass
simt vf: d: (0, 0, 2), f: 3.140000, s: pass
simt vf: d: (0, 0, 3), f: 3.140000, s: pass
simt vf: d: (0, 0, 4), f: 3.140000, s: pass
simt vf: d: (0, 0, 5), f: 3.140000, s: pass
simt vf: d: (0, 0, 6), f: 3.140000, s: pass
simt vf: d: (0, 0, 7), f: 3.140000, s: pass
simt vf: d: (0, 1, 0), f: 3.140000, s: pass
simt vf: d: (0, 1, 1), f: 3.140000, s: pass
......
```

