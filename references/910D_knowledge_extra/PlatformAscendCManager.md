# PlatformAscendCManager<a name="ZH-CN_TOPIC_0000002554344763"></a>

## 功能说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section36583473819"></a>

基于Kernel Launch算子工程，通过Kernel直调（Kernel Launch）方式调用算子的场景下，可能需要获取硬件平台相关信息，比如获取硬件平台的核数。PlatformAscendCManager类提供获取平台信息的功能：通过该类的GetInstance方法可以获取一个PlatformAscendC类的指针，再通过该指针获取硬件平台相关信息，支持获取的信息可参考[PlatformAscendC](PlatformAscendC.md)。

> **须知：** 
>-   使用该功能需要包含"tiling/platform/platform\_ascendc.h"头文件，并在编译脚本中链接tiling\_api、platform动态库。
>    -   包含头文件的样例如下：
>        ```
>        #include "tiling/platform/platform_ascendc.h"
>        ```
>    -   链接动态库的样例如下:
>        ```
>        add_executable(main main.cpp)
>        target_link_libraries(main PRIVATE
>          kernels
>          tiling_api
>          platform
>        )
>        ```
>-   当前该类仅支持如下型号：

## 函数原型<a name="zh-cn_topic_0000001796358754_section7979556121414"></a>

```
class PlatformAscendCManager {
public:
    static PlatformAscendC* GetInstance();
    // 在仅有CPU环境、无对应的NPU硬件环境时，需要传入customSocVersion来指定对应的AI处理器型号。注意：因为GetInstance实现属于单例模式，仅在第一次调用时传入的customSocVersion生效。
    static PlatformAscendC* GetInstance(const char *customSocVersion);
private:
...
}
```

## 参数说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section189014013619"></a>

<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p10223674448"><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p10223674448"></a><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p645511218169"><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p645511218169"></a><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p1922337124411"><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p1922337124411"></a><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001796358754_p4871895189"><a name="zh-cn_topic_0000001796358754_p4871895189"></a><a name="zh-cn_topic_0000001796358754_p4871895189"></a>customSocVersion</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p167701536957"><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p167701536957"></a><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p167701536957"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p4611154016587"><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p4611154016587"></a><a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_p4611154016587"></a>AI处理器型号。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001796358754_zh-cn_topic_0000001442758437_section320753512363"></a>

```
GetInfoFun() {
    ...
    auto coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNum();
    ...
    return;
}
```

