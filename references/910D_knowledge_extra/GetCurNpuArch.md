# GetCurNpuArch<a name="ZH-CN_TOPIC_0000002523343740"></a>

## 功能说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section36583473819"></a>

获取当前硬件平台芯片架构版本号。

## 函数原型<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
NpuArch GetCurNpuArch(void) const
```

## 参数说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section189014013619"></a>

无

## 返回值<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section25791320141317"></a>

当前硬件平台架构号的枚举类。该枚举类和AI处理器型号的对应关系请通过CANN软件安装后文件存储路径下include/platform/soc\_spec.h头文件获取。

<a name="table739123016439"></a>
<table><thead align="left"><tr id="row142010303433"><th class="cellrowborder" valign="top" width="38.84%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002433359052_p1883113061818"><a name="zh-cn_topic_0000002433359052_p1883113061818"></a><a name="zh-cn_topic_0000002433359052_p1883113061818"></a>产品</p>
</th>
<th class="cellrowborder" valign="top" width="61.160000000000004%" id="mcps1.1.3.1.2"><p id="p842043019436"><a name="p842043019436"></a><a name="p842043019436"></a>NPU_ARCH</p>
</th>
</tr>
</thead>
<tbody><tr id="row642023094317"><td class="cellrowborder" valign="top" width="38.84%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" valign="top" width="61.160000000000004%" headers="mcps1.1.3.1.2 "><p id="p1642015301431"><a name="p1642015301431"></a><a name="p1642015301431"></a>DAV_3510</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001664705472_zh-cn_topic_0000001442758437_section320753512363"></a>

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto npuArch = ascendcPlatform.GetCurNpuArch();
    // 根据所获得的版本型号自行设计Tiling策略
    // DAV_XXX请替换为实际的架构号
    if (socVersion == NpuArch::DAV_XXXX) {
        // ...
    }
    return ret;
}
```

