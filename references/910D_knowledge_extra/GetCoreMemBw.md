# GetCoreMemBw<a name="ZH-CN_TOPIC_0000002523344824"></a>

## 功能说明<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_section36583473819"></a>

获取硬件平台存储空间的带宽大小。硬件存储空间类型定义如下：

```
enum class CoreMemType {
    L0_A = 0, // 预留参数，暂不支持
    L0_B = 1, // 预留参数，暂不支持
    L0_C = 2, // 预留参数，暂不支持
    L1 = 3,   // 预留参数，暂不支持
    L2 = 4,
    UB = 5,   // 预留参数，暂不支持
    HBM = 6,
    RESERVED
};
```

## 函数原型<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_section13230182415108"></a>

```
void GetCoreMemBw(const CoreMemType &memType, uint64_t &bwSize) const
```

## 参数说明<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_section189014013619"></a>

<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p10223674448"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p10223674448"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="15.340000000000002%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p645511218169"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p645511218169"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="67.44%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1922337124411"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1922337124411"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p93508208198"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p93508208198"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p93508208198"></a>memType</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1435019203197"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1435019203197"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1435019203197"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1435022010192"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1435022010192"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1435022010192"></a>硬件存储空间类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_row18403121314196"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p5519131912459"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p5519131912459"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p5519131912459"></a>bwSize</p>
</td>
<td class="cellrowborder" valign="top" width="15.340000000000002%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1496852144310"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1496852144310"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p1496852144310"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.44%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p94961752154312"><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p94961752154312"></a><a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_p94961752154312"></a>对应硬件的存储空间的带宽大小。单位是Byte/cycle，cycle代表时钟周期。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_section25791320141317"></a>

无

## 约束说明<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001597681768_zh-cn_topic_0000001391767420_section320753512363"></a>

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t l2_bw;
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, l2_bw);
    // ...
    return ret;
}
```

