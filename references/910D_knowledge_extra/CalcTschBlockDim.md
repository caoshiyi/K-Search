# CalcTschBlockDim<a name="ZH-CN_TOPIC_0000002554423869"></a>

## 功能说明<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_section36583473819"></a>

针对Cube、Vector分离模式，用于计算Cube、Vector融合算子的numBlocks。针对Vector/Cube融合计算的算子，启动时，按照AIV和AIC组合启动，numBlocks用于设置启动多少个组合执行，比如某款AI处理器上有40个Vector核+20个Cube核，一个组合是2个Vector和1个Cube核，建议设置为20，此时会启动20个组合，即40个Vector和20个Cube核。使用该接口可以自动获取合适的numBlocks值。

获取该值后，使用SetBlockDim进行numBlocks的设置。

## 函数原型<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_section13230182415108"></a>

```
uint32_t CalcTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
```

## 参数说明<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_section189014013619"></a>

<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_table111938719446"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_row6223476444"><th class="cellrowborder" valign="top" width="17.22%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p10223674448"><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p10223674448"></a><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p10223674448"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="12.3%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p645511218169"><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p645511218169"></a><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p645511218169"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.48%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p1922337124411"><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p1922337124411"></a><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p1922337124411"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_row152234713443"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001675021153_p0240991576"><a name="zh-cn_topic_0000001675021153_p0240991576"></a><a name="zh-cn_topic_0000001675021153_p0240991576"></a>sliceNum</p>
</td>
<td class="cellrowborder" valign="top" width="12.3%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p167701536957"><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p167701536957"></a><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p167701536957"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.48%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p4611154016587"><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p4611154016587"></a><a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_p4611154016587"></a>数据切分的份数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001675021153_row049741215578"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001675021153_p174971412195717"><a name="zh-cn_topic_0000001675021153_p174971412195717"></a><a name="zh-cn_topic_0000001675021153_p174971412195717"></a>aicCoreNum</p>
</td>
<td class="cellrowborder" valign="top" width="12.3%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001675021153_p18497121213577"><a name="zh-cn_topic_0000001675021153_p18497121213577"></a><a name="zh-cn_topic_0000001675021153_p18497121213577"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.48%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001675021153_p5497161211575"><a name="zh-cn_topic_0000001675021153_p5497161211575"></a><a name="zh-cn_topic_0000001675021153_p5497161211575"></a><span>如果算子实现使用了</span><span>矩阵计算API</span><span>，请传入</span><a href="GetCoreNumAic.md">GetCoreNumAic</a><span>返回的数量</span>，否则传入0。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001675021153_row848120149576"><td class="cellrowborder" valign="top" width="17.22%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001675021153_p17481614135719"><a name="zh-cn_topic_0000001675021153_p17481614135719"></a><a name="zh-cn_topic_0000001675021153_p17481614135719"></a>aivCoreNum</p>
</td>
<td class="cellrowborder" valign="top" width="12.3%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001675021153_p10481314155712"><a name="zh-cn_topic_0000001675021153_p10481314155712"></a><a name="zh-cn_topic_0000001675021153_p10481314155712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.48%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001675021153_p1048101475710"><a name="zh-cn_topic_0000001675021153_p1048101475710"></a><a name="zh-cn_topic_0000001675021153_p1048101475710"></a><span>如果算子实现使用了</span><span>矢量计算API</span><span>，请传入</span><a href="GetCoreNumAiv.md">GetCoreNumAiv</a><span>返回的数量</span>，否则传入0。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_section25791320141317"></a>

返回用于底层任务调度的核数。

## 约束说明<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_section19165124931511"></a>

无

## 调用示例<a name="zh-cn_topic_0000001675021153_zh-cn_topic_0000001442758437_section320753512363"></a>

```
ge::graphStatus TilingXXX(gert::TilingContext* context) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    // ..按照aivNum切分数据，并进行计算
    uint32_t sliceNum = aivNum;
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(sliceNum, aicNum, aivNum));
    return ret;
}
```

