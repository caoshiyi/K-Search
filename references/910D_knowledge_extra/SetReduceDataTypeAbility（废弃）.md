# SetReduceDataTypeAbility（废弃）<a name="ZH-CN_TOPIC_0000002554424633"></a>

> **说明：** 
>该接口废弃，并将在后续版本移除，请不要使用该接口。请使用[SetReduceType](SetReduceType.md)进行设置。

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

设置Reduce操作类型、目的数据类型和源数据类型，仅对有归约操作的通信任务生效。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline bool SetReduceDataTypeAbility(HcclReduceOp op, HcclDataType dstDataType, HcclDataType srcDataType)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.799999999999999%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="70.43%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p167361341213"><a name="p167361341213"></a><a name="p167361341213"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p137362417119"><a name="p137362417119"></a><a name="p137362417119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p69241958121317"><a name="p69241958121317"></a><a name="p69241958121317"></a>归约操作类型，仅对有归约操作的通信任务生效。uint32_t类型，取值详见<a href="HCCL使用说明.md#table2469980529">表2</a>参数说明。</p>
</td>
</tr>
<tr id="row962121412103"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p13336171656"><a name="p13336171656"></a><a name="p13336171656"></a>dstDataType</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p17336191251"><a name="p17336191251"></a><a name="p17336191251"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p171784316267"><a name="p171784316267"></a><a name="p171784316267"></a>通信任务中输出数据的数据类型。uint8_t类型，该参数的取值范围请参考<a href="HCCL使用说明.md#table116710585514">表1</a>。</p>
<p id="p645895003920"><a name="p645895003920"></a><a name="p645895003920"></a>不同通信任务支持的输出数据类型不同，具体为：</p>
<a name="ul17268204422614"></a><a name="ul17268204422614"></a><ul id="ul17268204422614"><li>对于AllReduce、AllGather、AlltoAllV、AlltoAllvWrite通信任务：输出的数据类型必须与输入的数据类型一致。各通信任务支持的输入数据类型请参考<a href="#p113559415512">srcDataType</a>。</li><li>对于ReduceScatter通信任务，当输入的数据类型为int16_t、int32_t、half、float、bfloat16_t时，输出的数据类型必须与其一致；当输入的数据类型为int8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t时，输出的数据类型必须为half、bfloat16_t、float三者之一。</li></ul>
</td>
</tr>
<tr id="row1950815120100"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p113559415512"><a name="p113559415512"></a><a name="p113559415512"></a>srcDataType</p>
</td>
<td class="cellrowborder" valign="top" width="11.799999999999999%" headers="mcps1.2.4.1.2 "><p id="p535574457"><a name="p535574457"></a><a name="p535574457"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="70.43%" headers="mcps1.2.4.1.3 "><p id="p1339119178286"><a name="p1339119178286"></a><a name="p1339119178286"></a>通信任务中输入数据的数据类型。uint8_t类型，该参数的取值范围请参考<a href="HCCL使用说明.md#table116710585514">表1</a>。</p>
<p id="p14631313113"><a name="p14631313113"></a><a name="p14631313113"></a>不同通信任务支持的输入数据类型如下：</p>
<a name="ul142561725132915"></a><a name="ul142561725132915"></a><ul id="ul142561725132915"><li>AllReduce通信任务：支持的输入类型为int16_t、half、bfloat16_t、int32_t、float。</li><li>AllGather、AlltoAllV通信任务：支持的输入类型为int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、double。</li><li>ReduceScatter通信任务：支持的输入类型为int8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、half、bfloat16_t、int32_t、float。</li><li>AlltoAllvWrite通信任务：支持的输入类型为int8_t、uint8_t、hifloat8_t、fp8_e5m2_t、fp8_e4m3fn_t、int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float、int64_t、double。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

返回是否设置成功。

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1665082013318"></a>

```
Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
hccl.InitV2(contextGM, &tilingData);
auto ret = hccl.SetCcTilingV2(offsetof(AllReduceCustomTilingData, mc2CcTiling));
if (ret) {
    return;
}

hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_RESERVED, AscendC::HcclDataType::HCCL_DATA_TYPE_BFP16, AscendC::HcclDataType::HCCL_DATA_TYPE_BFP16);
```

