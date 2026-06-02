# ResetLoopModePara<a name="ZH-CN_TOPIC_0000002523344620"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

重置loop mode的参数。与[SetLoopModePara](SetLoopModePara.md)搭配使用，在使能loop mode并且设置loop mode的参数的数据搬运场景下，数据搬运结束后需要调用该函数来重置loop mode参数。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void ResetLoopModePara(DataCopyMVType type)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table1927871911568"></a>
<table><thead align="left"><tr id="row162781519105612"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p12278131911565"><a name="p12278131911565"></a><a name="p12278131911565"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.2.4.1.2"><p id="p1227971911567"><a name="p1227971911567"></a><a name="p1227971911567"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.2.4.1.3"><p id="p4279151905611"><a name="p4279151905611"></a><a name="p4279151905611"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row02796198566"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p1030265017216"><a name="p1030265017216"></a><a name="p1030265017216"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.2.4.1.2 "><p id="p8279719175614"><a name="p8279719175614"></a><a name="p8279719175614"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.2.4.1.3 "><p id="p202731227454"><a name="p202731227454"></a><a name="p202731227454"></a>数据搬运模式。DataCopyMVType为枚举类型，定义如下，具体参数说明请参考<a href="SetLoopModePara.md#table1166074612214">表3</a>。</p>
<a name="screen1354412162247"></a><a name="screen1354412162247"></a><pre class="screen" codetype="Cpp" id="screen1354412162247">enum class DataCopyMVType : uint8_t {
    UB_TO_OUT = 0,
    OUT_TO_UB = 1,
};</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section1227835243314"></a>

本示例中操作数数据类型为int8\_t。

```
AscendC::LocalTensor<int8_t> srcLocal = inQueueSrc.AllocTensor<int8_t>();
AscendC::DataCopyExtParams copyParams{2, 48 * sizeof(int8_t), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
AscendC::LoopModeParams loopParam2Ub {2, 2, 96, 128, 192, 288};
AscendC::SetLoopModePara(loopParam2Ub, DataCopyMVType::OUT_TO_UB);
AscendC::DataCopyPad<int8_t, PaddingMode::Compact(srcLocal, srcGlobal, copyParams, padParams); // 从GM->VECIN搬运 48 * 2 * 2 * 2 = 384Bytes
AscendC::ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
AscendC::LoopModeParams loopParam2Gm {2, 2, 128, 96, 288, 192};
AscendC::SetLoopModePara(loopParams2Gm, DataCopyMVType::UB_TO_OUT);
DataCopyPad<T, PaddingMode::Compact>(dstGlobal, srcLocal, copyParams);
AscendC::ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
```

