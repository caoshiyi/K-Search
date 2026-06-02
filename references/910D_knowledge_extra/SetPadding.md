# SetPadding<a name="ZH-CN_TOPIC_0000002554423813"></a>

## 功能说明<a name="section12666854152413"></a>

设置Pad信息。

## 函数原型<a name="section3966811254"></a>

```
void SetPadding(int64_t padHead, int64_t padTail, int64_t padUp, int64_t padDown, int64_t padLeft, int64_t padRight)
```

## 参数说明<a name="section42901691251"></a>

<a name="table15473556133416"></a>
<table><thead align="left"><tr id="row649395614341"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p1493185619349"><a name="p1493185619349"></a><a name="p1493185619349"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p17493135633420"><a name="p17493135633420"></a><a name="p17493135633420"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p19493145643410"><a name="p19493145643410"></a><a name="p19493145643410"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row11493125643415"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p104935566346"><a name="p104935566346"></a><a name="p104935566346"></a><span>padHead</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p19493156153410"><a name="p19493156153410"></a><a name="p19493156153410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p184931056113411"><a name="p184931056113411"></a><a name="p184931056113411"></a>D方向前Padding大小。</p>
</td>
</tr>
<tr id="row2493115613345"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p2493185611346"><a name="p2493185611346"></a><a name="p2493185611346"></a><span>padTail</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p549365663414"><a name="p549365663414"></a><a name="p549365663414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p54931056133413"><a name="p54931056133413"></a><a name="p54931056133413"></a>D方向后Padding大小。</p>
</td>
</tr>
<tr id="row1493356133411"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p10493155612347"><a name="p10493155612347"></a><a name="p10493155612347"></a><span>padUp</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p14493185614348"><a name="p14493185614348"></a><a name="p14493185614348"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p1449414569345"><a name="p1449414569345"></a><a name="p1449414569345"></a>H方向上Padding大小。</p>
</td>
</tr>
<tr id="row174942056173419"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p349425612347"><a name="p349425612347"></a><a name="p349425612347"></a><span>padDown</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1494756193418"><a name="p1494756193418"></a><a name="p1494756193418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p8494195620341"><a name="p8494195620341"></a><a name="p8494195620341"></a>H方向下Padding大小。</p>
</td>
</tr>
<tr id="row149495643416"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1249435633419"><a name="p1249435633419"></a><a name="p1249435633419"></a><span>padLeft</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p104941356193418"><a name="p104941356193418"></a><a name="p104941356193418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p12494135613347"><a name="p12494135613347"></a><a name="p12494135613347"></a>W方向左Padding大小。</p>
</td>
</tr>
<tr id="row14494556103414"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p174947567342"><a name="p174947567342"></a><a name="p174947567342"></a><span>padRight</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p749425618347"><a name="p749425618347"></a><a name="p749425618347"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p149425653416"><a name="p149425653416"></a><a name="p149425653416"></a>W方向右Padding大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1689131616255"></a>

无

## 约束说明<a name="section028892718252"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认padHead=0, padTail=0, padUp=0, padDown=0, padLeft=0, padRight=0。

## 调用示例<a name="section360933214252"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetPadding(padHead, padTail, padUp, padDown, padLeft, padRight);
```

