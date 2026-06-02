# SetDilation<a name="ZH-CN_TOPIC_0000002523343546"></a>

## 功能说明<a name="section465311396258"></a>

设置Dilation信息。

## 函数原型<a name="section168261244152511"></a>

```
void SetDilation(int64_t dilationD, int64_t dilationH, int64_t dilationW)
```

## 参数说明<a name="section209725566255"></a>

<a name="table199972113367"></a>
<table><thead align="left"><tr id="row1111862163612"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p9118172193611"><a name="p9118172193611"></a><a name="p9118172193611"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p151181321143620"><a name="p151181321143620"></a><a name="p151181321143620"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p15118112113367"><a name="p15118112113367"></a><a name="p15118112113367"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row411882123613"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p211822153616"><a name="p211822153616"></a><a name="p211822153616"></a><span>dilationD</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p811813210362"><a name="p811813210362"></a><a name="p811813210362"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p911872113611"><a name="p911872113611"></a><a name="p911872113611"></a>D方向<span>Dilation</span>大小。</p>
</td>
</tr>
<tr id="row411815214364"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p1411810211365"><a name="p1411810211365"></a><a name="p1411810211365"></a><span>dilationH</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p011813218367"><a name="p011813218367"></a><a name="p011813218367"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p16118102103618"><a name="p16118102103618"></a><a name="p16118102103618"></a>H方向<span>Dilation</span>大小。</p>
</td>
</tr>
<tr id="row161181521113610"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p17118162115368"><a name="p17118162115368"></a><a name="p17118162115368"></a><span>dilationW</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p13118721123610"><a name="p13118721123610"></a><a name="p13118721123610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p2011822111366"><a name="p2011822111366"></a><a name="p2011822111366"></a>W方向<span>Dilation</span>大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1465832102612"></a>

无

## 约束说明<a name="section6334131792612"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认dilationD=1, dilationH=1, dilationW=1。

## 调用示例<a name="section8746326122612"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetDilation(dilationD, dilationH, dilationW);
```

