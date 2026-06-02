# SetStride<a name="ZH-CN_TOPIC_0000002523343660"></a>

## 功能说明<a name="section157281433142619"></a>

设置Stride信息。

## 函数原型<a name="section2069193932616"></a>

```
void SetStride(int64_t strideD, int64_t strideH, int64_t strideW)
```

## 参数说明<a name="section9467134672618"></a>

<a name="table52615251377"></a>
<table><thead align="left"><tr id="row132220257376"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p183225254371"><a name="p183225254371"></a><a name="p183225254371"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p632213253377"><a name="p632213253377"></a><a name="p632213253377"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p1322192583714"><a name="p1322192583714"></a><a name="p1322192583714"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row53222025133718"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p3322142563716"><a name="p3322142563716"></a><a name="p3322142563716"></a><span>strideD</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p9322152513714"><a name="p9322152513714"></a><a name="p9322152513714"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p103221225153719"><a name="p103221225153719"></a><a name="p103221225153719"></a>D方向<span>Stride</span>大小。</p>
</td>
</tr>
<tr id="row1322122563716"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p153221825173718"><a name="p153221825173718"></a><a name="p153221825173718"></a><span>strideH</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1332332543712"><a name="p1332332543712"></a><a name="p1332332543712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p3323112513372"><a name="p3323112513372"></a><a name="p3323112513372"></a>H方向<span>Stride</span>大小。</p>
</td>
</tr>
<tr id="row1932322593715"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p173232025123716"><a name="p173232025123716"></a><a name="p173232025123716"></a><span>strideW</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p17323025183710"><a name="p17323025183710"></a><a name="p17323025183710"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p18323122513717"><a name="p18323122513717"></a><a name="p18323122513717"></a>W方向<span>Stride</span>大小。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1626965492617"></a>

无

## 约束说明<a name="section56681815272"></a>

在调用GetTiling接口前，本接口可选调用。若未调用本接口，默认strideD=1, strideH=1, strideW=1。

## 调用示例<a name="section2279191410271"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetStride(strideD, strideH, strideW);
```

