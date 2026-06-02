# SetOrgInputShape<a name="ZH-CN_TOPIC_0000002523343616"></a>

## 功能说明<a name="section658953516161"></a>

设置特征矩阵Input的原始形状。

## 函数原型<a name="section2389419169"></a>

```
void SetOrgInputShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi)
```

## 参数说明<a name="section948184811163"></a>

<a name="table10193433132917"></a>
<table><thead align="left"><tr id="row92256336296"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p622573319296"><a name="p622573319296"></a><a name="p622573319296"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p152258335290"><a name="p152258335290"></a><a name="p152258335290"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p1222553332915"><a name="p1222553332915"></a><a name="p1222553332915"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row72252033132914"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p11225143316291"><a name="p11225143316291"></a><a name="p11225143316291"></a><span>orgCi</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1722513372911"><a name="p1722513372911"></a><a name="p1722513372911"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p122513337299"><a name="p122513337299"></a><a name="p122513337299"></a><span>原始输入通道的大小</span>。</p>
</td>
</tr>
<tr id="row2225193319293"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p822533322914"><a name="p822533322914"></a><a name="p822533322914"></a><span>orgDi</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1722593302910"><a name="p1722593302910"></a><a name="p1722593302910"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p622563312915"><a name="p622563312915"></a><a name="p622563312915"></a><span>原始Input D维度大小</span>。</p>
</td>
</tr>
<tr id="row822523312294"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p5226113392915"><a name="p5226113392915"></a><a name="p5226113392915"></a><span>orgHi</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p14226133313297"><a name="p14226133313297"></a><a name="p14226133313297"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p522683372910"><a name="p522683372910"></a><a name="p522683372910"></a><span>原始Input H维度大小</span>。</p>
</td>
</tr>
<tr id="row4226163312292"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p162267330297"><a name="p162267330297"></a><a name="p162267330297"></a><span>orgWi</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p922616334296"><a name="p922616334296"></a><a name="p922616334296"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p16226103392912"><a name="p16226103392912"></a><a name="p16226103392912"></a><span>原始Input W维度大小</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section139371535169"></a>

无

## 约束说明<a name="section11909111014175"></a>

无

## 调用示例<a name="section197191711176"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetOrgInputShape(orgCi, orgDi, orgHi, orgWi);
```

