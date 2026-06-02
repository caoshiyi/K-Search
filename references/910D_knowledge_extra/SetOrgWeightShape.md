# SetOrgWeightShape<a name="ZH-CN_TOPIC_0000002523303932"></a>

## 功能说明<a name="section1639815442154"></a>

设置权重矩阵Weight的原始形状。

## 函数原型<a name="section14245021513"></a>

```
void SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw)
```

## 参数说明<a name="section13491856101517"></a>

<a name="table7191123232820"></a>
<table><thead align="left"><tr id="row11209732172818"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p62092321286"><a name="p62092321286"></a><a name="p62092321286"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p172091326288"><a name="p172091326288"></a><a name="p172091326288"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p72091232162817"><a name="p72091232162817"></a><a name="p72091232162817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row82097322287"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p3209203292813"><a name="p3209203292813"></a><a name="p3209203292813"></a><span>orgCo</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p2209332192819"><a name="p2209332192819"></a><a name="p2209332192819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p820913211280"><a name="p820913211280"></a><a name="p820913211280"></a><span>原始输出通道的大小</span>。</p>
</td>
</tr>
<tr id="row120914326285"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p32095329283"><a name="p32095329283"></a><a name="p32095329283"></a><span>orgKd</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p142096324289"><a name="p142096324289"></a><a name="p142096324289"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p7209193292814"><a name="p7209193292814"></a><a name="p7209193292814"></a><span>原始Weight D维度大小</span>。</p>
</td>
</tr>
<tr id="row22098324289"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p16209232192818"><a name="p16209232192818"></a><a name="p16209232192818"></a><span>orgKh</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p6209193272819"><a name="p6209193272819"></a><a name="p6209193272819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p92091328286"><a name="p92091328286"></a><a name="p92091328286"></a><span>原始Weight H维度大小</span>。</p>
</td>
</tr>
<tr id="row1020953232810"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p19209832102820"><a name="p19209832102820"></a><a name="p19209832102820"></a><span>orgKw</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p5209173242818"><a name="p5209173242818"></a><a name="p5209173242818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p1620963212818"><a name="p1620963212818"></a><a name="p1620963212818"></a><span>原始Weight W维度大小</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1625835167"></a>

无

## 约束说明<a name="section17147192019165"></a>

无

## 调用示例<a name="section194982277167"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetOrgWeightShape(cout, kd, kh, kw);
```

