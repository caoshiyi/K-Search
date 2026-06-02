# SetSingleOutputShape<a name="ZH-CN_TOPIC_0000002554423473"></a>

## 功能说明<a name="section1286701331918"></a>

设置单核上结果矩阵Output的形状。

## 函数原型<a name="section134719311196"></a>

```
void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM)
```

## 参数说明<a name="section15084118194"></a>

<a name="table322217833113"></a>
<table><thead align="left"><tr id="row162421986314"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p1924211810316"><a name="p1924211810316"></a><a name="p1924211810316"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p172424823112"><a name="p172424823112"></a><a name="p172424823112"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p1324210863111"><a name="p1324210863111"></a><a name="p1324210863111"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row724218893118"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p0242118133112"><a name="p0242118133112"></a><a name="p0242118133112"></a><span>singleCo</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p5242198193110"><a name="p5242198193110"></a><a name="p5242198193110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p152421483311"><a name="p152421483311"></a><a name="p152421483311"></a><span>单核上输出通道的大小</span>。</p>
</td>
</tr>
<tr id="row14242158183110"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p42425813115"><a name="p42425813115"></a><a name="p42425813115"></a><span>singleDo</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p324219823119"><a name="p324219823119"></a><a name="p324219823119"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p224388163111"><a name="p224388163111"></a><a name="p224388163111"></a><span>单核上Output D维度大小</span>。</p>
</td>
</tr>
<tr id="row1243387317"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p82431984319"><a name="p82431984319"></a><a name="p82431984319"></a><span>singleM</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1724358113113"><a name="p1724358113113"></a><a name="p1724358113113"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p224318833117"><a name="p224318833117"></a><a name="p224318833117"></a><span>单核上Output M维度大小</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section159985791912"></a>

无

## 约束说明<a name="section20930151192020"></a>

无

## 调用示例<a name="section1483419198204"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform);
conv3dApiTiling.SetSingleOutputShape(singleCo, singleDo, singleM);
```

