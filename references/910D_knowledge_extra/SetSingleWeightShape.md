# SetSingleWeightShape<a name="ZH-CN_TOPIC_0000002523343710"></a>

## 功能说明<a name="section4612182501712"></a>

设置单核上权重矩阵Weight的形状。

## 函数原型<a name="section1522293381715"></a>

```
void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw)
```

## 参数说明<a name="section48891544181711"></a>

<a name="table2080312388301"></a>
<table><thead align="left"><tr id="row1385753853018"><th class="cellrowborder" valign="top" width="24.25%" id="mcps1.1.4.1.1"><p id="p118571238193010"><a name="p118571238193010"></a><a name="p118571238193010"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.29%" id="mcps1.1.4.1.2"><p id="p1485743818308"><a name="p1485743818308"></a><a name="p1485743818308"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="58.46%" id="mcps1.1.4.1.3"><p id="p168570386301"><a name="p168570386301"></a><a name="p168570386301"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1685783811302"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p2857143817306"><a name="p2857143817306"></a><a name="p2857143817306"></a><span>singleCi</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1385733819308"><a name="p1385733819308"></a><a name="p1385733819308"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p1685783843016"><a name="p1685783843016"></a><a name="p1685783843016"></a><span>单核上输入通道的大小</span>。</p>
</td>
</tr>
<tr id="row19857163815309"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p4857113812303"><a name="p4857113812303"></a><a name="p4857113812303"></a><span>singleKd</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p118571438203016"><a name="p118571438203016"></a><a name="p118571438203016"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p3857113863019"><a name="p3857113863019"></a><a name="p3857113863019"></a><span>单核上Weight D维度大小</span>。</p>
</td>
</tr>
<tr id="row14857738183010"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p3857113843017"><a name="p3857113843017"></a><a name="p3857113843017"></a><span>singleKh</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p385733873019"><a name="p385733873019"></a><a name="p385733873019"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p28571838163018"><a name="p28571838163018"></a><a name="p28571838163018"></a><span>单核上Weight H维度大小</span>。</p>
</td>
</tr>
<tr id="row1385783813011"><td class="cellrowborder" valign="top" width="24.25%" headers="mcps1.1.4.1.1 "><p id="p685733873010"><a name="p685733873010"></a><a name="p685733873010"></a><span>singleKw</span></p>
</td>
<td class="cellrowborder" valign="top" width="17.29%" headers="mcps1.1.4.1.2 "><p id="p1785715386302"><a name="p1785715386302"></a><a name="p1785715386302"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="58.46%" headers="mcps1.1.4.1.3 "><p id="p985716384303"><a name="p985716384303"></a><a name="p985716384303"></a><span>单核上Weight W维度大小</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section591165715173"></a>

无

## 约束说明<a name="section1389013330184"></a>

无

## 调用示例<a name="section9741164115186"></a>

```
// 实例化Conv3D Api
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
Conv3dTilingApi::Conv3dTiling conv3dApiTiling(ascendcPlatform );
conv3dApiTiling.SetSingleWeightShape(singleCi, singleKd, singleKh, singleKw);
```

