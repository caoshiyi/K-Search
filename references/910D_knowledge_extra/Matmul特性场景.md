# Matmul特性场景<a name="ZH-CN_TOPIC_0000002523343926"></a>

除了前述介绍的[Matmul基本计算能力](Matmul使用说明.md)外，还请掌握Matmul的[基础知识](基础知识.md)和[算子实现](算子实现.md)。另外，Matmul矩阵编程还提供了适用于不同场景的处理能力及多种功能，具体场景和使用的关键接口或参数列于下表中，详细内容请见对应章节的介绍。

**表 1**  Matmul功能特性表

<a name="table165541417102915"></a>
<table><thead align="left"><tr id="row055416178295"><th class="cellrowborder" valign="top" width="40.1%" id="mcps1.2.3.1.1"><p id="p25541917182918"><a name="p25541917182918"></a><a name="p25541917182918"></a>特性描述</p>
</th>
<th class="cellrowborder" valign="top" width="59.9%" id="mcps1.2.3.1.2"><p id="p1855561719296"><a name="p1855561719296"></a><a name="p1855561719296"></a>涉及的关键API或参数</p>
</th>
</tr>
</thead>
<tbody><tr id="row355501792910"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p155551217132914"><a name="p155551217132914"></a><a name="p155551217132914"></a><a href="多核对齐切分.md">多核对齐切分</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p955571720298"><a name="p955571720298"></a><a name="p955571720298"></a><a href="SetDim.md">SetDim</a>、<a href="EnableMultiCoreSplitK.md">EnableMultiCoreSplitK</a>(多核切K场景)</p>
</td>
</tr>
<tr id="row1955511715296"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p3555171711292"><a name="p3555171711292"></a><a name="p3555171711292"></a><a href="多核非对齐切分.md">多核非对齐切分</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p158411416173315"><a name="p158411416173315"></a><a name="p158411416173315"></a><a href="SetTail.md">SetTail</a>、<a href="SetDim.md">SetDim</a>、<a href="EnableMultiCoreSplitK.md">EnableMultiCoreSplitK</a>(多核切K场景)</p>
</td>
</tr>
<tr id="row494289153013"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p5853634122310"><a name="p5853634122310"></a><a name="p5853634122310"></a><a href="异步场景处理.md">异步场景处理</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p1194212916304"><a name="p1194212916304"></a><a name="p1194212916304"></a><a href="Iterate.md">Iterate</a>、<a href="GetTensorC.md">GetTensorC</a>、<a href="IterateAll.md">IterateAll</a></p>
</td>
</tr>
<tr id="row17943181016308"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p1854719415227"><a name="p1854719415227"></a><a name="p1854719415227"></a><a href="MatmulCallBackFunc.md">自定义数据搬入搬出</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p129433104302"><a name="p129433104302"></a><a name="p129433104302"></a><a href="MatmulCallBackFunc.md">MatmulCallBackFunc</a>、<a href="SetUserDefInfo.md">SetUserDefInfo</a>、<a href="SetSelfDefineData.md">SetSelfDefineData</a></p>
</td>
</tr>
<tr id="row20954151283012"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p94242715239"><a name="p94242715239"></a><a name="p94242715239"></a><a href="矩阵乘输出的Channel拆分.md">矩阵乘输出的Channel拆分</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p1995411243014"><a name="p1995411243014"></a><a name="p1995411243014"></a><a href="MatmulConfig.md#table1761013213153">MatmulConfig</a>模板参数中的<a href="MatmulConfig.md#p5341543113715">isEnableChannelSplit</a>参数</p>
</td>
</tr>
<tr id="row12147214113012"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p5546154142210"><a name="p5546154142210"></a><a name="p5546154142210"></a><a href="矩阵向量乘.md">矩阵向量乘</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p1714721415303"><a name="p1714721415303"></a><a name="p1714721415303"></a><a href="SetAType.md">SetAType</a></p>
</td>
</tr>
<tr id="row050821573016"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p1559834499"><a name="p1559834499"></a><a name="p1559834499"></a><a href="MatmulPolicy.md#li1892181110422">上三角</a>/<a href="MatmulPolicy.md#li17292359134915">下三角</a>矩阵乘</p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p7508191514309"><a name="p7508191514309"></a><a name="p7508191514309"></a><a href="MatmulPolicy.md">MatmulPolicy</a>模板参数</p>
</td>
</tr>
<tr id="row91581116163018"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p141909471896"><a name="p141909471896"></a><a name="p141909471896"></a><a href="TSCM输入的矩阵乘.md">TSCM输入的矩阵乘</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p1115821663010"><a name="p1115821663010"></a><a name="p1115821663010"></a><a href="DataCopy.md">DataCopy</a></p>
</td>
</tr>
<tr id="row38171916133014"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p1218910471691"><a name="p1218910471691"></a><a name="p1218910471691"></a><a href="矩阵乘输出的N方向对齐.md">矩阵乘输出的N方向对齐</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p081816162305"><a name="p081816162305"></a><a name="p081816162305"></a><a href="SetCType.md">SetCType</a></p>
</td>
</tr>
<tr id="row85071417143011"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p81883473913"><a name="p81883473913"></a><a name="p81883473913"></a><a href="单次矩阵乘局部输出.md">单次矩阵乘局部输出</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p3206172503"><a name="p3206172503"></a><a name="p3206172503"></a><a href="MatmulConfig.md#table1761013213153">MatmulConfig</a>模板参数中的<a href="MatmulConfig.md#p2030383903419">isPartialOutput</a>参数</p>
</td>
</tr>
<tr id="row18629195711308"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p17187847894"><a name="p17187847894"></a><a name="p17187847894"></a><a href="AIC和AIV独立运行机制.md">AIC和AIV独立运行机制</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p36291657203019"><a name="p36291657203019"></a><a name="p36291657203019"></a><a href="MatmulConfig.md#table1761013213153">MatmulConfig</a>模板参数中的<a href="MatmulConfig.md#p9218181073719">enableMixDualMaster</a>参数</p>
</td>
</tr>
<tr id="row14178134313539"><td class="cellrowborder" valign="top" width="40.1%" headers="mcps1.2.3.1.1 "><p id="p13689165320534"><a name="p13689165320534"></a><a name="p13689165320534"></a><a href="MxMatmul场景.md">MxMatmul场景</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.9%" headers="mcps1.2.3.1.2 "><p id="p17689175365311"><a name="p17689175365311"></a><a name="p17689175365311"></a><a href="SetScaleAType.md">SetScaleAType</a>、<a href="SetScaleBType.md">SetScaleBType</a>、<a href="SetMadType.md">SetMadType</a>、<a href="SetTensorScaleA.md">SetTensorScaleA</a>、<a href="SetTensorScaleB.md">SetTensorScaleB</a></p>
</td>
</tr>
</tbody>
</table>

**表 2**  BatchMatmul功能特性表

<a name="table51511716105313"></a>
<table><thead align="left"><tr id="row11151816105317"><th class="cellrowborder" valign="top" width="40.04%" id="mcps1.2.3.1.1"><p id="p16337193014533"><a name="p16337193014533"></a><a name="p16337193014533"></a>特性描述</p>
</th>
<th class="cellrowborder" valign="top" width="59.96%" id="mcps1.2.3.1.2"><p id="p20337113075319"><a name="p20337113075319"></a><a name="p20337113075319"></a>主要涉及的API接口</p>
</th>
</tr>
</thead>
<tbody><tr id="row0151111665312"><td class="cellrowborder" rowspan="2" valign="top" width="40.04%" headers="mcps1.2.3.1.1 "><p id="p131521516145320"><a name="p131521516145320"></a><a name="p131521516145320"></a><a href="Batch-Matmul基础功能.md">BatchMatmul基础场景</a></p>
<p id="p15737133111548"><a name="p15737133111548"></a><a name="p15737133111548"></a></p>
</td>
<td class="cellrowborder" valign="top" width="59.96%" headers="mcps1.2.3.1.2 "><p id="p199019369246"><a name="p199019369246"></a><a name="p199019369246"></a><a href="IterateBatch.md#p553564441013">NORMAL</a>排布格式的BatchMatmul：<a href="IterateBatch.md">IterateBatch</a>、<a href="SetBatchInfoForNormal.md">SetBatchInfoForNormal</a></p>
</td>
</tr>
<tr id="row1773713155416"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p26136085910"><a name="p26136085910"></a><a name="p26136085910"></a><a href="IterateBatch.md#li298041002213">BSNGD</a>、<a href="IterateBatch.md#li6785191319227">SBNGD</a>、<a href="IterateBatch.md#li1922441712222">BNGS1S2</a>排布格式的BatchMatmul：<a href="IterateBatch.md">IterateBatch</a>、<a href="SetALayout.md">SetALayout</a><span>、</span><a href="SetBLayout.md">SetBLayout</a><span>、</span><a href="SetCLayout.md">SetCLayout</a><span>、</span><a href="SetBatchNum-109.md">SetBatchNum</a></p>
</td>
</tr>
<tr id="row1477101875315"><td class="cellrowborder" valign="top" width="40.04%" headers="mcps1.2.3.1.1 "><p id="p1977131865310"><a name="p1977131865310"></a><a name="p1977131865310"></a><a href="Batch-Matmul复用Bias矩阵.md">Batch Matmul复用Bias矩阵</a></p>
</td>
<td class="cellrowborder" valign="top" width="59.96%" headers="mcps1.2.3.1.2 "><p id="p106182015384"><a name="p106182015384"></a><a name="p106182015384"></a><a href="GetMMConfig.md">GetMMConfig</a>、<a href="IterateBatch.md">IterateBatch</a>、<a href="SetBatchInfoForNormal.md">SetBatchInfoForNormal</a></p>
</td>
</tr>
</tbody>
</table>

