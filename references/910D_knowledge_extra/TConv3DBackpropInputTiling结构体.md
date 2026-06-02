# TConv3DBackpropInputTiling结构体<a name="ZH-CN_TOPIC_0000002554424587"></a>

TConv3DBackpropInputTiling结构体包含Conv3DBackpropInput算子规格信息及Tiling切分算法的相关参数，被传递给Conv3DBackpropInput Kernel侧，用于数据切分、数据搬运和计算等。TConv3DBackpropInputTiling结构体的参数说明见下表。

用户通过调用[GetTiling](GetTiling-135.md)接口获取TConv3DBackpropInputTiling结构体，具体流程请参考[Conv3DBackpropInput Tiling使用说明](Conv3DBackpropInput-Tiling使用说明.md)。当前暂不支持用户自定义配置TConv3DBackpropInputTiling结构体中的参数。

**表 1**  TConv3DBackpropInputTiling结构说明

<a name="table1563162142915"></a>
<table><thead align="left"><tr id="row6563221112912"><th class="cellrowborder" valign="top" width="16.439999999999998%" id="mcps1.2.3.1.1"><p id="p1489815818568"><a name="p1489815818568"></a><a name="p1489815818568"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.56%" id="mcps1.2.3.1.2"><p id="p389885810564"><a name="p389885810564"></a><a name="p389885810564"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row166485435562"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p12655122361713"><a name="p12655122361713"></a><a name="p12655122361713"></a>batch</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p16554233171"><a name="p16554233171"></a><a name="p16554233171"></a>输入GradOutput的N，等于卷积正向输入Input的N。</p>
</td>
</tr>
<tr id="row956392122918"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p465522311174"><a name="p465522311174"></a><a name="p465522311174"></a>cin</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p0655123111717"><a name="p0655123111717"></a><a name="p0655123111717"></a>输出GradInput的Channel，等于卷积正向输入Input的Channel。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p86558236175"><a name="p86558236175"></a><a name="p86558236175"></a>cout</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1465520232173"><a name="p1465520232173"></a><a name="p1465520232173"></a>输入GradOutput的Channel。</p>
</td>
</tr>
<tr id="row1794932544319"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p182203554320"><a name="p182203554320"></a><a name="p182203554320"></a>cout1</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p3950625204320"><a name="p3950625204320"></a><a name="p3950625204320"></a>输入GradOutput的C1，等于cout/c0。</p>
</td>
</tr>
<tr id="row7514114217432"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p16191129134416"><a name="p16191129134416"></a><a name="p16191129134416"></a>cin1</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p2514114217435"><a name="p2514114217435"></a><a name="p2514114217435"></a>输出GradInput的C1，等于卷积正向输入Input的C1，等于cin/c0。</p>
</td>
</tr>
<tr id="row1824316119447"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p7808185515441"><a name="p7808185515441"></a><a name="p7808185515441"></a>cout1G</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1524301154415"><a name="p1524301154415"></a><a name="p1524301154415"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row84713317442"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p621565817445"><a name="p621565817445"></a><a name="p621565817445"></a>cin1G</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p12497191518152"><a name="p12497191518152"></a><a name="p12497191518152"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row155771959164314"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p99312013454"><a name="p99312013454"></a><a name="p99312013454"></a>c0</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p36212500359"><a name="p36212500359"></a><a name="p36212500359"></a>当前输入数据类型下C0的大小。该参数目前只支持取值为16。</p>
</td>
</tr>
<tr id="row1767895714430"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p738910314510"><a name="p738910314510"></a><a name="p738910314510"></a>c0Bits</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p8679957174316"><a name="p8679957174316"></a><a name="p8679957174316"></a>任意一个数除以c0等价的右移位数，例如c0=8则c0Bits=3，c0=16则c0Bits=4。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p86561123141716"><a name="p86561123141716"></a><a name="p86561123141716"></a>dout</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p8656223171713"><a name="p8656223171713"></a><a name="p8656223171713"></a>输入GradOutput的Depth大小，单位元素。</p>
</td>
</tr>
<tr id="row12649134314567"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p6656523121719"><a name="p6656523121719"></a><a name="p6656523121719"></a>ho</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p56561623141710"><a name="p56561623141710"></a><a name="p56561623141710"></a>输入GradOutput的Height大小，单位元素。</p>
</td>
</tr>
<tr id="row15649104314564"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p19656152317171"><a name="p19656152317171"></a><a name="p19656152317171"></a>wo</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p76565238179"><a name="p76565238179"></a><a name="p76565238179"></a>输入GradOutput的Width大小，单位元素。</p>
</td>
</tr>
<tr id="row79918445315"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p26561923131712"><a name="p26561923131712"></a><a name="p26561923131712"></a>di</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p146561523131712"><a name="p146561523131712"></a><a name="p146561523131712"></a>输出GradInput的Depth大小，等于卷积正向输入Input的Depth大小，单位元素。</p>
</td>
</tr>
<tr id="row1865135717313"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p3656192311176"><a name="p3656192311176"></a><a name="p3656192311176"></a>hi</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p865652351720"><a name="p865652351720"></a><a name="p865652351720"></a>输出GradInput的Height大小，等于卷积正向输入Input的Height大小，单位元素。</p>
</td>
</tr>
<tr id="row10687171523219"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1565620232175"><a name="p1565620232175"></a><a name="p1565620232175"></a>wi</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p7656162317176"><a name="p7656162317176"></a><a name="p7656162317176"></a>输出GradInput的Width大小，等于卷积正向输入Input的Width大小，单位元素。</p>
</td>
</tr>
<tr id="row101131455133519"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p2065682314172"><a name="p2065682314172"></a><a name="p2065682314172"></a>dk</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p76561023151717"><a name="p76561023151717"></a><a name="p76561023151717"></a>输入Weight的Depth大小，单位元素。</p>
</td>
</tr>
<tr id="row206338663217"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1665662316171"><a name="p1665662316171"></a><a name="p1665662316171"></a>hk</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p36564235178"><a name="p36564235178"></a><a name="p36564235178"></a>输入Weight的Height大小，单位元素。</p>
</td>
</tr>
<tr id="row1550011816327"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1965612341718"><a name="p1965612341718"></a><a name="p1965612341718"></a>wk</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p106563232171"><a name="p106563232171"></a><a name="p106563232171"></a>输入Weight的Width大小，单位元素。</p>
</td>
</tr>
<tr id="row115859102328"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p9656132301712"><a name="p9656132301712"></a><a name="p9656132301712"></a>group</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p565662320171"><a name="p565662320171"></a><a name="p565662320171"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row46519565324"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p126561123181714"><a name="p126561123181714"></a><a name="p126561123181714"></a>strideD</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1465652331720"><a name="p1465652331720"></a><a name="p1465652331720"></a>卷积反向计算中Stride的Depth大小，单位元素。</p>
</td>
</tr>
<tr id="row5342125410325"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p16568233170"><a name="p16568233170"></a><a name="p16568233170"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1365612317179"><a name="p1365612317179"></a><a name="p1365612317179"></a>卷积反向计算中StrideHeight大小，单位元素。</p>
</td>
</tr>
<tr id="row13899312123213"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p16561023101711"><a name="p16561023101711"></a><a name="p16561023101711"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p265602316172"><a name="p265602316172"></a><a name="p265602316172"></a>卷积反向计算中StrideWidth大小，单位元素。</p>
</td>
</tr>
<tr id="row16265157103310"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p9656423141716"><a name="p9656423141716"></a><a name="p9656423141716"></a>padFront</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p5537144832516"><a name="p5537144832516"></a><a name="p5537144832516"></a>卷积反向计算中输出矩阵GradInput Padding的Depth维度的前方向，单位元素。</p>
</td>
</tr>
<tr id="row636121063311"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p8656023181719"><a name="p8656023181719"></a><a name="p8656023181719"></a>padBack</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p156561123131713"><a name="p156561123131713"></a><a name="p156561123131713"></a>卷积反向计算中输出矩阵GradInput Padding的Depth维度的后方向，单位元素。</p>
</td>
</tr>
<tr id="row1230118112174"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p26571923191714"><a name="p26571923191714"></a><a name="p26571923191714"></a>padUp</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p474121812617"><a name="p474121812617"></a><a name="p474121812617"></a>卷积反向计算中输出矩阵GradInput Padding的Height维度的上方向，单位元素。</p>
</td>
</tr>
<tr id="row1893713371714"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p166571923151714"><a name="p166571923151714"></a><a name="p166571923151714"></a>padDown</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p365711232171"><a name="p365711232171"></a><a name="p365711232171"></a>卷积反向计算中输出矩阵GradInput Padding的Height维度的下方向，单位元素。</p>
</td>
</tr>
<tr id="row139340517177"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p116571223101710"><a name="p116571223101710"></a><a name="p116571223101710"></a>padLeft</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p065762371714"><a name="p065762371714"></a><a name="p065762371714"></a>卷积反向计算中输出矩阵GradInput Padding的Width维度的左方向，单位元素。</p>
</td>
</tr>
<tr id="row13609395176"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p065711233176"><a name="p065711233176"></a><a name="p065711233176"></a>padRight</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1657223191717"><a name="p1657223191717"></a><a name="p1657223191717"></a>卷积反向计算中输出矩阵GradInput Padding的Width维度的右方向，单位元素。</p>
</td>
</tr>
<tr id="row19141047161912"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1123685571919"><a name="p1123685571919"></a><a name="p1123685571919"></a>backpropPadTail</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p15915164791919"><a name="p15915164791919"></a><a name="p15915164791919"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row17489125704516"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p169541568468"><a name="p169541568468"></a><a name="p169541568468"></a>backpropPadUp</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p154851192710"><a name="p154851192710"></a><a name="p154851192710"></a>卷积反向计算中输入矩阵GradOutput Padding的Height维度的上方向，单位元素。</p>
</td>
</tr>
<tr id="row132072413463"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p151891898463"><a name="p151891898463"></a><a name="p151891898463"></a>backpropPadDown</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1215561012712"><a name="p1215561012712"></a><a name="p1215561012712"></a>卷积反向计算中输入矩阵GradOutput Padding的Height维度的下方向，单位元素。</p>
</td>
</tr>
<tr id="row153401922460"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p191892133462"><a name="p191892133462"></a><a name="p191892133462"></a>backpropPadLeft</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p53401218461"><a name="p53401218461"></a><a name="p53401218461"></a>卷积反向计算中输入矩阵GradOutput Padding的Width维度的左方向，单位元素。</p>
</td>
</tr>
<tr id="row051830154618"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1554261715468"><a name="p1554261715468"></a><a name="p1554261715468"></a>backpropPadRight</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p35188010461"><a name="p35188010461"></a><a name="p35188010461"></a>卷积反向计算中输入矩阵GradOutput Padding的Width维度的右方向，单位元素。</p>
</td>
</tr>
<tr id="row145035117178"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1165717230175"><a name="p1165717230175"></a><a name="p1165717230175"></a>dilationD</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1165732331719"><a name="p1165732331719"></a><a name="p1165732331719"></a>卷积反向计算中Dilation的Depth大小，单位元素。</p>
</td>
</tr>
<tr id="row1263816135177"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p13657723161718"><a name="p13657723161718"></a><a name="p13657723161718"></a>dilationH</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p15657123181717"><a name="p15657123181717"></a><a name="p15657123181717"></a>卷积反向计算中Dilation的Height大小，单位元素。</p>
</td>
</tr>
<tr id="row2058814151178"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p13657152351712"><a name="p13657152351712"></a><a name="p13657152351712"></a>dilationW</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p265782312179"><a name="p265782312179"></a><a name="p265782312179"></a>卷积反向计算中Dilation的Width大小，单位元素。</p>
</td>
</tr>
<tr id="row120385810363"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p55821571463"><a name="p55821571463"></a><a name="p55821571463"></a>al0Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1220345863611"><a name="p1220345863611"></a><a name="p1220345863611"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row11574114410461"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p18574164474615"><a name="p18574164474615"></a><a name="p18574164474615"></a>bl0Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p19574644164619"><a name="p19574644164619"></a><a name="p19574644164619"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row15431848164610"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1958295711465"><a name="p1958295711465"></a><a name="p1958295711465"></a>cl0Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p174394820469"><a name="p174394820469"></a><a name="p174394820469"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row846315463461"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p0582105717462"><a name="p0582105717462"></a><a name="p0582105717462"></a>al1Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p946314469460"><a name="p946314469460"></a><a name="p946314469460"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row102016310469"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p10201431164610"><a name="p10201431164610"></a><a name="p10201431164610"></a>bl1Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p020153116461"><a name="p020153116461"></a><a name="p020153116461"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row8476111112490"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p104891943104915"><a name="p104891943104915"></a><a name="p104891943104915"></a>singleCoreGroup</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p94771911104917"><a name="p94771911104917"></a><a name="p94771911104917"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row12648111411498"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1064814144491"><a name="p1064814144491"></a><a name="p1064814144491"></a>singleCoreCout</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p2064811145490"><a name="p2064811145490"></a><a name="p2064811145490"></a>单核M方向上计算cout数据量的大小。</p>
</td>
</tr>
<tr id="row83171163492"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p1318161684920"><a name="p1318161684920"></a><a name="p1318161684920"></a>singleCoreCout1</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p19318416154919"><a name="p19318416154919"></a><a name="p19318416154919"></a>单核上cout1的大小。</p>
</td>
</tr>
<tr id="row1588161316496"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p75111914125211"><a name="p75111914125211"></a><a name="p75111914125211"></a>singleCoreCin1</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p16938171365413"><a name="p16938171365413"></a><a name="p16938171365413"></a>单核上cin1的大小。</p>
</td>
</tr>
<tr id="row273615915491"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p139121712529"><a name="p139121712529"></a><a name="p139121712529"></a>singleCoreDin</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p573614914917"><a name="p573614914917"></a><a name="p573614914917"></a>单核上Din的大小。</p>
</td>
</tr>
<tr id="row1721137134910"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p20897131915520"><a name="p20897131915520"></a><a name="p20897131915520"></a>singleCoreHo</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p12994192317104"><a name="p12994192317104"></a><a name="p12994192317104"></a>单核K方向上计算ho数据量的大小。</p>
</td>
</tr>
<tr id="row5467194615546"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p6798195175417"><a name="p6798195175417"></a><a name="p6798195175417"></a>baseM</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p78643563313"><a name="p78643563313"></a><a name="p78643563313"></a>L0上M方向大小。</p>
</td>
</tr>
<tr id="row1321824812543"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p175145485412"><a name="p175145485412"></a><a name="p175145485412"></a>baseK</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p88615354331"><a name="p88615354331"></a><a name="p88615354331"></a>L0上K方向大小。</p>
</td>
</tr>
<tr id="row1658274213541"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p16494257195410"><a name="p16494257195410"></a><a name="p16494257195410"></a>baseN</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1487435103311"><a name="p1487435103311"></a><a name="p1487435103311"></a>L0上N方向大小。</p>
</td>
</tr>
<tr id="row192582448547"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p20478190185518"><a name="p20478190185518"></a><a name="p20478190185518"></a>baseD</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1975895313579"><a name="p1975895313579"></a><a name="p1975895313579"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row279419635516"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p285081419558"><a name="p285081419558"></a><a name="p285081419558"></a>baseBatch</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1879420665519"><a name="p1879420665519"></a><a name="p1879420665519"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row8389794553"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p42461117195512"><a name="p42461117195512"></a><a name="p42461117195512"></a>baseGroup</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p263414831918"><a name="p263414831918"></a><a name="p263414831918"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row7842172612588"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p166120482582"><a name="p166120482582"></a><a name="p166120482582"></a>stepM</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p14842102618581"><a name="p14842102618581"></a><a name="p14842102618581"></a>特征矩阵在L1中缓存的buffer M方向上baseM的倍数。</p>
</td>
</tr>
<tr id="row0743845175810"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p8239135114585"><a name="p8239135114585"></a><a name="p8239135114585"></a>stepN</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p2900175811564"><a name="p2900175811564"></a><a name="p2900175811564"></a>权重矩阵在L1中缓存的buffer N方向上baseN的倍数。</p>
</td>
</tr>
<tr id="row15479744135812"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p381945395820"><a name="p381945395820"></a><a name="p381945395820"></a>stepKa</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p29461533344"><a name="p29461533344"></a><a name="p29461533344"></a>特征矩阵在L1中缓存的buffer K方向上baseK的倍数。</p>
</td>
</tr>
<tr id="row16951943145811"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p930055665812"><a name="p930055665812"></a><a name="p930055665812"></a>stepKb</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p1991802533512"><a name="p1991802533512"></a><a name="p1991802533512"></a>权重矩阵在L1中缓存的buffer K方向上baseK的倍数。</p>
</td>
</tr>
<tr id="row1079912397584"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p13110359185816"><a name="p13110359185816"></a><a name="p13110359185816"></a>stepBatch</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p2799113910582"><a name="p2799113910582"></a><a name="p2799113910582"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row8649236105813"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p260421165919"><a name="p260421165919"></a><a name="p260421165919"></a>stepGroup</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p16501368582"><a name="p16501368582"></a><a name="p16501368582"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row155363655920"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p471413204594"><a name="p471413204594"></a><a name="p471413204594"></a>iterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p64961457193417"><a name="p64961457193417"></a><a name="p64961457193417"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row397281675914"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p17773192585915"><a name="p17773192585915"></a><a name="p17773192585915"></a>hf32Flag</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p597211618598"><a name="p597211618598"></a><a name="p597211618598"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row64531013549"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p0194191485411"><a name="p0194191485411"></a><a name="p0194191485411"></a>initOutputFlag</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p6452101545"><a name="p6452101545"></a><a name="p6452101545"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row696916541534"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p18969954185316"><a name="p18969954185316"></a><a name="p18969954185316"></a>reserved</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p3694103195420"><a name="p3694103195420"></a><a name="p3694103195420"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row1244613185917"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p18772192813594"><a name="p18772192813594"></a><a name="p18772192813594"></a>singleCoreBatch</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p84420133597"><a name="p84420133597"></a><a name="p84420133597"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row36691814195912"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p11917130125920"><a name="p11917130125920"></a><a name="p11917130125920"></a>singleCoreM</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p126698148596"><a name="p126698148596"></a><a name="p126698148596"></a>单核M方向上需要计算的数据量大小。</p>
</td>
</tr>
<tr id="row27121899592"><td class="cellrowborder" valign="top" width="16.439999999999998%" headers="mcps1.2.3.1.1 "><p id="p78841332115917"><a name="p78841332115917"></a><a name="p78841332115917"></a>singleCoreCin</p>
</td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.2.3.1.2 "><p id="p177126916598"><a name="p177126916598"></a><a name="p177126916598"></a>单核N方向上计算cin数据量的大小。</p>
</td>
</tr>
</tbody>
</table>

