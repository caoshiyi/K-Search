# TConv3DBpFilterTiling结构体<a name="ZH-CN_TOPIC_0000002554423879"></a>

TConv3DBpFilterTiling结构体包含Conv3dBackpropFilter算子规格信息及Tiling切分算法的相关参数，被传递给Conv3dBackpropFilter Kernel侧，用于数据切分、数据搬运和计算等。TConv3DBpFilterTiling结构体的参数说明见[表1](#table1563162142915)。

用户通过调用[GetTiling](GetTiling-149.md)接口获取TConv3DBpFilterTiling结构体，具体流程请参考[使用说明](Conv3DBackpropFilter-Tiling使用说明.md)。当前暂不支持用户自定义配置TConv3DBpFilterTiling结构体中的参数。

**表 1**  TConv3DBpFilterTiling结构说明

<a name="table1563162142915"></a>
<table><thead align="left"><tr id="row6563221112912"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.3.1.1"><p id="p1489815818568"><a name="p1489815818568"></a><a name="p1489815818568"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.55%" id="mcps1.2.3.1.2"><p id="p389885810564"><a name="p389885810564"></a><a name="p389885810564"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row166485435562"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p12655122361713"><a name="p12655122361713"></a><a name="p12655122361713"></a>batch</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p16554233171"><a name="p16554233171"></a><a name="p16554233171"></a>输入GradOutput的Batch，单位元素。</p>
</td>
</tr>
<tr id="row956392122918"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p465522311174"><a name="p465522311174"></a><a name="p465522311174"></a>cin</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p0655123111717"><a name="p0655123111717"></a><a name="p0655123111717"></a>输入Input的Channel，单位元素。</p>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p86558236175"><a name="p86558236175"></a><a name="p86558236175"></a>cout</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1465520232173"><a name="p1465520232173"></a><a name="p1465520232173"></a>输入GradOutput的Channel，单位元素。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p126426380412"><a name="p126426380412"></a><a name="p126426380412"></a>cin1G</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p15695449148"><a name="p15695449148"></a><a name="p15695449148"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row1864217381443"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p149212481859"><a name="p149212481859"></a><a name="p149212481859"></a>cout1G</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p178951458121413"><a name="p178951458121413"></a><a name="p178951458121413"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row1716519321955"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p86561123141716"><a name="p86561123141716"></a><a name="p86561123141716"></a>dout</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p8656223171713"><a name="p8656223171713"></a><a name="p8656223171713"></a>输入GradOutput的Depth，单位元素。</p>
</td>
</tr>
<tr id="row12649134314567"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p6656523121719"><a name="p6656523121719"></a><a name="p6656523121719"></a>ho</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p56561623141710"><a name="p56561623141710"></a><a name="p56561623141710"></a>输入GradOutput的Height，单位元素。</p>
</td>
</tr>
<tr id="row15649104314564"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p19656152317171"><a name="p19656152317171"></a><a name="p19656152317171"></a>wo</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p76565238179"><a name="p76565238179"></a><a name="p76565238179"></a>输入GradOutput的Width，单位元素。</p>
</td>
</tr>
<tr id="row79918445315"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p26561923131712"><a name="p26561923131712"></a><a name="p26561923131712"></a>di</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p146561523131712"><a name="p146561523131712"></a><a name="p146561523131712"></a>输入Input的Depth，单位元素。</p>
</td>
</tr>
<tr id="row1865135717313"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p3656192311176"><a name="p3656192311176"></a><a name="p3656192311176"></a>hi</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p865652351720"><a name="p865652351720"></a><a name="p865652351720"></a>输入Input的Height，单位元素。</p>
</td>
</tr>
<tr id="row10687171523219"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1565620232175"><a name="p1565620232175"></a><a name="p1565620232175"></a>wi</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p7656162317176"><a name="p7656162317176"></a><a name="p7656162317176"></a>输入Input的Width，单位元素。</p>
</td>
</tr>
<tr id="row101131455133519"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p2065682314172"><a name="p2065682314172"></a><a name="p2065682314172"></a>dk</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p76561023151717"><a name="p76561023151717"></a><a name="p76561023151717"></a>输出Weight的Depth，单位元素。</p>
</td>
</tr>
<tr id="row206338663217"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1665662316171"><a name="p1665662316171"></a><a name="p1665662316171"></a>hk</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p36564235178"><a name="p36564235178"></a><a name="p36564235178"></a>输出Weight的Height，单位元素。</p>
</td>
</tr>
<tr id="row1550011816327"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1965612341718"><a name="p1965612341718"></a><a name="p1965612341718"></a>wk</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p106563232171"><a name="p106563232171"></a><a name="p106563232171"></a>输出Weight的Width，单位元素。</p>
</td>
</tr>
<tr id="row115859102328"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p9656132301712"><a name="p9656132301712"></a><a name="p9656132301712"></a>group</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p565662320171"><a name="p565662320171"></a><a name="p565662320171"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row46519565324"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p126561123181714"><a name="p126561123181714"></a><a name="p126561123181714"></a>strideD</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1465652331720"><a name="p1465652331720"></a><a name="p1465652331720"></a>卷积反向计算中Stride的Depth，单位元素。</p>
</td>
</tr>
<tr id="row5342125410325"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p16568233170"><a name="p16568233170"></a><a name="p16568233170"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1365612317179"><a name="p1365612317179"></a><a name="p1365612317179"></a>卷积反向计算中Stride的Height，单位元素。</p>
</td>
</tr>
<tr id="row13899312123213"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p16561023101711"><a name="p16561023101711"></a><a name="p16561023101711"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p265602316172"><a name="p265602316172"></a><a name="p265602316172"></a>卷积反向计算中Stride的Width，单位元素。</p>
</td>
</tr>
<tr id="row16265157103310"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p9656423141716"><a name="p9656423141716"></a><a name="p9656423141716"></a>padFront</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p86568239176"><a name="p86568239176"></a><a name="p86568239176"></a>卷积反向计算中Padding的Depth维度的前方向，单位元素。</p>
</td>
</tr>
<tr id="row636121063311"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p8656023181719"><a name="p8656023181719"></a><a name="p8656023181719"></a>padBack</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p156561123131713"><a name="p156561123131713"></a><a name="p156561123131713"></a>卷积反向计算中Padding的Depth维度的后方向，单位元素。</p>
</td>
</tr>
<tr id="row1230118112174"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p26571923191714"><a name="p26571923191714"></a><a name="p26571923191714"></a>padUp</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p965752313178"><a name="p965752313178"></a><a name="p965752313178"></a>卷积反向计算中Padding的Height维度的上方向，单位元素。</p>
</td>
</tr>
<tr id="row1893713371714"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p166571923151714"><a name="p166571923151714"></a><a name="p166571923151714"></a>padDown</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p365711232171"><a name="p365711232171"></a><a name="p365711232171"></a>卷积反向计算中Padding的Height维度的下方向，单位元素。</p>
</td>
</tr>
<tr id="row139340517177"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p116571223101710"><a name="p116571223101710"></a><a name="p116571223101710"></a>padLeft</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p065762371714"><a name="p065762371714"></a><a name="p065762371714"></a>卷积反向计算中Padding的Width维度的左方向，单位元素。</p>
</td>
</tr>
<tr id="row13609395176"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p065711233176"><a name="p065711233176"></a><a name="p065711233176"></a>padRight</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1657223191717"><a name="p1657223191717"></a><a name="p1657223191717"></a>卷积反向计算中Padding的Width维度的右方向，单位元素。</p>
</td>
</tr>
<tr id="row145035117178"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1165717230175"><a name="p1165717230175"></a><a name="p1165717230175"></a>dilationD</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1165732331719"><a name="p1165732331719"></a><a name="p1165732331719"></a>卷积反向计算中Dilation的Depth，单位元素。</p>
</td>
</tr>
<tr id="row1263816135177"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p13657723161718"><a name="p13657723161718"></a><a name="p13657723161718"></a>dilationH</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p15657123181717"><a name="p15657123181717"></a><a name="p15657123181717"></a>卷积反向计算中Dilation的Height，单位元素。</p>
</td>
</tr>
<tr id="row2058814151178"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p13657152351712"><a name="p13657152351712"></a><a name="p13657152351712"></a>dilationW</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p265782312179"><a name="p265782312179"></a><a name="p265782312179"></a>卷积反向计算中Dilation的Width，单位元素。</p>
</td>
</tr>
<tr id="row165260176179"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1665782310174"><a name="p1665782310174"></a><a name="p1665782310174"></a>channelSize</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p19657182319174"><a name="p19657182319174"></a><a name="p19657182319174"></a>当前输入数据类型下C0的大小。该参数目前只支持取值为16。</p>
</td>
</tr>
<tr id="row26817511331"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p149613576341"><a name="p149613576341"></a><a name="p149613576341"></a>al0Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p17496757173413"><a name="p17496757173413"></a><a name="p17496757173413"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row169010814332"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p74977574342"><a name="p74977574342"></a><a name="p74977574342"></a>bl0Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p749775753419"><a name="p749775753419"></a><a name="p749775753419"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row4541141043319"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p15497165710343"><a name="p15497165710343"></a><a name="p15497165710343"></a>cl0Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1549718572342"><a name="p1549718572342"></a><a name="p1549718572342"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row413919139332"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p6497115714340"><a name="p6497115714340"></a><a name="p6497115714340"></a>al1Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p104975571346"><a name="p104975571346"></a><a name="p104975571346"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row91968226333"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p2497135713417"><a name="p2497135713417"></a><a name="p2497135713417"></a>bl1Pbuffer</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1497185793417"><a name="p1497185793417"></a><a name="p1497185793417"></a>1表示不使能DoubleBuffer，2表示使能DoubleBuffer。</p>
</td>
</tr>
<tr id="row1262711919174"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p88643512333"><a name="p88643512333"></a><a name="p88643512333"></a>baseM</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p78643563313"><a name="p78643563313"></a><a name="p78643563313"></a>L0上M方向大小，单位元素。</p>
</td>
</tr>
<tr id="row4760112615307"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p88617356338"><a name="p88617356338"></a><a name="p88617356338"></a>baseK</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p88615354331"><a name="p88615354331"></a><a name="p88615354331"></a>L0上K方向大小，单位元素。</p>
</td>
</tr>
<tr id="row32943020304"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p687335143319"><a name="p687335143319"></a><a name="p687335143319"></a>baseN</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1487435103311"><a name="p1487435103311"></a><a name="p1487435103311"></a>L0上N方向大小，单位元素。</p>
</td>
</tr>
<tr id="row29579325309"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p6861435193313"><a name="p6861435193313"></a><a name="p6861435193313"></a>m0</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p986735123313"><a name="p986735123313"></a><a name="p986735123313"></a>L0上最小分形M方向大小。</p>
</td>
</tr>
<tr id="row284814256918"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p168653553315"><a name="p168653553315"></a><a name="p168653553315"></a>k0</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p286163523313"><a name="p286163523313"></a><a name="p286163523313"></a>L0上最小分形K方向大小。</p>
</td>
</tr>
<tr id="row5751163523014"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p986235173313"><a name="p986235173313"></a><a name="p986235173313"></a>n0</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p10861035113319"><a name="p10861035113319"></a><a name="p10861035113319"></a>L0上最小分形N方向大小。</p>
</td>
</tr>
<tr id="row079153820307"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p58723513313"><a name="p58723513313"></a><a name="p58723513313"></a>stepM</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p53031522174010"><a name="p53031522174010"></a><a name="p53031522174010"></a>矩阵在L1中缓存的buffer M方向上baseM的倍数。</p>
</td>
</tr>
<tr id="row41755408309"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p108716355335"><a name="p108716355335"></a><a name="p108716355335"></a>stepN</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p38713553315"><a name="p38713553315"></a><a name="p38713553315"></a>矩阵在L1中缓存的buffer N方向上baseN的倍数。</p>
</td>
</tr>
<tr id="row6204742143019"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p18723533317"><a name="p18723533317"></a><a name="p18723533317"></a>stepKa</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p84801444586"><a name="p84801444586"></a><a name="p84801444586"></a>矩阵在L1中缓存的buffer K方向上baseK的倍数。</p>
</td>
</tr>
<tr id="row10143194417304"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p19871235183319"><a name="p19871235183319"></a><a name="p19871235183319"></a>stepKb</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1087535183320"><a name="p1087535183320"></a><a name="p1087535183320"></a>矩阵在L1中缓存的buffer K方向上baseK的倍数。</p>
</td>
</tr>
<tr id="row20418194618303"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p10496557113419"><a name="p10496557113419"></a><a name="p10496557113419"></a>iterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p138012333214"><a name="p138012333214"></a><a name="p138012333214"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row9616174833019"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p18739837191417"><a name="p18739837191417"></a><a name="p18739837191417"></a>bl1Bound</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1391273015141"><a name="p1391273015141"></a><a name="p1391273015141"></a>L1中载入GradOutput矩阵的最大数据量。</p>
</td>
</tr>
<tr id="row1265835033012"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1940713415140"><a name="p1940713415140"></a><a name="p1940713415140"></a>hf32Flag</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p597211618598"><a name="p597211618598"></a><a name="p597211618598"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row883245217308"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p28673573318"><a name="p28673573318"></a><a name="p28673573318"></a>singleCoreDK</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p78673513312"><a name="p78673513312"></a><a name="p78673513312"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row759165514308"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p286235193319"><a name="p286235193319"></a><a name="p286235193319"></a>singleCoreGroup</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p686133543314"><a name="p686133543314"></a><a name="p686133543314"></a>预留参数，用户无需感知。</p>
</td>
</tr>
<tr id="row053345783016"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1286133533310"><a name="p1286133533310"></a><a name="p1286133533310"></a>singleCoreCout</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p944710314447"><a name="p944710314447"></a><a name="p944710314447"></a>单核M方向上计算cout数据量的大小，单位元素。</p>
</td>
</tr>
<tr id="row1873735919307"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1186203510334"><a name="p1186203510334"></a><a name="p1186203510334"></a>singleCoreHo</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p786173583315"><a name="p786173583315"></a><a name="p786173583315"></a>单核K方向上计算ho数据量的大小，单位元素。</p>
</td>
</tr>
<tr id="row290415210318"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p16625182610919"><a name="p16625182610919"></a><a name="p16625182610919"></a>singleCoreBatch</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1084820251598"><a name="p1084820251598"></a><a name="p1084820251598"></a>单核上batch的大小，单位元素。</p>
</td>
</tr>
<tr id="row161303643116"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p138616351331"><a name="p138616351331"></a><a name="p138616351331"></a>singleCoreCin</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p158613355331"><a name="p158613355331"></a><a name="p158613355331"></a>单核N方向上计算cin数据量的大小，单位元素。</p>
</td>
</tr>
<tr id="row166822274346"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p922293516530"><a name="p922293516530"></a><a name="p922293516530"></a>totalL1Size</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1892092418416"><a name="p1892092418416"></a><a name="p1892092418416"></a>L1 size大小，单位元素。</p>
</td>
</tr>
<tr id="row44121229183418"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p177306512536"><a name="p177306512536"></a><a name="p177306512536"></a>singleCoreM</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1567214372416"><a name="p1567214372416"></a><a name="p1567214372416"></a>单核上M的大小，单位元素。</p>
</td>
</tr>
<tr id="row19839153143411"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p516345516537"><a name="p516345516537"></a><a name="p516345516537"></a>singleCoreN</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p1032594295316"><a name="p1032594295316"></a><a name="p1032594295316"></a>单核上N的大小，单位元素。</p>
</td>
</tr>
<tr id="row1417513423420"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.3.1.1 "><p id="p1535235816533"><a name="p1535235816533"></a><a name="p1535235816533"></a>singleCoreK</p>
</td>
<td class="cellrowborder" valign="top" width="83.55%" headers="mcps1.2.3.1.2 "><p id="p51341745205318"><a name="p51341745205318"></a><a name="p51341745205318"></a>单核上K的大小，单位元素。</p>
</td>
</tr>
</tbody>
</table>

