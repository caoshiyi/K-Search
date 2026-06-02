# TCubeTiling结构体<a name="ZH-CN_TOPIC_0000002523304772"></a>

TCubeTiling结构体包含Matmul Tiling切分算法的相关参数，被传递给Matmul Kernel侧，用于Matmul的切块、搬运和计算过程等。TCubeTiling结构体的参数说明见[表1](#table1563162142915)。

**表 1**  TCubeTiling结构说明

<a name="table1563162142915"></a>
<table><thead align="left"><tr id="row6563221112912"><th class="cellrowborder" valign="top" width="15.079999999999998%" id="mcps1.2.4.1.1"><p id="p1489815818568"><a name="p1489815818568"></a><a name="p1489815818568"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="8.309999999999999%" id="mcps1.2.4.1.2"><p id="p1089817585561"><a name="p1089817585561"></a><a name="p1089817585561"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.61%" id="mcps1.2.4.1.3"><p id="p389885810564"><a name="p389885810564"></a><a name="p389885810564"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row166485435562"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p2899158115617"><a name="p2899158115617"></a><a name="p2899158115617"></a>usedCoreNum</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p148992058115619"><a name="p148992058115619"></a><a name="p148992058115619"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p38991558185616"><a name="p38991558185616"></a><a name="p38991558185616"></a>使用的AI处理器核数，请根据实际情况设置。取值范围为：[1, AI处理器最大核数]。该参数与shape相关参数的关系为：usedCoreNum = (M / singleCoreM) * (N / singleCoreN)。</p>
</td>
</tr>
<tr id="row956392122918"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p1789925835613"><a name="p1789925835613"></a><a name="p1789925835613"></a>M, N, Ka, Kb</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p1889915586561"><a name="p1889915586561"></a><a name="p1889915586561"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p198991658115617"><a name="p198991658115617"></a><a name="p198991658115617"></a>A、B、C矩阵原始输入的shape大小，以元素为单位。M, Ka为A矩阵原始输入的Shape，Kb, N为B矩阵原始输入的Shape。</p>
<a name="ul649113422368"></a><a name="ul649113422368"></a><ul id="ul649113422368"><li>大小约束<div class="p" id="p359955915916"><a name="p359955915916"></a><a name="p359955915916"></a>除<span id="ph166474141701"><a name="ph166474141701"></a><a name="ph166474141701"></a>Ascend 950PR/Ascend 950DT</span>外，下述场景需要使能MatmulConfig中的<a href="MatmulConfig.md#p13786349422">intrinsicsCheck</a>参数，以完成Matmul计算。<a name="ul11644191013411"></a><a name="ul11644191013411"></a><ul id="ul11644191013411"><li>若A矩阵为ND格式，不进行转置，Ka大于65535时需要使能intrinsicsCheck参数，M无大小限制；进行转置，M大于65535时需要使能intrinsicsCheck参数，Ka无大小限制。</li><li>若B矩阵为ND格式，不进行转置，N大于65535时需要使能intrinsicsCheck参数，Kb无大小限制；进行转置，Kb大于65535时需要使能intrinsicsCheck参数，N无大小限制。</li></ul>
</div>
</li><li>对齐约束<a name="ul1055914453710"></a><a name="ul1055914453710"></a><ul id="ul1055914453710"><li>若A矩阵以NZ格式输入，则M需要以16个元素对齐，Ka需要以C0_size对齐；若B矩阵以NZ格式输入，Kb需要以C0_size对齐，N需要以16个元素对齐。</li><li>若A、B矩阵为ND格式，无对齐约束。</li></ul>
<p id="p182910378520"><a name="p182910378520"></a><a name="p182910378520"></a><strong id="b164651949135214"><a name="b164651949135214"></a><a name="b164651949135214"></a>注意：</strong>NZ格式的输入，float数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t数据类型的C0_size为32，int4b_t/fp4x2_e2m1_t/fp4x2_e1m2_t数据类型的C0_size为64。</p>
</li></ul>
</td>
</tr>
<tr id="row36481043185619"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p11899125875617"><a name="p11899125875617"></a><a name="p11899125875617"></a>singleCoreM, singleCoreN, singleCoreK</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p198997586561"><a name="p198997586561"></a><a name="p198997586561"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p6154454164619"><a name="p6154454164619"></a><a name="p6154454164619"></a>A、B、C矩阵单核内shape大小，以元素为单位。该参数取值必须大于0。</p>
<p id="p1154854114616"><a name="p1154854114616"></a><a name="p1154854114616"></a>singleCoreK = K，多核处理时不对K进行切分；singleCoreM &lt;= M；singleCoreN &lt;= N。</p>
<p id="p1815415545463"><a name="p1815415545463"></a><a name="p1815415545463"></a><strong id="b15489103520172"><a name="b15489103520172"></a><a name="b15489103520172"></a>注意</strong>：若A矩阵以NZ格式输入，则singleCoreM需要以16个元素对齐，singleCoreK需要以C0_size * fractal_num对齐；若B矩阵以NZ格式输入，则singleCoreK需要以C0_size * fractal_num对齐，singleCoreN需要以16个元素对齐。</p>
<p id="p115435418463"><a name="p115435418463"></a><a name="p115435418463"></a>NZ格式的输入，half/bfloat16_t数据类型的C0_size为16，fractal_num为1，float数据类型的C0_size为8，fractal_num为2，int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t数据类型的C0_size为32，fractal_num为1，int4b_t/fp4x2_e2m1_t/fp4x2_e1m2_t数据类型的C0_size为64，fractal_num为1。其中，fractal_num表示为满足计算中的对齐要求需要的C0_size个数。</p>
</td>
</tr>
<tr id="row16491543185617"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p17899165811566"><a name="p17899165811566"></a><a name="p17899165811566"></a>baseM, baseN, baseK</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p790045818567"><a name="p790045818567"></a><a name="p790045818567"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p6650125471110"><a name="p6650125471110"></a><a name="p6650125471110"></a>A、B、C矩阵参与一次矩阵乘指令的shape大小，以元素为单位。</p>
<p id="p7900105816569"><a name="p7900105816569"></a><a name="p7900105816569"></a>A、B、C矩阵参与一次矩阵乘的shape大小需要按分形对齐，其含义请参考<a href="Mmad.md">Mmad</a>中的数据格式说明。</p>
<p id="p18764185218271"><a name="p18764185218271"></a><a name="p18764185218271"></a><strong id="b16145148296"><a name="b16145148296"></a><a name="b16145148296"></a>注意：</strong>该参数取值必须大于0。<a href="MxMatmul场景.md">MxMatmul场景</a>，baseK必须为64的倍数。</p>
</td>
</tr>
<tr id="row12649134314567"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p490012584566"><a name="p490012584566"></a><a name="p490012584566"></a>depthA1, depthB1</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p1690075812563"><a name="p1690075812563"></a><a name="p1690075812563"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p159001319298"><a name="p159001319298"></a><a name="p159001319298"></a>A1、B1中全载基本块的份数，depthA1为A1中全载baseM * baseK的份数，depthB1为B1中全载baseN * baseK的份数。</p>
<p id="p7900165865616"><a name="p7900165865616"></a><a name="p7900165865616"></a><strong id="b19701911182911"><a name="b19701911182911"></a><a name="b19701911182911"></a>注意：</strong>该参数取值必须大于0。</p>
</td>
</tr>
<tr id="row15649104314564"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p139009583566"><a name="p139009583566"></a><a name="p139009583566"></a>stepM， stepN，stepKa，stepKb</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p59001358115617"><a name="p59001358115617"></a><a name="p59001358115617"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p173911943653"><a name="p173911943653"></a><a name="p173911943653"></a>stepM为左矩阵在A1中缓存的buffer M方向上baseM的倍数。</p>
<p id="p2900175811564"><a name="p2900175811564"></a><a name="p2900175811564"></a>stepN为右矩阵在B1中缓存的buffer N方向上baseN的倍数。</p>
<p id="p29461533344"><a name="p29461533344"></a><a name="p29461533344"></a>stepKa为左矩阵在A1中缓存的buffer Ka方向上baseK的倍数。</p>
<p id="p1991802533512"><a name="p1991802533512"></a><a name="p1991802533512"></a>stepKb为右矩阵在B1中缓存的buffer Kb方向上baseK的倍数。</p>
<p id="p144394467546"><a name="p144394467546"></a><a name="p144394467546"></a><strong id="b119021762918"><a name="b119021762918"></a><a name="b119021762918"></a>注意：</strong>该参数取值必须大于0。</p>
</td>
</tr>
<tr id="row79918445315"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p2051215216314"><a name="p2051215216314"></a><a name="p2051215216314"></a>isBias</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p151255233118"><a name="p151255233118"></a><a name="p151255233118"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p191969189218"><a name="p191969189218"></a><a name="p191969189218"></a>是否使能Bias，参数取值如下：</p>
<a name="ul2034963072116"></a><a name="ul2034963072116"></a><ul id="ul2034963072116"><li>0：不使能Bias（默认值）。</li><li>1：使能Bias。</li></ul>
<p id="p096617137251"><a name="p096617137251"></a><a name="p096617137251"></a><strong id="b118489326295"><a name="b118489326295"></a><a name="b118489326295"></a>注意：</strong>该参数不支持除上述外的其他取值，设置为其他值时参数行为未定义。</p>
</td>
</tr>
<tr id="row1865135717313"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p1620315053211"><a name="p1620315053211"></a><a name="p1620315053211"></a>transLength</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p1820412501329"><a name="p1820412501329"></a><a name="p1820412501329"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p14204135019321"><a name="p14204135019321"></a><a name="p14204135019321"></a>max(A1Length, B1Length, C1Length, BiasLength)。其中，A1Length, B1Length, C1Length, BiasLength分别表示A/B/C/Bias矩阵在计算过程中需要临时占用的UB空间大小。</p>
</td>
</tr>
<tr id="row10687171523219"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p1820410501321"><a name="p1820410501321"></a><a name="p1820410501321"></a>iterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p92041150193215"><a name="p92041150193215"></a><a name="p92041150193215"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p1420415053217"><a name="p1420415053217"></a><a name="p1420415053217"></a>一次Iterate计算出[baseM, baseN]大小的C矩阵分片，Iterate完成后，Matmul会自动偏移下一次Iterate输出的C矩阵位置，iterOrder表示自动偏移的顺序。参数取值如下：</p>
<a name="ul319720890"></a><a name="ul319720890"></a><ul id="ul319720890"><li>0：先往M轴方向偏移再往N轴方向偏移。</li><li>1：先往N轴方向偏移再往M轴方向偏移。</li></ul>
<p id="p1661846258"><a name="p1661846258"></a><a name="p1661846258"></a><strong id="b153454072918"><a name="b153454072918"></a><a name="b153454072918"></a>注意：</strong>该参数不支持除上述外的其他取值，设置为其他值时参数行为未定义。</p>
</td>
</tr>
<tr id="row101131455133519"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p107381117133611"><a name="p107381117133611"></a><a name="p107381117133611"></a>dbL0A, dbL0B,</p>
<p id="p4271194073614"><a name="p4271194073614"></a><a name="p4271194073614"></a>dbL0C</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p485064133611"><a name="p485064133611"></a><a name="p485064133611"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p1116295610311"><a name="p1116295610311"></a><a name="p1116295610311"></a>MTE1是否开启double buffer。</p>
<p id="p085014493611"><a name="p085014493611"></a><a name="p085014493611"></a>dbL0A：左矩阵MTE1是否开启double buffer；dbL0B：右矩阵MTE1是否开启double buffer；dbL0C：MMAD是否开启double buffer。参数取值如下：</p>
<a name="ul96977356551"></a><a name="ul96977356551"></a><ul id="ul96977356551"><li>1：不开启double buffer。</li><li>2：开启double buffer。</li></ul>
<p id="p186694310172"><a name="p186694310172"></a><a name="p186694310172"></a><strong id="b159801142152911"><a name="b159801142152911"></a><a name="b159801142152911"></a>注意：</strong>该参数不支持除上述外的其他取值，设置为其他值时参数行为未定义。</p>
</td>
</tr>
<tr id="row206338663217"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p102041050153220"><a name="p102041050153220"></a><a name="p102041050153220"></a>shareMode</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p32041509322"><a name="p32041509322"></a><a name="p32041509322"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p11204105013216"><a name="p11204105013216"></a><a name="p11204105013216"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row1550011816327"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p82041150183214"><a name="p82041150183214"></a><a name="p82041150183214"></a>shareL1Size</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p4204850133215"><a name="p4204850133215"></a><a name="p4204850133215"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p2933169125013"><a name="p2933169125013"></a><a name="p2933169125013"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row115859102328"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p551122813336"><a name="p551122813336"></a><a name="p551122813336"></a>shareL0CSize</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p1651828163316"><a name="p1651828163316"></a><a name="p1651828163316"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p1766061215505"><a name="p1766061215505"></a><a name="p1766061215505"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row46519565324"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p1851102819332"><a name="p1851102819332"></a><a name="p1851102819332"></a>shareUbSize</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p1051528103314"><a name="p1051528103314"></a><a name="p1051528103314"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p1261131425014"><a name="p1261131425014"></a><a name="p1261131425014"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row5342125410325"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p1451132883311"><a name="p1451132883311"></a><a name="p1451132883311"></a>batchM</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p3518285336"><a name="p3518285336"></a><a name="p3518285336"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p051192813310"><a name="p051192813310"></a><a name="p051192813310"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row13899312123213"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p175122873319"><a name="p175122873319"></a><a name="p175122873319"></a>batchN</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p165192873317"><a name="p165192873317"></a><a name="p165192873317"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p833283819533"><a name="p833283819533"></a><a name="p833283819533"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row16265157103310"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p115222814337"><a name="p115222814337"></a><a name="p115222814337"></a>singleBatchM</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p65202873320"><a name="p65202873320"></a><a name="p65202873320"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p2523286333"><a name="p2523286333"></a><a name="p2523286333"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row636121063311"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p152182833318"><a name="p152182833318"></a><a name="p152182833318"></a>singleBatchN</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p35282853311"><a name="p35282853311"></a><a name="p35282853311"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p171307421531"><a name="p171307421531"></a><a name="p171307421531"></a>该参数预留，开发者无需关注。</p>
</td>
</tr>
<tr id="row1464747139"><td class="cellrowborder" valign="top" width="15.079999999999998%" headers="mcps1.2.4.1.1 "><p id="p194641647938"><a name="p194641647938"></a><a name="p194641647938"></a>mxTypePara</p>
</td>
<td class="cellrowborder" valign="top" width="8.309999999999999%" headers="mcps1.2.4.1.2 "><p id="p54647471537"><a name="p54647471537"></a><a name="p54647471537"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="76.61%" headers="mcps1.2.4.1.3 "><p id="p157933526912"><a name="p157933526912"></a><a name="p157933526912"></a>组合参数，在<a href="MxMatmul场景.md">MxMatmul场景</a>使用 ，表示scaleA/scaleB载入L1的大小与A/B矩阵载入L1大小的倍数，具体如下：</p>
<a name="ul587817175385"></a><a name="ul587817175385"></a><ul id="ul587817175385"><li>0~6bit表示scaleA与A矩阵在K方向载入数据量的比例系数，scaleFactorKa，即scaleFactorKa=scaleA在K方向载入数据量/A矩阵在K方向载入数据量，数据范围为[1, 127]；</li><li>8~14bit表示scaleB与B矩阵在K方向载入数据量的比例系数，scaleFactorKb，即scaleFactorKb=scaleB在K方向载入数据量/B矩阵在K方向载入数据量，数据范围为[1, 127]；</li><li>16~22bit表示scaleA与A矩阵在M方向载入数据量的比例系数，scaleFactorM，即scaleFactorM=scaleA在M方向载入数据量/A矩阵在M方向载入数据量，数据范围为[1, 127]；</li><li>24~30bit表示scaleB与B矩阵在N方向载入数据量的比例系数，scaleFactorN，即scaleFactorN=scaleB在N方向载入数据量/B矩阵在N方向载入数据量，数据范围为[1, 127]；</li></ul>
<p id="p458918471366"><a name="p458918471366"></a><a name="p458918471366"></a>注意：</p>
<a name="ul1582962810398"></a><a name="ul1582962810398"></a><ul id="ul1582962810398"><li>对于scaleA矩阵，仅当Ka方向全载时，支持使能M方向的多倍载入。即baseK * stepKa * scaleFactorKa &gt;= singleCoreK时，才能设置scaleFactorM为大于1的取值。</li><li>对于scaleB矩阵，仅当Kb方向全载时，支持使能N方向的多倍载入。即baseK * stepKb * scaleFactorKb &gt;= singleCoreK时，才能设置scaleFactorN为大于1的取值。</li><li>scaleA、scaleB在M、N、K方向的载入数据量不能超过实际大小。</li><li>该参数仅在MDL模板生效。</li></ul>
</td>
</tr>
</tbody>
</table>

多数情况下，用户通过调用[GetTiling](GetTiling.md)接口获取TCubeTiling结构体，具体流程请参考[使用说明](Matmul-Tiling类使用说明.md)。如果用户自定义TCubeTiling参数，各个参数的取值需要满足[表1](#table1563162142915)和[表2](#table1275812182115)中的对应参数的约束条件。如果用户通过调用[GetTiling](GetTiling.md)接口获取TCubeTiling结构体后，需要修改调整Tiling，请参考如下TCubeTiling参数约束和性能调优推荐取值，进行参数的设置。

-   TCubeTiling参数约束

    一组合法的TCubeTiling参数需要同时满足[表2](#table1275812182115)中的所有约束条件。若Matmul对象的MatmulConfig模板为[MDL模板](MatmulConfig.md#table6981133810309)，除[表2](#table1275812182115)外，还同时需要满足[表3 MDL模板补充约束条件](#table1094616401179)。

    **表 2**  TCubeTiling约束条件

    <a name="table1275812182115"></a>
    <table><thead align="left"><tr id="row1758191841114"><th class="cellrowborder" valign="top" width="60.91%" id="mcps1.2.3.1.1"><p id="p27581518121112"><a name="p27581518121112"></a><a name="p27581518121112"></a>约束条件</p>
    </th>
    <th class="cellrowborder" valign="top" width="39.09%" id="mcps1.2.3.1.2"><p id="p12758101881112"><a name="p12758101881112"></a><a name="p12758101881112"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row37586182115"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p266613529127"><a name="p266613529127"></a><a name="p266613529127"></a>usedCoreNum &lt;= aiCoreCnt</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p275861814116"><a name="p275861814116"></a><a name="p275861814116"></a>使用核数小于等于当前AI处理器的最大核数</p>
    </td>
    </tr>
    <tr id="row197582186116"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p1559310172131"><a name="p1559310172131"></a><a name="p1559310172131"></a>baseM * baseK * sizeof(A_type) * dbL0A&lt; l0a_size</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p675831811112"><a name="p675831811112"></a><a name="p675831811112"></a>A矩阵base块不超过l0a buffer大小</p>
    </td>
    </tr>
    <tr id="row18758121841117"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p12329227133"><a name="p12329227133"></a><a name="p12329227133"></a>baseN * baseK * sizeof(B_type) * dbL0B &lt; l0b_size</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p1375831871118"><a name="p1375831871118"></a><a name="p1375831871118"></a>B矩阵base块不超过l0b buffer大小</p>
    </td>
    </tr>
    <tr id="row107581318151116"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p5448142618134"><a name="p5448142618134"></a><a name="p5448142618134"></a>baseM * baseN * sizeof(l0c_type) * dbL0C &lt; l0c_size，其中l0c_type为int32_t或者float数据类型。</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p9758191831113"><a name="p9758191831113"></a><a name="p9758191831113"></a>C矩阵base块不超过l0c buffer大小</p>
    </td>
    </tr>
    <tr id="row15758161820118"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p235915320132"><a name="p235915320132"></a><a name="p235915320132"></a>baseN * sizeof(Bias_type) &lt; biasT_size</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p154498325208"><a name="p154498325208"></a><a name="p154498325208"></a>Bias的base块不超过BiasTable buffer大小</p>
    </td>
    </tr>
    <tr id="row1775911181117"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p144681038171311"><a name="p144681038171311"></a><a name="p144681038171311"></a>stepM * stepKa * db = depthA1</p>
    <p id="p16520537142410"><a name="p16520537142410"></a><a name="p16520537142410"></a>db这里表示为左矩阵MTE2是否开启double buffer，即L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer）</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p2048112415219"><a name="p2048112415219"></a><a name="p2048112415219"></a>depthA1的取值与stepM * stepKa  * db相同</p>
    </td>
    </tr>
    <tr id="row1759161811115"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p24711944181311"><a name="p24711944181311"></a><a name="p24711944181311"></a>stepN * stepKb * db = depthB1</p>
    <p id="p6513144761117"><a name="p6513144761117"></a><a name="p6513144761117"></a>db这里表示为右矩阵MTE2是否开启double buffer，即L1是否开启double buffer，取值1（不开启double buffer）或2（开启double buffer）</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p1883135514213"><a name="p1883135514213"></a><a name="p1883135514213"></a>depthB1的取值与stepN * stepKb  * db相同</p>
    </td>
    </tr>
    <tr id="row247315891418"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p29616206165"><a name="p29616206165"></a><a name="p29616206165"></a>对于A矩阵在L1上的缓存块大小AL1Size、B矩阵在L1上的缓存块大小BL1Size必须满足：</p>
    <a name="ul4903213172015"></a><a name="ul4903213172015"></a><ul id="ul4903213172015"><li>无bias场景<p id="p10482105019521"><a name="p10482105019521"></a><a name="p10482105019521"></a>AL1Size + BL1Size &lt;= L1_size</p>
    </li></ul>
    <a name="ul17506029122011"></a><a name="ul17506029122011"></a><ul id="ul17506029122011"><li>有bias场景<p id="p145711954105215"><a name="p145711954105215"></a><a name="p145711954105215"></a>AL1Size + BL1Size + baseN * sizeof(Bias_type) &lt;= L1_size</p>
    </li></ul>
    <div class="p" id="p1690313373185"><a name="p1690313373185"></a><a name="p1690313373185"></a>其中，AL1Size、BL1Size的计算方式如下：<a name="ul2862655315"></a><a name="ul2862655315"></a><ul id="ul2862655315"><li>转置场景：<p id="p118972391815"><a name="p118972391815"></a><a name="p118972391815"></a>AL1Size = CeilDiv(baseM, C0_size) * baseK * depthA1 * sizeof(A_type)</p>
    <p id="p14315819101820"><a name="p14315819101820"></a><a name="p14315819101820"></a>BL1Size = baseN * baseK * depthB1 * sizeof(B_type)</p>
    </li><li>非转置场景：<p id="p11567631171816"><a name="p11567631171816"></a><a name="p11567631171816"></a>AL1Size = baseM * baseK * depthA1 * sizeof(A_type)</p>
    <p id="p369942812186"><a name="p369942812186"></a><a name="p369942812186"></a>BL1Size = CeilDiv(baseN, C0_size)* baseK * depthB1 * sizeof(B_type)</p>
    </li></ul>
    </div>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p2047419821417"><a name="p2047419821417"></a><a name="p2047419821417"></a>A矩阵、B矩阵和Bias在L1缓存块满足L1 buffer大小限制；</p>
    <p id="p4767112213712"><a name="p4767112213712"></a><a name="p4767112213712"></a><strong id="b143344328214"><a name="b143344328214"></a><a name="b143344328214"></a>注意：</strong>float数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t数据类型的C0_size为32，int4b_t/fp4x2_e2m1_t/fp4x2_e1m2_t数据类型的C0_size为64。</p>
    </td>
    </tr>
    <tr id="row25793104147"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p1496513306143"><a name="p1496513306143"></a><a name="p1496513306143"></a>baseM * baseK, baseK * baseN和baseM * baseN按照NZ格式的分形对齐</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p1579710111417"><a name="p1579710111417"></a><a name="p1579710111417"></a>A矩阵、B矩阵、C矩阵的base块需要满足对齐约束：</p>
    <a name="ul02397128219"></a><a name="ul02397128219"></a><ul id="ul02397128219"><li>baseM和baseN需要以16个元素对齐；A矩阵非转置且B矩阵转置场景，baseK需要以C0_size对齐；其余场景（A矩阵转置或B矩阵非转置场景），baseK以16个元素对齐；</li></ul>
    <p id="p123912122213"><a name="p123912122213"></a><a name="p123912122213"></a><strong id="b14239512224"><a name="b14239512224"></a><a name="b14239512224"></a>注意：</strong>float/int32_t数据类型的C0_size为8，half/bfloat16_t数据类型的C0_size为16，int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t数据类型的C0_size为32，int4b_t/fp4x2_e2m1_t/fp4x2_e1m2_t数据类型的C0_size为64。</p>
    </td>
    </tr>
    <tr id="row7204716162516"><td class="cellrowborder" valign="top" width="60.91%" headers="mcps1.2.3.1.1 "><p id="p1477216333352"><a name="p1477216333352"></a><a name="p1477216333352"></a><a href="MxMatmul场景.md">MxMatmul场景</a>，如果A与B矩阵的位置同时为GM，对singleCoreK没有特殊限制，在这种情况下，若scaleA和scaleB的K方向大小（即Ceil(singleCoreK, 32)）为奇数，用户需自行在scaleA和scaleB的K方向补0至偶数；对于其它A、B矩阵逻辑位置的组合情况，即A与B矩阵的位置不同时为GM，singleCoreK以32个元素向上对齐后的数值必须是32的偶数倍；</p>
    <p id="p542513380355"><a name="p542513380355"></a><a name="p542513380355"></a>输入数据类型为fp4x2_e2m1_t/fp4x2_e1m2_t时，内轴必须为偶数。</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.09%" headers="mcps1.2.3.1.2 "><p id="p16264723163518"><a name="p16264723163518"></a><a name="p16264723163518"></a>scaleA/scaleB的数据类型是fp8_e8m0_t，K方向必须2字节连续，scaleA/scaleB的K方向是A/B矩阵K的1/32；A与B矩阵的位置不同时为GM时，singleCoreK以32个元素向上对齐后的数值，必须是32的偶数倍。</p>
    <p id="p4610152911425"><a name="p4610152911425"></a><a name="p4610152911425"></a>在MxMatmul场景，输入数据类型为fp4x2_e2m1_t/fp4x2_e1m2_t，计算时的最小单元为8字节，需要将2个4字节的元素拼成一个8字节进行计算，内轴必须为偶数。</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 3**  MDL模板补充约束条件

    <a name="table1094616401179"></a>
    <table><thead align="left"><tr id="row1594614011711"><th class="cellrowborder" valign="top" width="60.89%" id="mcps1.2.3.1.1"><p id="p1167941591819"><a name="p1167941591819"></a><a name="p1167941591819"></a>约束条件</p>
    </th>
    <th class="cellrowborder" valign="top" width="39.11%" id="mcps1.2.3.1.2"><p id="p767910159185"><a name="p767910159185"></a><a name="p767910159185"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row11946104010179"><td class="cellrowborder" valign="top" width="60.89%" headers="mcps1.2.3.1.1 "><p id="p2035217311187"><a name="p2035217311187"></a><a name="p2035217311187"></a>Ka不全载时，即Ka / baseK &gt; stepKa，stepM = 1</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.11%" headers="mcps1.2.3.1.2 "><p id="p1794616408177"><a name="p1794616408177"></a><a name="p1794616408177"></a>K方向非全载时，M方向只能逐块搬运</p>
    </td>
    </tr>
    <tr id="row79461840151716"><td class="cellrowborder" valign="top" width="60.89%" headers="mcps1.2.3.1.1 "><p id="p117637151820"><a name="p117637151820"></a><a name="p117637151820"></a>Kb不全载时，即Kb / baseK &gt; stepKb，stepN = 1</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.11%" headers="mcps1.2.3.1.2 "><p id="p10350104814157"><a name="p10350104814157"></a><a name="p10350104814157"></a>K方向非全载时，N方向只能逐块搬运</p>
    </td>
    </tr>
    <tr id="row594624021712"><td class="cellrowborder" valign="top" width="60.89%" headers="mcps1.2.3.1.1 "><p id="p145758516182"><a name="p145758516182"></a><a name="p145758516182"></a>kaStepIter_ % kbStepIter_ = 0或者kbStepIter_ % kaStepIter_ = 0</p>
    <p id="p5936658161817"><a name="p5936658161817"></a><a name="p5936658161817"></a>kaStepIter_ = CeilDiv(tiling_-&gt;singleCoreK_, tiling_-&gt;baseK * tiling_-&gt;stepKa)</p>
    <p id="p11177114191911"><a name="p11177114191911"></a><a name="p11177114191911"></a>kbStepIter_ = CeilDiv(tiling_-&gt;singleCoreK_, tiling_-&gt;baseK * tiling_-&gt;stepKb)</p>
    </td>
    <td class="cellrowborder" valign="top" width="39.11%" headers="mcps1.2.3.1.2 "><p id="p19541146171417"><a name="p19541146171417"></a><a name="p19541146171417"></a>MDL模板K方向循环搬运要求Ka和Kb方向迭代次数为倍数关系</p>
    <p id="p755282114307"><a name="p755282114307"></a><a name="p755282114307"></a>kaStepIter_ ：Ka方向循环搬运迭代次数</p>
    <p id="p19779185616305"><a name="p19779185616305"></a><a name="p19779185616305"></a>kbStepIter_ ：Kb方向循环搬运迭代次数</p>
    </td>
    </tr>
    </tbody>
    </table>

-   性能调优推荐取值

    根据Tiling调优经验，部分TCubeTiling参数值或取值方式推荐如下：

    -   base块推荐\(baseM, baseN, baseK\)：\(128, 256, 64\)
    -   dbL0A / dbL0B = 2
    -   depthA1 / \(stepM \* stepKa\) = 2
    -   depthB1 / \(stepN \* stepKb\) = 2
    -   优先设置参数stepKa/stepKb，使得K方向全载，再考虑M方向或N方向全载

