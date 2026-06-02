# MatmulConfig<a name="ZH-CN_TOPIC_0000002523303586"></a>

模板参数MatmulConfig，用于配置Matmul模板信息以及相关的配置参数。不配置默认使能Norm模板，Norm模板的介绍请参考[表 模板特性](#table6981133810309)。MatmulConfig的参数说明见[表2](#table1761013213153)。MatmulConfig的定义方式有：

-   该模板参数可选取提供的模板之一，当前提供的MatmulConfig模板取值范围为【CFG\_NORM、CFG\_MDL、CFG\_IBSHARE\_NORM、MM\_CFG\_BB】，分别对应默认的Norm、MDL、IBShare、BasicBlock模板。各模板的介绍请参考[表1](#table6981133810309)。
-   该模板参数可以基于各类获取模板的接口，自定义模板参数配置，获取自定义模板。各类获取模板的接口包括：[GetNormalConfig](GetNormalConfig.md)、[GetMDLConfig](GetMDLConfig.md)、[GetSpecialMDLConfig](GetSpecialMDLConfig.md)、[GetIBShareNormConfig](GetIBShareNormConfig.md)、[GetBasicConfig](GetBasicConfig.md)。
-   另外，MatmulConfig可拆分为[MatmulShapeParams](GetMMConfig.md#table16317184295116)、[MatmulQuantParams](GetMMConfig.md#table8313111211573)、[MatmulBatchParams](GetMMConfig.md#table15129204644)、[MatmulFuncParams](GetMMConfig.md#table66217141862)二级子Config，使用[GetMMConfig](GetMMConfig.md)接口，设置需要的二级子Config和[MatmulConfigMode](GetMMConfig.md#table17837129144319)，可以更加灵活的获取自定义的模板参数配置MatmulConfig。

**表 1**  模板特性

<a name="table6981133810309"></a>
<table><thead align="left"><tr id="row89821538123014"><th class="cellrowborder" valign="top" width="12.07%" id="mcps1.2.5.1.1"><p id="p798219386309"><a name="p798219386309"></a><a name="p798219386309"></a>模板</p>
</th>
<th class="cellrowborder" valign="top" width="47.48%" id="mcps1.2.5.1.2"><p id="p298220381301"><a name="p298220381301"></a><a name="p298220381301"></a>实现</p>
</th>
<th class="cellrowborder" valign="top" width="19.8%" id="mcps1.2.5.1.3"><p id="p17982143810307"><a name="p17982143810307"></a><a name="p17982143810307"></a>优点</p>
</th>
<th class="cellrowborder" valign="top" width="20.65%" id="mcps1.2.5.1.4"><p id="p135161238183310"><a name="p135161238183310"></a><a name="p135161238183310"></a>推荐使用场景</p>
</th>
</tr>
</thead>
<tbody><tr id="row898216383307"><td class="cellrowborder" valign="top" width="12.07%" headers="mcps1.2.5.1.1 "><p id="p159827389308"><a name="p159827389308"></a><a name="p159827389308"></a>Norm</p>
</td>
<td class="cellrowborder" valign="top" width="47.48%" headers="mcps1.2.5.1.2 "><p id="p20982163818301"><a name="p20982163818301"></a><a name="p20982163818301"></a>支持L1缓存多个基本块，MTE2分多次从GM搬运基本块到L1，每次搬运一份基本块，已搬的基本块不清空。（举例说明：Tiling结构体中的<a href="TCubeTiling结构体.md#p490012584566">depthA1</a>=6，代表搬入6份A矩阵基本块到L1，1次搬运一份基本块，MTE2进行6次搬运）。</p>
</td>
<td class="cellrowborder" valign="top" width="19.8%" headers="mcps1.2.5.1.3 "><p id="p15982133873012"><a name="p15982133873012"></a><a name="p15982133873012"></a>可以提前启动MTE1流水（因为搬1份基本块就可以做MTE1后面的运算）。</p>
</td>
<td class="cellrowborder" valign="top" width="20.65%" headers="mcps1.2.5.1.4 "><p id="p1516143812333"><a name="p1516143812333"></a><a name="p1516143812333"></a>默认使能Norm模板。</p>
</td>
</tr>
<tr id="row7982133823017"><td class="cellrowborder" valign="top" width="12.07%" headers="mcps1.2.5.1.1 "><p id="p109823386305"><a name="p109823386305"></a><a name="p109823386305"></a>MDL，SpecialMDL</p>
</td>
<td class="cellrowborder" valign="top" width="47.48%" headers="mcps1.2.5.1.2 "><p id="p109827382301"><a name="p109827382301"></a><a name="p109827382301"></a>支持L1缓存多个基本块，MTE2从GM到L1的搬运为一次性“大包”搬运。（举例说明：Tiling结构体中的<a href="TCubeTiling结构体.md#p490012584566">depthA1</a>=6，代表一次性搬入6份A矩阵基本块到L1，MTE2进行1次搬运）。MDL模板与SpecialMDL模板的差异见<a href="#table1761013213153">表2</a>。</p>
</td>
<td class="cellrowborder" valign="top" width="19.8%" headers="mcps1.2.5.1.3 "><p id="p11982133853019"><a name="p11982133853019"></a><a name="p11982133853019"></a>对于一般的大shape场景，可以减少MTE2的循环搬运，提升性能。</p>
</td>
<td class="cellrowborder" valign="top" width="20.65%" headers="mcps1.2.5.1.4 "><p id="p12516173815332"><a name="p12516173815332"></a><a name="p12516173815332"></a>大shape场景。</p>
</td>
</tr>
<tr id="row18205178191611"><td class="cellrowborder" valign="top" width="12.07%" headers="mcps1.2.5.1.1 "><p id="p14205158141611"><a name="p14205158141611"></a><a name="p14205158141611"></a>IBShare</p>
</td>
<td class="cellrowborder" valign="top" width="47.48%" headers="mcps1.2.5.1.2 "><p id="p1718483210164"><a name="p1718483210164"></a><a name="p1718483210164"></a>MIX场景下，A矩阵或B矩阵GM地址相同的时候，通过共享L1 Buffer，减少MTE2搬运。</p>
</td>
<td class="cellrowborder" valign="top" width="19.8%" headers="mcps1.2.5.1.3 "><p id="p9205148141613"><a name="p9205148141613"></a><a name="p9205148141613"></a>减少MTE2搬运，提升性能。</p>
</td>
<td class="cellrowborder" valign="top" width="20.65%" headers="mcps1.2.5.1.4 "><p id="p1620612811613"><a name="p1620612811613"></a><a name="p1620612811613"></a>MIX场景多个AIV的A矩阵或B矩阵GM地址相同。</p>
<p id="p17971530111117"><a name="p17971530111117"></a><a name="p17971530111117"></a>注意：IBShare模板要求多个AIV复用的A/B矩阵必须在L1 Buffer上全载。</p>
</td>
</tr>
<tr id="row10672034184611"><td class="cellrowborder" valign="top" width="12.07%" headers="mcps1.2.5.1.1 "><p id="p56743418461"><a name="p56743418461"></a><a name="p56743418461"></a>BasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="47.48%" headers="mcps1.2.5.1.2 "><p id="p12671334114616"><a name="p12671334114616"></a><a name="p12671334114616"></a>在无<a href="多核非对齐切分.md">尾块</a>的场景，基本块大小确定的情况下，通过<a href="GetBasicConfig.md">GetBasicConfig</a>接口配置输入的基本块大小，固定MTE1每次搬运的矩阵大小及每次矩阵乘计算的矩阵大小，减少参数计算量。</p>
</td>
<td class="cellrowborder" valign="top" width="19.8%" headers="mcps1.2.5.1.3 "><p id="p11671934134618"><a name="p11671934134618"></a><a name="p11671934134618"></a>减少MTE1矩阵搬运和矩阵乘计算过程中的参数计算开销。</p>
</td>
<td class="cellrowborder" valign="top" width="20.65%" headers="mcps1.2.5.1.4 "><p id="p13673341467"><a name="p13673341467"></a><a name="p13673341467"></a>无尾块，基本块（baseM,baseN）大小确定。</p>
</td>
</tr>
<tr id="row9694133953718"><td class="cellrowborder" valign="top" width="12.07%" headers="mcps1.2.5.1.1 "><p id="p14694139193713"><a name="p14694139193713"></a><a name="p14694139193713"></a>SpecialBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="47.48%" headers="mcps1.2.5.1.2 "><p id="p8694193923718"><a name="p8694193923718"></a><a name="p8694193923718"></a>在无<a href="多核非对齐切分.md">尾块</a>的场景，基本块和单核内shape大小确定的情况下，通过配置MatmulConfig参数，在BasicBlock模板的基础上进一步消除头开销Scalar计算。</p>
</td>
<td class="cellrowborder" valign="top" width="19.8%" headers="mcps1.2.5.1.3 "><p id="p13694639173720"><a name="p13694639173720"></a><a name="p13694639173720"></a>减少MTE1矩阵搬运和矩阵乘计算过程中的参数计算开销，并进一步消除头开销Scalar计算。</p>
</td>
<td class="cellrowborder" valign="top" width="20.65%" headers="mcps1.2.5.1.4 "><p id="p16941939123713"><a name="p16941939123713"></a><a name="p16941939123713"></a>无尾块，基本块（baseM,baseN）和单核内shape（singleCoreM,singleCoreN）大小确定。</p>
<p id="p31371744194218"><a name="p31371744194218"></a><a name="p31371744194218"></a>注意：相同场景下，推荐使用<a href="GetMatmulApiTiling.md">常量化</a>来获得更好的性能收益。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  MatmulConfig参数说明

<a name="table1761013213153"></a>
<table><thead align="left"><tr id="row10610153241512"><th class="cellrowborder" valign="top" width="21.330000000000002%" id="mcps1.2.4.1.1"><p id="p161012320150"><a name="p161012320150"></a><a name="p161012320150"></a>参数</p>
</th>
<th class="cellrowborder" valign="top" width="55.190000000000005%" id="mcps1.2.4.1.2"><p id="p106101732111516"><a name="p106101732111516"></a><a name="p106101732111516"></a>说明</p>
</th>
<th class="cellrowborder" valign="top" width="23.480000000000004%" id="mcps1.2.4.1.3"><p id="p1433210535817"><a name="p1433210535817"></a><a name="p1433210535817"></a>支持模板：Norm, MDL, SpecialMDL, IBShare, BasicBlock, SpecialBasicBlock</p>
</th>
</tr>
</thead>
<tbody><tr id="row46108327151"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p14610173217151"><a name="p14610173217151"></a><a name="p14610173217151"></a>doNorm</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p261173291512"><a name="p261173291512"></a><a name="p261173291512"></a>使能Norm模板。参数取值如下：</p>
<a name="ul15611193211154"></a><a name="ul15611193211154"></a><ul id="ul15611193211154"><li>true：使能Norm模板。</li><li>false：不使能Norm模板。</li></ul>
<p id="p26111132121517"><a name="p26111132121517"></a><a name="p26111132121517"></a>不指定模板的情况默认使能Norm模板。</p>
<p id="p8135721367"><a name="p8135721367"></a><a name="p8135721367"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p23337531816"><a name="p23337531816"></a><a name="p23337531816"></a>Norm</p>
</td>
</tr>
<tr id="row6337171194114"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p18611113217150"><a name="p18611113217150"></a><a name="p18611113217150"></a>doBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p9611193217155"><a name="p9611193217155"></a><a name="p9611193217155"></a>使能BasicBlock模板。模板参数取值如下：</p>
<a name="ul32395615508"></a><a name="ul32395615508"></a><ul id="ul32395615508"><li>true：使能BasicBlock模板。</li><li>false：不使能BasicBlock模板。</li></ul>
<p id="p12885122075118"><a name="p12885122075118"></a><a name="p12885122075118"></a>调用GetBasicConfig接口获取BasicBlock模板时，该参数被置为true。注意：</p>
<a name="ul151532016526"></a><a name="ul151532016526"></a><ul id="ul151532016526"><li>BasicBlock模板暂不支持输入为int8_t, int4_t数据类型的A、B矩阵，支持half/float/bfloat16_t数据类型的A、B矩阵。</li><li>BasicBlock模板暂不支持A矩阵为标量数据Scalar或向量数据Vector。</li><li>BasicBlock模板暂不支持ScheduleType::OUTER_PRODUCT的数据搬运模式。</li></ul>
<p id="p162207162097"><a name="p162207162097"></a><a name="p162207162097"></a><span id="ph1722041614915"><a name="ph1722041614915"></a><a name="ph1722041614915"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</p>
<p id="p10924173410583"><a name="p10924173410583"></a><a name="p10924173410583"></a></p>
<p id="p19924634105816"><a name="p19924634105816"></a><a name="p19924634105816"></a></p>
<p id="p11287139273"><a name="p11287139273"></a><a name="p11287139273"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1670563664118"><a name="p1670563664118"></a><a name="p1670563664118"></a>BasicBlock</p>
</td>
</tr>
<tr id="row18173114024816"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p36121832151520"><a name="p36121832151520"></a><a name="p36121832151520"></a>doMultiDataLoad</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1861223271512"><a name="p1861223271512"></a><a name="p1861223271512"></a>使能MDL模板。参数取值如下：</p>
<a name="ul11612132151519"></a><a name="ul11612132151519"></a><ul id="ul11612132151519"><li>true：使能MDL模板。</li><li>false：不使能MDL模板。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p103335531818"><a name="p103335531818"></a><a name="p103335531818"></a>MDL</p>
</td>
</tr>
<tr id="row3158195518240"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p57371018162511"><a name="p57371018162511"></a><a name="p57371018162511"></a>basicM</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p7737518152511"><a name="p7737518152511"></a><a name="p7737518152511"></a><span id="ph4737101818259"><a name="ph4737101818259"></a><a name="ph4737101818259"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseM参数含义相同，Matmul计算时base块M轴长度，以元素为单位。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p673713183250"><a name="p673713183250"></a><a name="p673713183250"></a>BasicBlock、SpecialBasicBlock</p>
</td>
</tr>
<tr id="row6737191152520"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p17737718202511"><a name="p17737718202511"></a><a name="p17737718202511"></a>basicN</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1573811813258"><a name="p1573811813258"></a><a name="p1573811813258"></a><span id="ph15738181802511"><a name="ph15738181802511"></a><a name="ph15738181802511"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseN参数含义相同，Matmul计算时base块N轴长度，以元素为单位。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1738618102519"><a name="p1738618102519"></a><a name="p1738618102519"></a>BasicBlock、SpecialBasicBlock</p>
</td>
</tr>
<tr id="row5489135815245"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p673817181254"><a name="p673817181254"></a><a name="p673817181254"></a>basicK</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p473851819255"><a name="p473851819255"></a><a name="p473851819255"></a><span id="ph5738141810250"><a name="ph5738141810250"></a><a name="ph5738141810250"></a>与<a href="TCubeTiling结构体.md">TCubeTiling结构体</a>中的baseK参数含义相同，Matmul计算时base块K轴长度，以元素为单位。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p373881817259"><a name="p373881817259"></a><a name="p373881817259"></a>BasicBlock、SpecialBasicBlock</p>
</td>
</tr>
<tr id="row10157155844914"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p13786349422"><a name="p13786349422"></a><a name="p13786349422"></a>intrinsicsCheck</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p19428173891615"><a name="p19428173891615"></a><a name="p19428173891615"></a><span id="ph1075313367139"><a name="ph1075313367139"></a><a name="ph1075313367139"></a>当左矩阵或右矩阵在单核上内轴（即尾轴）大于等于65535（元素个数）时，是否使能循环执行数据从<span id="ph610031519596"><a name="ph610031519596"></a><a name="ph610031519596"></a>Global Memory</span>到<span id="ph6551115913423"><a name="ph6551115913423"></a><a name="ph6551115913423"></a><span id="ph455120597421"><a name="ph455120597421"></a><a name="ph455120597421"></a>L1 Buffer</span></span>的搬入。例如，左矩阵A[M, K]，单核上的内轴数据singleCoreK大于65535，配置该参数为true后，API内部通过循环执行数据的搬入。参数取值如下：</span></p>
<a name="ul143331631192217"></a><a name="ul143331631192217"></a><ul id="ul143331631192217"><li><span id="ph97531536101313"><a name="ph97531536101313"></a><a name="ph97531536101313"></a>false：当左矩阵或右矩阵在单核上内轴大于等于65535时，不使能循环执行数据的搬入（默认值）。</span></li><li><span id="ph1675453618137"><a name="ph1675453618137"></a><a name="ph1675453618137"></a>true：当左矩阵或右矩阵在单核上内轴大于等于65535时，使能循环执行数据的搬入。</span></li></ul>
<p id="p131018512498"><a name="p131018512498"></a><a name="p131018512498"></a><span id="ph1875573611319"><a name="ph1875573611319"></a><a name="ph1875573611319"></a>注意：MxMatmul场景仅支持false。</span></p>
<p id="p11148123114812"><a name="p11148123114812"></a><a name="p11148123114812"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p33336531686"><a name="p33336531686"></a><a name="p33336531686"></a>所有模板</p>
</td>
</tr>
<tr id="row1610193125011"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1960754911593"><a name="p1960754911593"></a><a name="p1960754911593"></a>isNBatch</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p176131132121514"><a name="p176131132121514"></a><a name="p176131132121514"></a><span id="ph2094817481611"><a name="ph2094817481611"></a><a name="ph2094817481611"></a>是否多Batch输入多Batch输出。仅对BatchMatmul有效，使能该参数后，仅支持Norm模板，且需调用<a href="IterateNBatch.md">IterateNBatch</a>实现多Batch输入多Batch输出。参数取值如下：</span></p>
<a name="ul261310324151"></a><a name="ul261310324151"></a><ul id="ul261310324151"><li><span id="ph1948124141620"><a name="ph1948124141620"></a><a name="ph1948124141620"></a>false：不使能多Batch（默认值）。</span></li><li><span id="ph894916421614"><a name="ph894916421614"></a><a name="ph894916421614"></a>true：使能多Batch。</span><p id="p557316104491"><a name="p557316104491"></a><a name="p557316104491"></a></p>
</li></ul>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p103330531684"><a name="p103330531684"></a><a name="p103330531684"></a>Norm</p>
<p id="p19488183051515"><a name="p19488183051515"></a><a name="p19488183051515"></a></p>
</td>
</tr>
<tr id="row123141467505"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p115998191507"><a name="p115998191507"></a><a name="p115998191507"></a>enVecND2NZ</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p11614143215153"><a name="p11614143215153"></a><a name="p11614143215153"></a><span id="ph932733018176"><a name="ph932733018176"></a><a name="ph932733018176"></a>使能通过vector指令进行ND2NZ。使能时需要设置<a href="SetLocalWorkspace.md">SetLocalWorkspace</a>。参数取值如下：</span></p>
<a name="ul196141232181511"></a><a name="ul196141232181511"></a><ul id="ul196141232181511"><li><span id="ph8332430171715"><a name="ph8332430171715"></a><a name="ph8332430171715"></a>false：不使能通过vector指令进行ND2NZ（默认值）。</span></li><li><span id="ph123360306178"><a name="ph123360306178"></a><a name="ph123360306178"></a>true：使能通过vector指令进行ND2NZ。</span></li></ul>
<p id="p78127493504"><a name="p78127493504"></a><a name="p78127493504"></a><span id="ph173386306173"><a name="ph173386306173"></a><a name="ph173386306173"></a>注意：MxMatmul场景仅支持false。</span></p>
<p id="p7315204555918"><a name="p7315204555918"></a><a name="p7315204555918"></a></p>
<p id="p11906113245515"><a name="p11906113245515"></a><a name="p11906113245515"></a></p>
<p id="p17150111316542"><a name="p17150111316542"></a><a name="p17150111316542"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p183331753782"><a name="p183331753782"></a><a name="p183331753782"></a>所有模板</p>
</td>
</tr>
<tr id="row123661830162619"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p9841058182613"><a name="p9841058182613"></a><a name="p9841058182613"></a>doSpecialBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1842584265"><a name="p1842584265"></a><a name="p1842584265"></a>使能SpecialBasicBlock模板。参数取值如下：</p>
<a name="ul158495872615"></a><a name="ul158495872615"></a><ul id="ul158495872615"><li>true：使能SpecialBasicBlock模板。</li><li>false：不使能SpecialBasicBlock模板。</li></ul>
<p id="p168425822611"><a name="p168425822611"></a><a name="p168425822611"></a>本质上也是BasicBlock模板，但消除了头开销Scalar计算。</p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1684558122613"><a name="p1684558122613"></a><a name="p1684558122613"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row08782018275"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1069545415287"><a name="p1069545415287"></a><a name="p1069545415287"></a>doMTE2Preload</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1469615418285"><a name="p1469615418285"></a><a name="p1469615418285"></a><span id="ph069611540287"><a name="ph069611540287"></a><a name="ph069611540287"></a>在MTE2流水间隙较大，且M/N数值较大时可通过该参数开启对应M/N方向的预加载功能，开启后能减小MTE2间隙，提升性能。预加载功能仅在MDL模板有效（不支持SpecialMDL模板）。参数取值如下：</span></p>
<a name="ul1669635442817"></a><a name="ul1669635442817"></a><ul id="ul1669635442817"><li><span id="ph5696754102814"><a name="ph5696754102814"></a><a name="ph5696754102814"></a>0：不开启（默认值）。</span></li><li><span id="ph13696125492810"><a name="ph13696125492810"></a><a name="ph13696125492810"></a>1：开启M方向preload。</span></li><li><span id="ph19696954162810"><a name="ph19696954162810"></a><a name="ph19696954162810"></a>2：开启N方向preload。</span></li></ul>
<p id="p26964540281"><a name="p26964540281"></a><a name="p26964540281"></a><span id="ph06968546284"><a name="ph06968546284"></a><a name="ph06968546284"></a>注意：开启M/N方向的预加载功能时需保证K全载且M/N方向开启<a href="DoubleBuffer.md">DoubleBuffer</a>；其中，M方向的K全载条件为：singleCoreK/baseK &lt;= stepKa；N方向的K全载条件为：singleCoreK/baseK &lt;= stepKb。</span></p>
<p id="p15756194417562"><a name="p15756194417562"></a><a name="p15756194417562"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1269655492812"><a name="p1269655492812"></a><a name="p1269655492812"></a>MDL</p>
</td>
</tr>
<tr id="row7380132910274"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1537113372811"><a name="p1537113372811"></a><a name="p1537113372811"></a>singleCoreM</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p143711633142817"><a name="p143711633142817"></a><a name="p143711633142817"></a><span id="ph437113316283"><a name="ph437113316283"></a><a name="ph437113316283"></a>单核内M轴shape大小，以元素为单位。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p15371113342816"><a name="p15371113342816"></a><a name="p15371113342816"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row14221935142717"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p193711533182816"><a name="p193711533182816"></a><a name="p193711533182816"></a>singleCoreN</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p12371113317285"><a name="p12371113317285"></a><a name="p12371113317285"></a><span id="ph16371183352817"><a name="ph16371183352817"></a><a name="ph16371183352817"></a>单核内N轴shape大小，以元素为单位。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p20371123322815"><a name="p20371123322815"></a><a name="p20371123322815"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row2884203882713"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p163721933122816"><a name="p163721933122816"></a><a name="p163721933122816"></a>singleCoreK</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p037233342816"><a name="p037233342816"></a><a name="p037233342816"></a><span id="ph1372113317288"><a name="ph1372113317288"></a><a name="ph1372113317288"></a>单核内K轴shape大小，以元素为单位。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p137219332280"><a name="p137219332280"></a><a name="p137219332280"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row14718124132719"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p63721133102819"><a name="p63721133102819"></a><a name="p63721133102819"></a>stepM</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p9372173312816"><a name="p9372173312816"></a><a name="p9372173312816"></a><span id="ph837233332818"><a name="ph837233332818"></a><a name="ph837233332818"></a>左矩阵在A1中缓存的bufferM方向上baseM的倍数。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p937211335284"><a name="p937211335284"></a><a name="p937211335284"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row1439143292717"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1372143317281"><a name="p1372143317281"></a><a name="p1372143317281"></a>stepN</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1337203352814"><a name="p1337203352814"></a><a name="p1337203352814"></a><span id="ph15372123312815"><a name="ph15372123312815"></a><a name="ph15372123312815"></a>右矩阵在B1中缓存的bufferN方向上baseN的倍数。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1937216339283"><a name="p1937216339283"></a><a name="p1937216339283"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row18521626182719"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p437243312815"><a name="p437243312815"></a><a name="p437243312815"></a>baseMN</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p183722033192810"><a name="p183722033192810"></a><a name="p183722033192810"></a>baseM*baseN的大小。</p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p12372123362813"><a name="p12372123362813"></a><a name="p12372123362813"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row13484523102713"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p183725334288"><a name="p183725334288"></a><a name="p183725334288"></a>singleCoreMN</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p4373113312819"><a name="p4373113312819"></a><a name="p4373113312819"></a>singleCoreM*singleCoreN的大小。</p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p18373233162813"><a name="p18373233162813"></a><a name="p18373233162813"></a>SpecialBasicBlock</p>
</td>
</tr>
<tr id="row74123713526"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p661414320154"><a name="p661414320154"></a><a name="p661414320154"></a>enUnitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p17614103216152"><a name="p17614103216152"></a><a name="p17614103216152"></a><span id="ph1065621641910"><a name="ph1065621641910"></a><a name="ph1065621641910"></a>使能UnitFlag功能，使计算与搬运流水并行，提高性能。Norm, IBShare下默认使能，MDL下默认不使能。参数取值如下：</span></p>
<a name="ul152765814315"></a><a name="ul152765814315"></a><ul id="ul152765814315"><li><span id="ph13657141617192"><a name="ph13657141617192"></a><a name="ph13657141617192"></a>false：不使能UnitFlag功能。</span></li><li><span id="ph166581616141912"><a name="ph166581616141912"></a><a name="ph166581616141912"></a>true：使能UnitFlag功能。</span></li></ul>
<p id="p6593921135211"><a name="p6593921135211"></a><a name="p6593921135211"></a><span id="ph10659101671916"><a name="ph10659101671916"></a><a name="ph10659101671916"></a>注意：MxMatmul场景，仅在NORM/MDL模板、A和scaleA不转置、 B和scaleB转置、C为ND格式，输出到GM场景下，使能UnitFlag功能有性能收益。</span></p>
<p id="p1694113115913"><a name="p1694113115913"></a><a name="p1694113115913"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1233310531184"><a name="p1233310531184"></a><a name="p1233310531184"></a>MDL、Norm、IBShare</p>
</td>
</tr>
<tr id="row512861285216"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p561413231516"><a name="p561413231516"></a><a name="p561413231516"></a>isPerTensor</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p4464151913361"><a name="p4464151913361"></a><a name="p4464151913361"></a><span id="ph346512199361"><a name="ph346512199361"></a><a name="ph346512199361"></a>A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否为per tensor。</span></p>
<a name="ul11402322163620"></a><a name="ul11402322163620"></a><ul id="ul11402322163620"><li><span id="ph022382416360"><a name="ph022382416360"></a><a name="ph022382416360"></a>true：per tensor量化。</span></li><li><span id="ph174121537163315"><a name="ph174121537163315"></a><a name="ph174121537163315"></a>false：per channel量化。</span></li></ul>
<p id="p171131416165420"><a name="p171131416165420"></a><a name="p171131416165420"></a><span id="ph24131137133319"><a name="ph24131137133319"></a><a name="ph24131137133319"></a>注意：MxMatmul场景仅支持false。</span></p>
<p id="p167941241195917"><a name="p167941241195917"></a><a name="p167941241195917"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1585251091914"><a name="p1585251091914"></a><a name="p1585251091914"></a>MDL、SpecialMDL</p>
</td>
</tr>
<tr id="row39041116175212"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p2061493291516"><a name="p2061493291516"></a><a name="p2061493291516"></a>hasAntiQuantOffset</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p116141032121517"><a name="p116141032121517"></a><a name="p116141032121517"></a><span id="ph166961850950"><a name="ph166961850950"></a><a name="ph166961850950"></a>A矩阵half类型输入且B矩阵int8_t类型输入场景，使能B矩阵量化时是否使用offset系数。</span></p>
<p id="p142014566541"><a name="p142014566541"></a><a name="p142014566541"></a><span id="ph1697950658"><a name="ph1697950658"></a><a name="ph1697950658"></a>注意：MxMatmul场景仅支持false。</span></p>
<p id="p34351052403"><a name="p34351052403"></a><a name="p34351052403"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1875895117180"><a name="p1875895117180"></a><a name="p1875895117180"></a>MDL、SpecialMDL</p>
</td>
</tr>
<tr id="row75301545102517"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1387745902510"><a name="p1387745902510"></a><a name="p1387745902510"></a>doIBShareNorm</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p287735942510"><a name="p287735942510"></a><a name="p287735942510"></a>使能IBShare模板。参数取值如下：</p>
<a name="ul17877125942510"></a><a name="ul17877125942510"></a><ul id="ul17877125942510"><li>false：不使能IBShare。</li><li>true：使能IBShare。</li></ul>
<p id="p148771596255"><a name="p148771596255"></a><a name="p148771596255"></a>IBShare的功能是能够复用L1上相同的A矩阵或B矩阵数据，开启后在数据复用场景能够避免重复搬运数据到L1。</p>
<p id="p19219186210"><a name="p19219186210"></a><a name="p19219186210"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p138771459122511"><a name="p138771459122511"></a><a name="p138771459122511"></a>IBShare</p>
</td>
</tr>
<tr id="row14493144816251"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p16100970268"><a name="p16100970268"></a><a name="p16100970268"></a>doSpecialMDL</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p610087162614"><a name="p610087162614"></a><a name="p610087162614"></a>使能SpecialMDL模板。参数取值如下：</p>
<a name="ul14100374264"></a><a name="ul14100374264"></a><ul id="ul14100374264"><li>true：使能SpecialMDL模板。</li><li>false：不使能SpecialMDL模板。</li></ul>
<p id="p310027112620"><a name="p310027112620"></a><a name="p310027112620"></a>MDL模板的一种特殊场景：Matmul K方向不全载时（singleCoreK/baseK &gt; stepKb），默认仅支持stepN设置为1，使能SpecailMDL模板后支持stepN=2。</p>
<p id="p510057152611"><a name="p510057152611"></a><a name="p510057152611"></a>注意：使能SpecialMDL模板时，<a href="#p36121832151520">doMultiDataLoad</a>参数取值必须为false。</p>
<p id="p15928105818016"><a name="p15928105818016"></a><a name="p15928105818016"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p710097102617"><a name="p710097102617"></a><a name="p710097102617"></a>SpecialMDL</p>
</td>
</tr>
<tr id="row1552593018294"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1259218334298"><a name="p1259218334298"></a><a name="p1259218334298"></a>enableInit</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p25922339292"><a name="p25922339292"></a><a name="p25922339292"></a>是否启用Init函数，不使能Init函数能够提升常量传播效果，优化性能。默认使能。</p>
<a name="ul18592333112914"></a><a name="ul18592333112914"></a><ul id="ul18592333112914"><li>false：不使能Init函数。</li><li>true：使能Init函数。</li></ul>
<p id="p10592203310294"><a name="p10592203310294"></a><a name="p10592203310294"></a>注意：MxMatmul场景仅支持true。</p>
<p id="p1976919494211"><a name="p1976919494211"></a><a name="p1976919494211"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p13592123319295"><a name="p13592123319295"></a><a name="p13592123319295"></a>所有模板</p>
</td>
</tr>
<tr id="row15745926182914"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p7593533192910"><a name="p7593533192910"></a><a name="p7593533192910"></a>batchMode</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p059314332299"><a name="p059314332299"></a><a name="p059314332299"></a><span id="ph115931633172912"><a name="ph115931633172912"></a><a name="ph115931633172912"></a>BatchMatmul场景中Layout类型为NORMAL时，设置BatchMatmul输入A/B矩阵的多batch数据总和与<span id="ph6593193372910"><a name="ph6593193372910"></a><a name="ph6593193372910"></a>L1 Buffer</span>的大小关系。参数取值如下：</span></p>
<a name="ul10593183316296"></a><a name="ul10593183316296"></a><ul id="ul10593183316296"><li><span id="ph5593733142910"><a name="ph5593733142910"></a><a name="ph5593733142910"></a>BatchMode::BATCH_LESS_THAN_L1：多batch数据总和&lt;<span id="ph145931033102914"><a name="ph145931033102914"></a><a name="ph145931033102914"></a>L1 Buffer</span> Size；</span></li><li><span id="ph25931334295"><a name="ph25931334295"></a><a name="ph25931334295"></a>BatchMode::BATCH_LARGE_THAN_L1：多batch数据总和&gt;<span id="ph10593333142912"><a name="ph10593333142912"></a><a name="ph10593333142912"></a>L1 Buffer</span> Size；</span></li><li><span id="ph9593203310298"><a name="ph9593203310298"></a><a name="ph9593203310298"></a>BatchMode::SINGLE_LARGE_THAN_L1：单batch数据总和&gt;<span id="ph259363362918"><a name="ph259363362918"></a><a name="ph259363362918"></a>L1 Buffer</span> Size。</span></li></ul>
<p id="p1019873318266"><a name="p1019873318266"></a><a name="p1019873318266"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p759323372916"><a name="p759323372916"></a><a name="p759323372916"></a>Norm</p>
</td>
</tr>
<tr id="row1320214209913"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1053115374920"><a name="p1053115374920"></a><a name="p1053115374920"></a>enableEnd</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p16203182010913"><a name="p16203182010913"></a><a name="p16203182010913"></a>Matmul计算过程中是否需要调用<a href="End.md">End</a>函数，该参数可用于优化性能。参数取值如下：</p>
<a name="ul743118519106"></a><a name="ul743118519106"></a><ul id="ul743118519106"><li>true：Matmul计算过程中需要调用End函数（默认值）。</li><li>false：不需要调用End函数。End处理相关的代码都会在编译期删除，从而优化性能。例如，<a href="GetTensorC.md#li17508136205415">异步场景</a>不需要调用End函数，可以将该参数置为false。</li></ul>
<p id="p7353123710294"><a name="p7353123710294"></a><a name="p7353123710294"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1203720090"><a name="p1203720090"></a><a name="p1203720090"></a>所有模板</p>
</td>
</tr>
<tr id="row13752356111"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1714310136111"><a name="p1714310136111"></a><a name="p1714310136111"></a>enableGetTensorC</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p127001521201113"><a name="p127001521201113"></a><a name="p127001521201113"></a>Matmul计算过程中是否需要调用<a href="GetTensorC.md">GetTensorC</a>函数，该参数可用于优化性能。参数取值如下：</p>
<a name="ul670072151115"></a><a name="ul670072151115"></a><ul id="ul670072151115"><li>true：Matmul计算过程中需要调用GetTensorC函数（默认值）。</li><li>false：不需要调用GetTensorC函数。GetTensorC处理相关的代码都会在编译期删除，从而优化性能。</li></ul>
<p id="p54021643202919"><a name="p54021643202919"></a><a name="p54021643202919"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p3551153701314"><a name="p3551153701314"></a><a name="p3551153701314"></a>所有模板</p>
</td>
</tr>
<tr id="row11479447122"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1115019619124"><a name="p1115019619124"></a><a name="p1115019619124"></a>enableSetOrgShape</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p82491151210"><a name="p82491151210"></a><a name="p82491151210"></a>Matmul计算过程中是否需要调用<a href="SetOrgShape.md">SetOrgShape</a>函数，该参数可用于优化性能。参数取值如下：</p>
<a name="ul1324011171211"></a><a name="ul1324011171211"></a><ul id="ul1324011171211"><li>true：Matmul计算过程中需要调用SetOrgShape函数（默认值）。</li><li>false：不需要调用SetOrgShape函数。SetOrgShape处理相关的代码都会在编译期删除，从而优化性能。</li></ul>
<p id="p159073494299"><a name="p159073494299"></a><a name="p159073494299"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p29754381139"><a name="p29754381139"></a><a name="p29754381139"></a>所有模板</p>
</td>
</tr>
<tr id="row14852111312415"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p15762821943"><a name="p15762821943"></a><a name="p15762821943"></a>enableSetBias</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p676214215419"><a name="p676214215419"></a><a name="p676214215419"></a><span id="ph167621021349"><a name="ph167621021349"></a><a name="ph167621021349"></a>是否使能计算Bias。该参数可用于优化性能。参数取值如下：</span></p>
<a name="ul127624211842"></a><a name="ul127624211842"></a><ul id="ul127624211842"><li><span id="ph77628211447"><a name="ph77628211447"></a><a name="ph77628211447"></a>true：使能计算Bias（默认值）。若输入带有Bias，实现过程中做Bias的搬运、计算等。</span></li><li><span id="ph156494381610"><a name="ph156494381610"></a><a name="ph156494381610"></a>false：不计算Bias。Bias处理相关的代码都会在编译期删除，从而优化性能。</span></li></ul>
<p id="p72541357112919"><a name="p72541357112919"></a><a name="p72541357112919"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p137625218413"><a name="p137625218413"></a><a name="p137625218413"></a>所有模板</p>
</td>
</tr>
<tr id="row1749017115134"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p324511911134"><a name="p324511911134"></a><a name="p324511911134"></a>enableSetTail</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p476141571317"><a name="p476141571317"></a><a name="p476141571317"></a>Matmul计算过程中是否需要调用<a href="SetTail.md">SetTail</a>函数，该参数可用于优化性能。参数取值如下：</p>
<a name="ul177671517137"></a><a name="ul177671517137"></a><ul id="ul177671517137"><li>true：Matmul计算过程中需要调用SetTail函数（默认值）。</li><li>false：不需要调用SetTail函数。SetTail处理相关的代码都会在编译期删除，从而优化性能。</li></ul>
<p id="p11772551281"><a name="p11772551281"></a><a name="p11772551281"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p17142840161320"><a name="p17142840161320"></a><a name="p17142840161320"></a>所有模板</p>
</td>
</tr>
<tr id="row15867122713146"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p16671203414142"><a name="p16671203414142"></a><a name="p16671203414142"></a>enableQuantVector</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p4741341121416"><a name="p4741341121416"></a><a name="p4741341121416"></a>Matmul计算过程中是否需要调用<a href="SetQuantVector.md">SetQuantVector</a>和<a href="SetQuantScalar.md">SetQuantScalar</a>函数，该参数可用于优化性能。参数取值如下：</p>
<a name="ul19741941101419"></a><a name="ul19741941101419"></a><ul id="ul19741941101419"><li>true：Matmul计算过程中需要调用SetQuantVector和SetQuantScalar函数（默认值）。</li><li>false：不需要调用SetQuantVector和SetQuantScalar函数。SetQuantVector和SetQuantScalar处理相关的代码都会在编译期删除，从而优化性能。</li></ul>
<p id="p1276102212914"><a name="p1276102212914"></a><a name="p1276102212914"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p686732713146"><a name="p686732713146"></a><a name="p686732713146"></a>所有模板</p>
</td>
</tr>
<tr id="row19984112611273"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p5706172082417"><a name="p5706172082417"></a><a name="p5706172082417"></a>enableSetDefineData</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p16706162062418"><a name="p16706162062418"></a><a name="p16706162062418"></a>使能模板参数MatmulCallBack（自定义回调函数）时，用于允许/禁止设置回调函数需要的计算数据或在GM上存储的数据地址等信息。</p>
<p id="p1759415315284"><a name="p1759415315284"></a><a name="p1759415315284"></a>参数取值如下：</p>
<a name="ul1645324517277"></a><a name="ul1645324517277"></a><ul id="ul1645324517277"><li>true：允许设置（默认值）。</li><li>false：不允许设置。SetSelfDefineData处理相关的代码都会在编译期删除，从而优化性能。</li></ul>
<p id="p48242511646"><a name="p48242511646"></a><a name="p48242511646"></a>注意：MxMatmul场景仅支持false。</p>
<p id="p6903131419309"><a name="p6903131419309"></a><a name="p6903131419309"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1535131717282"><a name="p1535131717282"></a><a name="p1535131717282"></a>Norm、MDL</p>
</td>
</tr>
<tr id="row139491313513"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p17777181323710"><a name="p17777181323710"></a><a name="p17777181323710"></a>iterateMode</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1318918126374"><a name="p1318918126374"></a><a name="p1318918126374"></a>用于优化Matmul计算的头开销。具体为，对Iterate系列接口（包括<a href="Iterate.md">Iterate</a>、<a href="IterateAll.md">IterateAll</a>、<a href="IterateBatch.md">IterateBatch</a>、<a href="IterateNBatch.md">IterateNBatch</a>）的优化，当使能某种模式时，表示Matmul计算过程中只调用该种模式对应的一个Iterate系列接口，其它Iterate系列接口相关的代码都会在编译期删除，从而优化性能。该参数为IterateMode类型。参数取值如下：</p>
<a name="ul9836195918247"></a><a name="ul9836195918247"></a><ul id="ul9836195918247"><li>ITERATE_MODE_NORMAL：对于Iterate系列接口，Matmul计算过程中只调用<a href="Iterate.md">Iterate</a>接口。</li><li>ITERATE_MODE_ALL：对于Iterate系列接口，Matmul计算过程中只调用<a href="IterateAll.md">IterateAll</a>接口。</li><li>ITERATE_MODE_BATCH：对于Iterate系列接口，Matmul计算过程中只调用<a href="IterateBatch.md">IterateBatch</a>接口。</li><li>ITERATE_MODE_N_BATCH：对于Iterate系列接口，Matmul计算过程中只调用<a href="IterateNBatch.md">IterateNBatch</a>接口。</li><li>ITERATE_MODE_DEFAULT：默认值，不限定调用Iterate系列接口的个数，不使能计算头开销的优化。</li></ul>
<p id="p08131655183316"><a name="p08131655183316"></a><a name="p08131655183316"></a><span id="ph1084115442015"><a name="ph1084115442015"></a><a name="ph1084115442015"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</p>
<p id="p41446267302"><a name="p41446267302"></a><a name="p41446267302"></a></p>
<p id="p20136428103014"><a name="p20136428103014"></a><a name="p20136428103014"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p85094248514"><a name="p85094248514"></a><a name="p85094248514"></a>所有模板</p>
</td>
</tr>
<tr id="row237325418268"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p20423826122714"><a name="p20423826122714"></a><a name="p20423826122714"></a>enableReuse</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p14238268272"><a name="p14238268272"></a><a name="p14238268272"></a><span id="ph14231826102716"><a name="ph14231826102716"></a><a name="ph14231826102716"></a><a href="SetSelfDefineData.md">SetSelfDefineData</a></span><span id="ph1642319261273"><a name="ph1642319261273"></a><a name="ph1642319261273"></a>函数设置的回调函数中的dataPtr是否直接传递计算数据。若未调用SetSelfDefineData设置dataPtr，该参数仅支持默认值true。参数取值如下：</span></p>
<a name="ul14423192611275"></a><a name="ul14423192611275"></a><ul id="ul14423192611275"><li><span id="ph642318266274"><a name="ph642318266274"></a><a name="ph642318266274"></a>true：直接传递计算数据，仅限单个值。</span></li><li><span id="ph2423152612720"><a name="ph2423152612720"></a><a name="ph2423152612720"></a>false：传递GM上存储的数据地址信息。</span></li></ul>
<p id="p12423726112713"><a name="p12423726112713"></a><a name="p12423726112713"></a><span id="ph104237262278"><a name="ph104237262278"></a><a name="ph104237262278"></a>注意：MxMatmul场景仅支持true。</span></p>
<p id="p1740857163816"><a name="p1740857163816"></a><a name="p1740857163816"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p204245266271"><a name="p204245266271"></a><a name="p204245266271"></a>Norm、MDL</p>
</td>
</tr>
<tr id="row8452155192620"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p3424132682715"><a name="p3424132682715"></a><a name="p3424132682715"></a>enableUBReuse</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p15424102618271"><a name="p15424102618271"></a><a name="p15424102618271"></a><span id="ph542412265278"><a name="ph542412265278"></a><a name="ph542412265278"></a>是否使能<span id="ph16424826132717"><a name="ph16424826132717"></a><a name="ph16424826132717"></a>Unified Buffer</span>复用。在<span id="ph1142432622711"><a name="ph1142432622711"></a><a name="ph1142432622711"></a><span id="ph042419261271"><a name="ph042419261271"></a><a name="ph042419261271"></a>Unified Buffer</span></span>空间足够的条件下（<span id="ph15424126162717"><a name="ph15424126162717"></a><a name="ph15424126162717"></a><span id="ph16424192614271"><a name="ph16424192614271"></a><a name="ph16424192614271"></a>Unified Buffer</span></span>空间大于4倍TCubeTiling的<a href="TCubeTiling结构体.md#p1620315053211">transLength</a>参数），使能该参数后，<span id="ph34246261279"><a name="ph34246261279"></a><a name="ph34246261279"></a><span id="ph154242264276"><a name="ph154242264276"></a><a name="ph154242264276"></a>Unified Buffer</span></span>空间分为互不重叠的两份，分别存储Matmul计算相邻前后两轮迭代的数据，后一轮迭代数据的搬入将不必等待前一轮迭代的<span id="ph542410265274"><a name="ph542410265274"></a><a name="ph542410265274"></a><span id="ph1424192616274"><a name="ph1424192616274"></a><a name="ph1424192616274"></a>Unified Buffer</span></span>空间释放，<span>从而优化流水</span>。参数取值如下：</span></p>
<a name="ul19424326102719"></a><a name="ul19424326102719"></a><ul id="ul19424326102719"><li><span id="ph342482692719"><a name="ph342482692719"></a><a name="ph342482692719"></a>true：使能<span id="ph11424182614272"><a name="ph11424182614272"></a><a name="ph11424182614272"></a>Unified Buffer</span>复用。</span></li><li><span id="ph19424182618274"><a name="ph19424182618274"></a><a name="ph19424182618274"></a>false：不使能<span id="ph17424526112712"><a name="ph17424526112712"></a><a name="ph17424526112712"></a>Unified Buffer</span>复用。</span></li></ul>
<p id="p12425426102711"><a name="p12425426102711"></a><a name="p12425426102711"></a><span id="ph1742511260278"><a name="ph1742511260278"></a><a name="ph1742511260278"></a><span id="ph8425526132713"><a name="ph8425526132713"></a><a name="ph8425526132713"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p12425102622717"><a name="p12425102622717"></a><a name="p12425102622717"></a>MDL</p>
</td>
</tr>
<tr id="row1664404892612"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1742510268276"><a name="p1742510268276"></a><a name="p1742510268276"></a>enableL1CacheUB</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p2042552613276"><a name="p2042552613276"></a><a name="p2042552613276"></a><span id="ph20425326192720"><a name="ph20425326192720"></a><a name="ph20425326192720"></a>是否使能<span id="ph134251126162713"><a name="ph134251126162713"></a><a name="ph134251126162713"></a>L1 Buffer</span>缓存<span id="ph1942512616274"><a name="ph1942512616274"></a><a name="ph1942512616274"></a>Unified Buffer</span>计算块。建议在MTE3和MTE2流水串行较多的场景使用。参数取值如下：</span></p>
<a name="ul942512622716"></a><a name="ul942512622716"></a><ul id="ul942512622716"><li><span id="ph1942516264270"><a name="ph1942516264270"></a><a name="ph1942516264270"></a>true：使能<span id="ph17425626102719"><a name="ph17425626102719"></a><a name="ph17425626102719"></a>L1 Buffer</span>缓存<span id="ph8425122642711"><a name="ph8425122642711"></a><a name="ph8425122642711"></a>Unified Buffer</span>计算块。</span></li><li><span id="ph1942562613273"><a name="ph1942562613273"></a><a name="ph1942562613273"></a>false：不使能<span id="ph8425122662719"><a name="ph8425122662719"></a><a name="ph8425122662719"></a>L1 Buffer</span>缓存<span id="ph7425826112719"><a name="ph7425826112719"></a><a name="ph7425826112719"></a>Unified Buffer</span>计算块。</span></li></ul>
<p id="p4425426142714"><a name="p4425426142714"></a><a name="p4425426142714"></a><span id="ph17426226122720"><a name="ph17426226122720"></a><a name="ph17426226122720"></a>若要使能<span id="ph74261826152710"><a name="ph74261826152710"></a><a name="ph74261826152710"></a>L1 Buffer</span>缓存<span id="ph20426192632719"><a name="ph20426192632719"></a><a name="ph20426192632719"></a>Unified Buffer</span>计算块，必须在Tiling实现中调用<a href="SetMatmulConfigParams.md">SetMatmulConfigParams</a>接口将参数enableL1CacheUBIn设置为true。</span></p>
<p id="p0426182662710"><a name="p0426182662710"></a><a name="p0426182662710"></a><span id="ph144261226202715"><a name="ph144261226202715"></a><a name="ph144261226202715"></a><span id="ph164261026142710"><a name="ph164261026142710"></a><a name="ph164261026142710"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</span></p>
<p id="p74151058194420"><a name="p74151058194420"></a><a name="p74151058194420"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p3426202672715"><a name="p3426202672715"></a><a name="p3426202672715"></a>MDL</p>
</td>
</tr>
<tr id="row83321943171614"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p193339432166"><a name="p193339432166"></a><a name="p193339432166"></a>intraBlockPartSum</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p18333643151614"><a name="p18333643151614"></a><a name="p18333643151614"></a>用于分离模式下的Vector、Cube计算融合场景，使能两个AIV核的一次计算结果（baseM * baseN大小的矩阵分片）在<span id="ph410020151596"><a name="ph410020151596"></a><a name="ph410020151596"></a>L0C Buffer</span>上累加，参数取值如下：</p>
<a name="ul6142174391815"></a><a name="ul6142174391815"></a><ul id="ul6142174391815"><li>false：不使能两个AIV核的计算结果在<span id="ph14723172442616"><a name="ph14723172442616"></a><a name="ph14723172442616"></a>L0C Buffer</span>上的累加（默认值）。</li><li>true：使能两个AIV核的计算结果在<span id="ph1952464822614"><a name="ph1952464822614"></a><a name="ph1952464822614"></a>L0C Buffer</span>上的累加。</li></ul>
<p id="p18590152142717"><a name="p18590152142717"></a><a name="p18590152142717"></a><span id="ph16733431193219"><a name="ph16733431193219"></a><a name="ph16733431193219"></a>Ascend 950PR/Ascend 950DT</span>不支持该参数。</p>
<p id="p146541836184518"><a name="p146541836184518"></a><a name="p146541836184518"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p6333174320169"><a name="p6333174320169"></a><a name="p6333174320169"></a>Norm</p>
</td>
</tr>
<tr id="row20766138152918"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p6449410202910"><a name="p6449410202910"></a><a name="p6449410202910"></a>IterateOrder</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p13449151032917"><a name="p13449151032917"></a><a name="p13449151032917"></a><span id="ph2044991082912"><a name="ph2044991082912"></a><a name="ph2044991082912"></a>Matmul做矩阵运算的循环迭代顺序，与<a href="TCubeTiling结构体.md#table1563162142915">表1</a>中的iterateOrder参数含义相同。当ScheduleType参数取值为ScheduleType::OUTER_PRODUCT时，本参数生效。参数取值如下：</span></p>
<p id="p18241038144118"><a name="p18241038144118"></a><a name="p18241038144118"></a><span id="ph1939916182436"><a name="ph1939916182436"></a><a name="ph1939916182436"></a>ORDER_M：先往M轴方向偏移再往N轴方向偏移。</span></p>
<p id="p08015513429"><a name="p08015513429"></a><a name="p08015513429"></a><span id="ph16134922144318"><a name="ph16134922144318"></a><a name="ph16134922144318"></a>ORDER_N：先往N轴方向偏移再往M轴方向偏移。</span></p>
<p id="p1259735213413"><a name="p1259735213413"></a><a name="p1259735213413"></a><span id="ph315742574314"><a name="ph315742574314"></a><a name="ph315742574314"></a>UNDEF：当前无效。</span></p>
<p id="p4449161018293"><a name="p4449161018293"></a><a name="p4449161018293"></a><span id="ph18449101011294"><a name="ph18449101011294"></a><a name="ph18449101011294"></a>注：Norm模板的Matmul场景、MDL模板使用时，若IterateOrder取值ORDER_M，<a href="TCubeTiling结构体.md#table1563162142915">TCubeTiling结构</a>中的stepN需要大于1，IterateOrder取值ORDER_N时，TCubeTiling结构中的stepM需要大于1。MxMatmul仅支持MDL模板。</span></p>
<p id="p5195134813135"><a name="p5195134813135"></a><a name="p5195134813135"></a><span id="ph13195848191310"><a name="ph13195848191310"></a><a name="ph13195848191310"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</p>
<p id="p9290134614513"><a name="p9290134614513"></a><a name="p9290134614513"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p11450141020291"><a name="p11450141020291"></a><a name="p11450141020291"></a>Norm、MDL</p>
</td>
</tr>
<tr id="row145763310294"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p114501910122916"><a name="p114501910122916"></a><a name="p114501910122916"></a>scheduleType</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p1645021011294"><a name="p1645021011294"></a><a name="p1645021011294"></a><span id="ph134509109299"><a name="ph134509109299"></a><a name="ph134509109299"></a>配置Matmul数据搬运模式。参数取值如下：</span></p>
<a name="ul1045011109292"></a><a name="ul1045011109292"></a><ul id="ul1045011109292"><li><span id="ph204501910172918"><a name="ph204501910172918"></a><a name="ph204501910172918"></a>ScheduleType::INNER_PRODUCT：默认模式，在K方向上做MTE1的循环搬运；</span></li><li><span id="ph17450910122915"><a name="ph17450910122915"></a><a name="ph17450910122915"></a>ScheduleType::OUTER_PRODUCT：在M或N方向上做MTE1的循环搬运；使能后，需要与IterateOrder参数配合使用。</span><div class="p" id="p9450141022919"><a name="p9450141022919"></a><a name="p9450141022919"></a><span id="ph54511910122918"><a name="ph54511910122918"></a><a name="ph54511910122918"></a>该配置当前只在BatchMatmul场景（使能Norm模板）或 Matmul场景（使能MDL模板或Norm模板）生效。</span><a name="ul845191019294"></a><a name="ul845191019294"></a><ul id="ul845191019294"><li><span id="ph124516109293"><a name="ph124516109293"></a><a name="ph124516109293"></a>若IterateOrder取值ORDER_M，则N方向循环搬运（在singleCoreN大于baseN场景可能有性能提升），即B矩阵的MTE1搬运并行；</span></li><li><span id="ph945110107295"><a name="ph945110107295"></a><a name="ph945110107295"></a>若IterateOrder取值ORDER_N，则M方向循环搬运（在singleCoreM大于baseM场景可能有性能提升），即A矩阵的MTE1搬运并行；</span></li><li><span id="ph1645171092912"><a name="ph1645171092912"></a><a name="ph1645171092912"></a>不能同时使能M方向和N方向循环搬运；</span></li></ul>
</div>
</li></ul>
<p id="p645119103299"><a name="p645119103299"></a><a name="p645119103299"></a><span id="ph12451210152916"><a name="ph12451210152916"></a><a name="ph12451210152916"></a>注：</span></p>
<a name="ul745121016295"></a><a name="ul745121016295"></a><ul id="ul745121016295"><li><span id="ph19451171010292"><a name="ph19451171010292"></a><a name="ph19451171010292"></a>Norm模板的Batch Matmul场景或者MDL模板中，singleCoreK&gt;baseK时，不能使能ScheduleType::OUTER_PRODUCT取值，需使用默认模式。</span></li><li><span id="ph8741177274"><a name="ph8741177274"></a><a name="ph8741177274"></a>Norm模板或MDL模板的Matmul场景，仅支持在纯Cube模式（只有矩阵计算）下配置ScheduleType::OUTER_PRODUCT。</span></li><li><span id="ph124515107298"><a name="ph124515107298"></a><a name="ph124515107298"></a>MDL模板仅在调用<a href="IterateAll.md">IterateAll</a>计算的场景支持配置ScheduleType::OUTER_PRODUCT。</span></li><li><span id="ph17451161018295"><a name="ph17451161018295"></a><a name="ph17451161018295"></a>仅在C矩阵输出至GM时，支持配置ScheduleType::OUTER_PRODUCT。</span></li><li>MxMatmul仅支持MDL模板。</li></ul>
<p id="p1745211010292"><a name="p1745211010292"></a><a name="p1745211010292"></a><span id="ph1445215105292"><a name="ph1445215105292"></a><a name="ph1445215105292"></a><span id="ph545215100296"><a name="ph545215100296"></a><a name="ph545215100296"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p271755317475"><a name="p271755317475"></a><a name="p271755317475"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p144521910122917"><a name="p144521910122917"></a><a name="p144521910122917"></a>Norm、MDL</p>
</td>
</tr>
<tr id="row1823941872917"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1399103817292"><a name="p1399103817292"></a><a name="p1399103817292"></a>enableDoubleCache</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p9991738142917"><a name="p9991738142917"></a><a name="p9991738142917"></a><span id="ph149911738122912"><a name="ph149911738122912"></a><a name="ph149911738122912"></a>开启IBShare模板后，在<span id="ph1899163892912"><a name="ph1899163892912"></a><a name="ph1899163892912"></a>L1 Buffer</span>上是否同时缓存两块数据。参数取值如下：</span></p>
<a name="ul59911338152919"></a><a name="ul59911338152919"></a><ul id="ul59911338152919"><li><span id="ph1899233832917"><a name="ph1899233832917"></a><a name="ph1899233832917"></a>false：<span id="ph14992203818292"><a name="ph14992203818292"></a><a name="ph14992203818292"></a>L1 Buffer</span>上同时缓存一块数据（默认值）。</span></li><li><span id="ph499214386296"><a name="ph499214386296"></a><a name="ph499214386296"></a>true：使能<span id="ph159923388291"><a name="ph159923388291"></a><a name="ph159923388291"></a>L1 Buffer</span>上同时缓存两块数据。</span></li></ul>
<p id="p11992123819293"><a name="p11992123819293"></a><a name="p11992123819293"></a><span id="ph1699210381295"><a name="ph1699210381295"></a><a name="ph1699210381295"></a>注意：该参数取值为true时，需要控制基本块大小，防止两块数据的缓存超过<span id="ph5992173842912"><a name="ph5992173842912"></a><a name="ph5992173842912"></a>L1 Buffer</span>大小限制。</span></p>
<p id="p99924385298"><a name="p99924385298"></a><a name="p99924385298"></a><span id="ph199921738182916"><a name="ph199921738182916"></a><a name="ph199921738182916"></a><span id="ph999243810293"><a name="ph999243810293"></a><a name="ph999243810293"></a>Ascend 950PR/Ascend 950DT</span>仅支持取值为false。</span></p>
<p id="p3787163364813"><a name="p3787163364813"></a><a name="p3787163364813"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p14992938132911"><a name="p14992938132911"></a><a name="p14992938132911"></a>IBShare</p>
</td>
</tr>
<tr id="row881220125309"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1223441413018"><a name="p1223441413018"></a><a name="p1223441413018"></a>isBiasBatch</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p2235714193020"><a name="p2235714193020"></a><a name="p2235714193020"></a><span id="ph11235181411306"><a name="ph11235181411306"></a><a name="ph11235181411306"></a>批量多Batch的Matmul场景，即BatchMatmul场景，Bias的大小是否带有Batch轴。参数取值如下：</span></p>
<a name="ul152350145302"></a><a name="ul152350145302"></a><ul id="ul152350145302"><li><span id="ph8235111412304"><a name="ph8235111412304"></a><a name="ph8235111412304"></a>true：Bias带有Batch轴，Bias大小为Batch * N（默认值）。</span></li><li><span id="ph823518142300"><a name="ph823518142300"></a><a name="ph823518142300"></a>false：Bias不带Batch轴，Bias大小为N，多Batch计算Matmul时，会复用Bias。</span><p id="p0605194632616"><a name="p0605194632616"></a><a name="p0605194632616"></a><span id="ph16605746142610"><a name="ph16605746142610"></a><a name="ph16605746142610"></a>注意：BatchMode::SINGLE_LARGE_THAN_L1场景仅支持设置为true。</span></p>
</li></ul>
<p id="p6470248153220"><a name="p6470248153220"></a><a name="p6470248153220"></a><span id="ph960534612265"><a name="ph960534612265"></a><a name="ph960534612265"></a><span id="ph160544610261"><a name="ph160544610261"></a><a name="ph160544610261"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p696514451327"><a name="p696514451327"></a><a name="p696514451327"></a><span id="ph284875253411"><a name="ph284875253411"></a><a name="ph284875253411"></a></span></p>
<p id="p18144543123214"><a name="p18144543123214"></a><a name="p18144543123214"></a><span id="ph484818522347"><a name="ph484818522347"></a><a name="ph484818522347"></a></span></p>
<p id="p27571530183220"><a name="p27571530183220"></a><a name="p27571530183220"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1923612147308"><a name="p1923612147308"></a><a name="p1923612147308"></a>Norm</p>
</td>
</tr>
<tr id="row8965173713300"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p64593399303"><a name="p64593399303"></a><a name="p64593399303"></a>enableStaticPadZeros</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p845914395308"><a name="p845914395308"></a><a name="p845914395308"></a><span id="ph14608395304"><a name="ph14608395304"></a><a name="ph14608395304"></a>使用常量化的Tiling参数时，在左矩阵和右矩阵搬运到<span id="ph13460183912307"><a name="ph13460183912307"></a><a name="ph13460183912307"></a>L1 Buffer</span>的过程中，是否自动按照常量化的singleM/singleN/singleK及baseM/baseN/baseK大小补零。关于常量化Tiling参数的详细内容请参考<a href="GetMatmulApiTiling.md">GetMatmulApiTiling</a>。</span></p>
<p id="p1546017391303"><a name="p1546017391303"></a><a name="p1546017391303"></a><span id="ph2460163913304"><a name="ph2460163913304"></a><a name="ph2460163913304"></a>仅支持GM输入的ND2NZ格式的补零，其他场景需要用户自行补零。参数取值如下：</span></p>
<a name="ul12460183919304"></a><a name="ul12460183919304"></a><ul id="ul12460183919304"><li><span id="ph3460173993020"><a name="ph3460173993020"></a><a name="ph3460173993020"></a>false：搬运时不自动补零，需要用户自行补零（默认值）。</span></li><li><span id="ph105477613528"><a name="ph105477613528"></a><a name="ph105477613528"></a>true：搬运时按照常量化的singleM/singleN/singleK及baseM/baseN/baseK大小自动补零。</span></li></ul>
<p id="p721371133319"><a name="p721371133319"></a><a name="p721371133319"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p94602392302"><a name="p94602392302"></a><a name="p94602392302"></a>Norm、MDL、IBShare</p>
</td>
</tr>
<tr id="row42634380348"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p2030383903419"><a name="p2030383903419"></a><a name="p2030383903419"></a>isPartialOutput</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p63031939193420"><a name="p63031939193420"></a><a name="p63031939193420"></a><span id="ph1057174216711"><a name="ph1057174216711"></a><a name="ph1057174216711"></a>是否开启PartialOutput功能，即控制Matmul顺序输出K方向的基本块计算方式：Matmul一次Iterate计算的K轴是否进行累加计算。参数取值如下：</span></p>
<a name="ul43035394341"></a><a name="ul43035394341"></a><ul id="ul43035394341"><li><span id="ph969194219713"><a name="ph969194219713"></a><a name="ph969194219713"></a>true：开启PartialOutput功能，一次Iterate的K轴不进行累加计算，Matmul每次计算输出局部baseK的baseM * baseN大小的矩阵分片。</span></li><li><span id="ph13787421177"><a name="ph13787421177"></a><a name="ph13787421177"></a>false：不开启PartialOutput功能，一次Iterate的K轴进行累加计算，Matmul每次计算输出SingleCoreK长度的baseM * baseN大小的矩阵分片。</span></li></ul>
<p id="p16303123993415"><a name="p16303123993415"></a><a name="p16303123993415"></a><span id="ph13793426710"><a name="ph13793426710"></a><a name="ph13793426710"></a><span id="ph203032039133413"><a name="ph203032039133413"></a><a name="ph203032039133413"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p1055713485314"><a name="p1055713485314"></a><a name="p1055713485314"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p193045396342"><a name="p193045396342"></a><a name="p193045396342"></a>MDL</p>
</td>
</tr>
<tr id="row1788619817374"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p9218181073719"><a name="p9218181073719"></a><a name="p9218181073719"></a>enableMixDualMaster</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p22184108371"><a name="p22184108371"></a><a name="p22184108371"></a><span id="ph10218101083714"><a name="ph10218101083714"></a><a name="ph10218101083714"></a>是否使能MixDualMaster（双主模式）。区别于MIX模式（包含矩阵计算和矢量计算）通过消息机制驱动AIC运行，双主模式为AIC和AIV独立运行代码，不依赖消息驱动，用于提升性能。该参数默认值为false，仅能在以下场景设置为true：</span></p>
<a name="ul1221811033715"></a><a name="ul1221811033715"></a><ul id="ul1221811033715"><li><span id="ph10218310193712"><a name="ph10218310193712"></a><a name="ph10218310193712"></a>核函数的类型为MIX，同时AIC核数 : AIV核数为1:1。</span></li><li><span id="ph1921814102377"><a name="ph1921814102377"></a><a name="ph1921814102377"></a>核函数的类型为MIX，同时AIC核数 : AIV核数为1:2，且A矩阵和B矩阵同时使能<a href="Matmul使用说明.md#table1188045714378">IBSHARE</a>参数。</span></li></ul>
<p id="p321971003719"><a name="p321971003719"></a><a name="p321971003719"></a><span id="ph921911101378"><a name="ph921911101378"></a><a name="ph921911101378"></a>注意，使能MixDualMaster场景，需要满足：</span></p>
<a name="ul621941023719"></a><a name="ul621941023719"></a><ul id="ul621941023719"><li><span id="ph921918107377"><a name="ph921918107377"></a><a name="ph921918107377"></a>同一算子中所有Matmul对象的该参数取值必须保持一致。</span></li><li><span id="ph192191210153718"><a name="ph192191210153718"></a><a name="ph192191210153718"></a>A/B/Bias矩阵只支持从GM搬入。</span></li><li><span id="ph1621991013372"><a name="ph1621991013372"></a><a name="ph1621991013372"></a>获取矩阵计算结果只支持调用<a href="IterateAll.md">IterateAll</a>接口输出到GlobalTensor<span id="ph20219141053713"><a name="ph20219141053713"></a><a name="ph20219141053713"></a>或者LocalTensor</span>，即计算结果放置于Global Memory<span id="ph3219131053714"><a name="ph3219131053714"></a><a name="ph3219131053714"></a><span id="ph42191810133716"><a name="ph42191810133716"></a><a name="ph42191810133716"></a>或者Local Memory </span></span>的地址，不能调用<a href="GetTensorC.md">GetTensorC</a>等接口获取结果。</span></li></ul>
<a name="ul63521417153210"></a><a name="ul63521417153210"></a>
<p id="p1622091073715"><a name="p1622091073715"></a><a name="p1622091073715"></a><span id="ph14220151023713"><a name="ph14220151023713"></a><a name="ph14220151023713"></a><span id="ph1422071093713"><a name="ph1422071093713"></a><a name="ph1422071093713"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p26341657155313"><a name="p26341657155313"></a><a name="p26341657155313"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p14220191013374"><a name="p14220191013374"></a><a name="p14220191013374"></a>Norm</p>
</td>
</tr>
<tr id="row1596912952711"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p3706172016247"><a name="p3706172016247"></a><a name="p3706172016247"></a>isA2B2Shared</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p670620208249"><a name="p670620208249"></a><a name="p670620208249"></a><span id="ph8904417145515"><a name="ph8904417145515"></a><a name="ph8904417145515"></a>是否开启A2和B2的全局管理，即控制所有Matmul对象是否共用A2和B2的double buffer机制。该配置为全局配置，所有Matmul对象取值必须保持一致。注意，开启时，A矩阵、B矩阵的基本块大小均不能超过32KB。</span></p>
<p id="p16801131202911"><a name="p16801131202911"></a><a name="p16801131202911"></a><span id="ph590791718559"><a name="ph590791718559"></a><a name="ph590791718559"></a>参数取值如下：</span></p>
<a name="ul815675114271"></a><a name="ul815675114271"></a><ul id="ul815675114271"><li><span id="ph10908111714556"><a name="ph10908111714556"></a><a name="ph10908111714556"></a>true：开启。</span></li><li><span id="ph690916179551"><a name="ph690916179551"></a><a name="ph690916179551"></a>false：关闭（默认值）。</span></li></ul>
<p id="p1524135812332"><a name="p1524135812332"></a><a name="p1524135812332"></a><span id="ph2583105444517"><a name="ph2583105444517"></a><a name="ph2583105444517"></a><span id="ph1024175814333"><a name="ph1024175814333"></a><a name="ph1024175814333"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p1163513205412"><a name="p1163513205412"></a><a name="p1163513205412"></a></p>
<p id="p4658451159"><a name="p4658451159"></a><a name="p4658451159"></a><span id="ph8910141713555"><a name="ph8910141713555"></a><a name="ph8910141713555"></a>注意：MxMatmul场景下该参数仅支持false。</span></p>
<p id="p10175105816509"><a name="p10175105816509"></a><a name="p10175105816509"></a><span id="ph14759102624513"><a name="ph14759102624513"></a><a name="ph14759102624513"></a>该参数取值为true时，建议同时设置enUnitFlag参数为true，使搬运与计算流水并行，提高性能。</span></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p1558171962810"><a name="p1558171962810"></a><a name="p1558171962810"></a>Norm、MDL</p>
</td>
</tr>
<tr id="row533672923717"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p5341543113715"><a name="p5341543113715"></a><a name="p5341543113715"></a>isEnableChannelSplit</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p15341143123720"><a name="p15341143123720"></a><a name="p15341143123720"></a><span id="ph19341543153717"><a name="ph19341543153717"></a><a name="ph19341543153717"></a>是否使能channel_split功能。正常情况下，Matmul计算出的CubeFormat::NZ格式的C矩阵分形为16*16，假设此时的分形个数为x，channel_split功能是使获得的C矩阵分形为16*8，同时分形个数变为2x。注意，当前仅在Matmul计算结果C矩阵的Format为CubeFormat::NZ，TYPE为float类型，矩阵乘结果CO1为float类型，输出到Global Memory的场景，支持使能该参数。参数取值如下：</span></p>
<a name="ul83411243103720"></a><a name="ul83411243103720"></a><ul id="ul83411243103720"><li><span id="ph7341443183720"><a name="ph7341443183720"></a><a name="ph7341443183720"></a>false：默认值，不使能channel_split功能，输出的分形为16*16。</span></li><li><span id="ph4341114310372"><a name="ph4341114310372"></a><a name="ph4341114310372"></a>true：使能channel_split功能，输出的分形为16*8。</span></li></ul>
<p id="p7342144383715"><a name="p7342144383715"></a><a name="p7342144383715"></a><span id="ph43421643153712"><a name="ph43421643153712"></a><a name="ph43421643153712"></a><span id="ph43421143123716"><a name="ph43421143123716"></a><a name="ph43421143123716"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p313165620541"><a name="p313165620541"></a><a name="p313165620541"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p11342154319378"><a name="p11342154319378"></a><a name="p11342154319378"></a>所有模板</p>
</td>
</tr>
<tr id="row14275332173711"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p5342843193712"><a name="p5342843193712"></a><a name="p5342843193712"></a>enableKdimReorderLoad</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p183421443203712"><a name="p183421443203712"></a><a name="p183421443203712"></a><span id="ph7342124313712"><a name="ph7342124313712"></a><a name="ph7342124313712"></a>是否使能K轴错峰加载数据。基于相同Tiling参数，执行Matmul计算时，如果多核的左矩阵或者右矩阵相同，且存储于Global Memory，多个核一般会同时访问相同地址以加载矩阵数据，引发同地址访问冲突，影响性能。使能该参数后，多核执行Matmul时，将尽量在相同时间访问矩阵的不同Global Memory地址，减少地址访问冲突概率，提升性能。该参数功能只支持MDL模板，建议K轴较大且左矩阵和右矩阵均非全载场景使能参数。参数取值如下。</span></p>
<a name="ul133428436372"></a><a name="ul133428436372"></a><ul id="ul133428436372"><li><span id="ph19343643183712"><a name="ph19343643183712"></a><a name="ph19343643183712"></a>false：默认值，关闭K轴错峰加载数据的功能。</span></li><li><span id="ph5343174318377"><a name="ph5343174318377"></a><a name="ph5343174318377"></a>true：开启K轴错峰加载数据的功能。</span></li></ul>
<p id="p1234344315377"><a name="p1234344315377"></a><a name="p1234344315377"></a><span id="ph634394333710"><a name="ph634394333710"></a><a name="ph634394333710"></a><span id="ph5343204318379"><a name="ph5343204318379"></a><a name="ph5343204318379"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p86759126565"><a name="p86759126565"></a><a name="p86759126565"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p63441443133711"><a name="p63441443133711"></a><a name="p63441443133711"></a>MDL</p>
</td>
</tr>
<tr id="row11846332582"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p4232938145515"><a name="p4232938145515"></a><a name="p4232938145515"></a>isCO1Shared</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p8914436155210"><a name="p8914436155210"></a><a name="p8914436155210"></a>是否使能CO1内存共享，由该参数与sharedCO1BufferSize参数指定CO1划分块数，缓存到CO1中的数据块数不能超过CO1划分的块数，即未被GetTensorC获取的Iterate计算生成的结果个数不能超过CO1划分的块数。该配置为全局配置，所有Matmul对象取值必须保持一致。参数取值如下：</p>
<a name="ul141834458491"></a><a name="ul141834458491"></a><ul id="ul141834458491"><li>true：开启CO1内存共享。</li><li>false：默认值，关闭CO1内存共享。</li></ul>
<p id="p1027621295611"><a name="p1027621295611"></a><a name="p1027621295611"></a><span id="ph202761412155617"><a name="ph202761412155617"></a><a name="ph202761412155617"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</p>
<p id="p2229516185615"><a name="p2229516185615"></a><a name="p2229516185615"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p27972445810"><a name="p27972445810"></a><a name="p27972445810"></a>Norm、IBShare</p>
</td>
</tr>
<tr id="row1192910342815"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p123283817554"><a name="p123283817554"></a><a name="p123283817554"></a>sharedCO1BufferSize</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p4438133021914"><a name="p4438133021914"></a><a name="p4438133021914"></a>指定CO1共享的一份Buffer大小。uint32_t类型，支持的取值为32*1024、64*1024、128*1024。</p>
<p id="p17381931105618"><a name="p17381931105618"></a><a name="p17381931105618"></a><span id="ph14738831195616"><a name="ph14738831195616"></a><a name="ph14738831195616"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</p>
<p id="p20129119175612"><a name="p20129119175612"></a><a name="p20129119175612"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p2072231113918"><a name="p2072231113918"></a><a name="p2072231113918"></a>Norm、IBShare</p>
</td>
</tr>
<tr id="row20366104713238"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p10367184732311"><a name="p10367184732311"></a><a name="p10367184732311"></a>bmmOutMode</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p123671847122313"><a name="p123671847122313"></a><a name="p123671847122313"></a>预留参数。</p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p136714782312"><a name="p136714782312"></a><a name="p136714782312"></a>预留参数</p>
</td>
</tr>
<tr id="row1664381971219"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p84588523128"><a name="p84588523128"></a><a name="p84588523128"></a>enableL1BankConflictOptimise</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p8458135281220"><a name="p8458135281220"></a><a name="p8458135281220"></a><span id="ph20827245182013"><a name="ph20827245182013"></a><a name="ph20827245182013"></a>是否使能L1上的Bank冲突优化。在Tiling侧调用<a href="EnableL1BankConflictOptimise.md">EnableL1BankConflictOptimise</a>接口获取能否使能该参数的结果，并与<a href="基本流程.md#li578045965">TilingKey</a>机制配合使用，在Kernel侧增加代码实现分支。若使能该参数，基于相同Tiling参数执行Matmul计算时，对A、B矩阵和MxMatmul场景的ScaleA、ScaleB矩阵不再连续分配L1 Buffer的空间，在DoubleBuffer场景下，并行计算的数据分别被分配在L1 Buffer的上半部空间和下半部空间，非DoubleBuf场景，数据被分配在L1 Buffer的上半部空间；另外，Bias被分配在L1 Buffer的上半部空间，向量的量化/反量化场景的量化系数被分配在L1 Buffer的下半部空间。参数取值如下。</span></p>
<a name="ul445935211218"></a><a name="ul445935211218"></a><ul id="ul445935211218"><li><span id="ph95151518265"><a name="ph95151518265"></a><a name="ph95151518265"></a>false：默认值，关闭L1 Bank冲突优化。</span></li><li><span id="ph11787103219614"><a name="ph11787103219614"></a><a name="ph11787103219614"></a>true：开启L1 Bank冲突优化。</span></li></ul>
<p id="p145915217126"><a name="p145915217126"></a><a name="p145915217126"></a><span id="ph94591852121214"><a name="ph94591852121214"></a><a name="ph94591852121214"></a><span id="ph24590524125"><a name="ph24590524125"></a><a name="ph24590524125"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</span></p>
<p id="p8840132719561"><a name="p8840132719561"></a><a name="p8840132719561"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p3459185201214"><a name="p3459185201214"></a><a name="p3459185201214"></a>MDL</p>
</td>
</tr>
<tr id="row1769811525720"><td class="cellrowborder" valign="top" width="21.330000000000002%" headers="mcps1.2.4.1.1 "><p id="p1769814521177"><a name="p1769814521177"></a><a name="p1769814521177"></a>enableRelu</p>
</td>
<td class="cellrowborder" valign="top" width="55.190000000000005%" headers="mcps1.2.4.1.2 "><p id="p176987521079"><a name="p176987521079"></a><a name="p176987521079"></a>是否使能对矩阵乘的输出矩阵C做Relu修正。开启该功能后，输出矩阵中负数值被修正为0。参数取值如下。</p>
<a name="ul20786933173717"></a><a name="ul20786933173717"></a><ul id="ul20786933173717"><li>false：默认值，关闭对输出矩阵C的Relu修正功能。</li><li>true：开启对输出矩阵C的Relu修正功能。</li></ul>
<p id="p123411620113817"><a name="p123411620113817"></a><a name="p123411620113817"></a><span id="ph9572133010214"><a name="ph9572133010214"></a><a name="ph9572133010214"></a>Ascend 950PR/Ascend 950DT</span>支持该参数。</p>
<p id="p581553015564"><a name="p581553015564"></a><a name="p581553015564"></a></p>
</td>
<td class="cellrowborder" valign="top" width="23.480000000000004%" headers="mcps1.2.4.1.3 "><p id="p18698205218714"><a name="p18698205218714"></a><a name="p18698205218714"></a>所有模板</p>
</td>
</tr>
</tbody>
</table>

