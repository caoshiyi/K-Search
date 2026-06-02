# MaskReg搬入<a name="ZH-CN_TOPIC_0000002523303982"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1622020982912"><a name="p1622020982912"></a><a name="p1622020982912"></a><span id="ph1522010992915"><a name="ph1522010992915"></a><a name="ph1522010992915"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

Reg矢量计算数据搬运接口，适用于从UB或RegTensor搬入MaskReg。

## 函数原型<a name="section620mcpsimp"></a>

```
// MaskReg搬入使用 AddrReg 存储偏移量
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void LoadAlign(MaskReg& mask, __ubuf__ T* srcAddr, AddrReg offset);
// MaskReg搬入POST_MODE_NORMAL 场景
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void LoadAlign(MaskReg& mask, __ubuf__ T* srcAddr);
// MaskReg搬入POST_MODE_UPDATE 场景
template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void LoadAlign(MaskReg& mask, __ubuf__ T* &srcAddr, int32_t offset);
// MaskReg从RegTensor搬入
template <typename T = DefaultType, int16_t offset, typename U>
__simd_callee__ inline void MaskGenWithRegTensor(MaskReg& dst, U& srcReg);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  MaskReg搬入使用AddrReg存储偏移量参数说明

<a name="table14132101714462"></a>
<table><thead align="left"><tr id="row19176617124620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p117621724612"><a name="p117621724612"></a><a name="p117621724612"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p417681717463"><a name="p417681717463"></a><a name="p417681717463"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p5176017164614"><a name="p5176017164614"></a><a name="p5176017164614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1217616175468"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p220141795213"><a name="p220141795213"></a><a name="p220141795213"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1421718217522"><a name="p1421718217522"></a><a name="p1421718217522"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p4201131716521"><a name="p4201131716521"></a><a name="p4201131716521"></a>操作数数据类型。支持的数据类型为b8/b16/b32。</p>
</td>
</tr>
<tr id="row142116557342"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p10201517105219"><a name="p10201517105219"></a><a name="p10201517105219"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p5217152117529"><a name="p5217152117529"></a><a name="p5217152117529"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p62011179525"><a name="p62011179525"></a><a name="p62011179525"></a>搬运模式， MaskDist类型。取值如下：</p>
<a name="ul122012176529"></a><a name="ul122012176529"></a><ul id="ul122012176529"><li>DIST_NORM，对齐约束为VL/8Byte，正常模式，搬运VL/8Byte数据。</li><li>DIST_US，对齐约束为VL/16Byte，上采样模式，搬运VL/16Byte数据，每bit重复一次。</li><li>DIST_DS，对齐约束为min(32, VL/4)Byte，下采样模式，搬运VL/4Byte数据，每间隔1bit被舍弃。</li></ul>
</td>
</tr>
<tr id="row1417671716463"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p12176201754617"><a name="p12176201754617"></a><a name="p12176201754617"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1317616173468"><a name="p1317616173468"></a><a name="p1317616173468"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p2176141714466"><a name="p2176141714466"></a><a name="p2176141714466"></a>目的操作数，类型为MaskReg。</p>
</td>
</tr>
<tr id="row958051624416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p9580161634416"><a name="p9580161634416"></a><a name="p9580161634416"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p2581416144417"><a name="p2581416144417"></a><a name="p2581416144417"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p195811716134418"><a name="p195811716134418"></a><a name="p195811716134418"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row126641434416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1671114174418"><a name="p1671114174418"></a><a name="p1671114174418"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p567151494415"><a name="p567151494415"></a><a name="p567151494415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p1185418151015"><a name="p1185418151015"></a><a name="p1185418151015"></a>实际搬运UB起始地址为 srcAddr + offset。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  MaskReg搬入POST\_MODE\_NORMAL场景参数说明

<a name="table628514470512"></a>
<table><thead align="left"><tr id="row11285847135113"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p1228517474515"><a name="p1228517474515"></a><a name="p1228517474515"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p728544715512"><a name="p728544715512"></a><a name="p728544715512"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p1828504712511"><a name="p1828504712511"></a><a name="p1828504712511"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row179511801312"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1195434010116"><a name="p1195434010116"></a><a name="p1195434010116"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1095412403115"><a name="p1095412403115"></a><a name="p1095412403115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p9954194017111"><a name="p9954194017111"></a><a name="p9954194017111"></a>操作数数据类型。支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row14285547195114"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p152851447135115"><a name="p152851447135115"></a><a name="p152851447135115"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p12285204719512"><a name="p12285204719512"></a><a name="p12285204719512"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p119983578116"><a name="p119983578116"></a><a name="p119983578116"></a>搬运模式， MaskDist类型。取值如下：</p>
<a name="ul8998357018"></a><a name="ul8998357018"></a><ul id="ul8998357018"><li>DIST_NORM，对齐约束为VL/8Byte，正常模式，搬运VL/8Byte数据。</li><li>DIST_US，对齐约束为VL/16Byte，上采样模式，搬运VL/16Byte数据，每bit重复一次。</li><li>DIST_DS，对齐约束为min(32, VL/4)Byte，下采样模式，搬运VL/4Byte数据，每间隔1bit被舍弃。</li></ul>
</td>
</tr>
<tr id="row128694775111"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p82861747165119"><a name="p82861747165119"></a><a name="p82861747165119"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p42861847185110"><a name="p42861847185110"></a><a name="p42861847185110"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p1628613479514"><a name="p1628613479514"></a><a name="p1628613479514"></a>目的操作数，类型为MaskTensor。</p>
</td>
</tr>
<tr id="row5286204711515"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p11286947185119"><a name="p11286947185119"></a><a name="p11286947185119"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1428654785111"><a name="p1428654785111"></a><a name="p1428654785111"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p028634712517"><a name="p028634712517"></a><a name="p028634712517"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  MaskReg搬入POST\_MODE\_UPDATE场景参数说明

<a name="table27711449115114"></a>
<table><thead align="left"><tr id="row277115491518"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p10771164985116"><a name="p10771164985116"></a><a name="p10771164985116"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p157711349165119"><a name="p157711349165119"></a><a name="p157711349165119"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p18771144955117"><a name="p18771144955117"></a><a name="p18771144955117"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row594961117316"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1747652234"><a name="p1747652234"></a><a name="p1747652234"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p67476522313"><a name="p67476522313"></a><a name="p67476522313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p074785218311"><a name="p074785218311"></a><a name="p074785218311"></a>操作数数据类型。支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row2771194935120"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p674710521332"><a name="p674710521332"></a><a name="p674710521332"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p19748185219316"><a name="p19748185219316"></a><a name="p19748185219316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p16748652437"><a name="p16748652437"></a><a name="p16748652437"></a>搬运模式， MaskDist类型。取值如下：</p>
<a name="ul474817521538"></a><a name="ul474817521538"></a><ul id="ul474817521538"><li>DIST_NORM，对齐约束为VL/8Byte，正常模式，搬运VL/8Byte数据。</li><li>DIST_US，对齐约束为VL/16Byte，上采样模式，搬运VL/16Byte数据，每bit重复一次。</li><li>DIST_DS，对齐约束为min(32, VL/4)Byte，下采样模式，搬运VL/4Byte数据，每间隔1bit被舍弃。</li></ul>
</td>
</tr>
<tr id="row813984120556"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p483720285383"><a name="p483720285383"></a><a name="p483720285383"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p883792893817"><a name="p883792893817"></a><a name="p883792893817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p1246583315216"><a name="p1246583315216"></a><a name="p1246583315216"></a>用于控制是否使能post update。</p>
<a name="ul121721550125114"></a><a name="ul121721550125114"></a><ul id="ul121721550125114"><li>POST_MODE_NORMAL，正常场景，UB操作数地址不更新。</li></ul>
<a name="ul51081954165116"></a><a name="ul51081954165116"></a><ul id="ul51081954165116"><li>POST_MODE_UPDATE，POST_MODE_UPDATE场景使用，UB地址同时作为输入和输出，每次调用会更新。</li></ul>
</td>
</tr>
<tr id="row177164912514"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1274813526317"><a name="p1274813526317"></a><a name="p1274813526317"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p67481152130"><a name="p67481152130"></a><a name="p67481152130"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p127481952930"><a name="p127481952930"></a><a name="p127481952930"></a>目的操作数，类型为MaskTensor。</p>
</td>
</tr>
<tr id="row77711349115117"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p874835220317"><a name="p874835220317"></a><a name="p874835220317"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p207481652238"><a name="p207481652238"></a><a name="p207481652238"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p11748125219315"><a name="p11748125219315"></a><a name="p11748125219315"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row1077274912512"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p15772849175112"><a name="p15772849175112"></a><a name="p15772849175112"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p167721849125110"><a name="p167721849125110"></a><a name="p167721849125110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><div class="p" id="p1461773515108"><a name="p1461773515108"></a><a name="p1461773515108"></a>当offset为int32_t类型时，POST_MODE_NORMAL与POST_MODE_UPDATE含义不一致。<a name="ul16772949185118"></a><a name="ul16772949185118"></a><ul id="ul16772949185118"><li>POST_MODE_NORMAL 场景：实际搬运UB起始地址为srcAddr + offset。</li><li>POST_MODE_UPDATE 场景：实际搬运UB起始地址为srcAddr，搬运后执行地址更新 srcAddr +=  offset。</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

**表 4**  MaskReg从RegTensor搬入参数说明

<a name="table67195217515"></a>
<table><thead align="left"><tr id="row177135218510"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p177252135115"><a name="p177252135115"></a><a name="p177252135115"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p187145212514"><a name="p187145212514"></a><a name="p187145212514"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p6725212512"><a name="p6725212512"></a><a name="p6725212512"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row187352125115"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p137865211964"><a name="p137865211964"></a><a name="p137865211964"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p2093610545613"><a name="p2093610545613"></a><a name="p2093610545613"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p1378612211263"><a name="p1378612211263"></a><a name="p1378612211263"></a>操作数数据类型。支持的数据类型为b16/b32。</p>
</td>
</tr>
<tr id="row48145245117"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p178619211462"><a name="p178619211462"></a><a name="p178619211462"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p97591856562"><a name="p97591856562"></a><a name="p97591856562"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p97861421861"><a name="p97861421861"></a><a name="p97861421861"></a><span id="ph77864211060"><a name="ph77864211060"></a><a name="ph77864211060"></a>源操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row681152195115"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p15875225118"><a name="p15875225118"></a><a name="p15875225118"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p6805255117"><a name="p6805255117"></a><a name="p6805255117"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p6845214516"><a name="p6845214516"></a><a name="p6845214516"></a>目的操作数，类型为MaskTensor。</p>
</td>
</tr>
<tr id="row1813524519"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p3805214518"><a name="p3805214518"></a><a name="p3805214518"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1489521516"><a name="p1489521516"></a><a name="p1489521516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p88105215512"><a name="p88105215512"></a><a name="p88105215512"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row5845213515"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p993521510"><a name="p993521510"></a><a name="p993521510"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1918529514"><a name="p1918529514"></a><a name="p1918529514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p19165214516"><a name="p19165214516"></a><a name="p19165214516"></a>offset决定了srcReg中数据搬运的起始地址，当数据类型为T为b16时，计算公式为offset* 16，由于VL为256Byte，因此此时offset取值范围为0-15，同理，T为b32数据类型时，计算公式为offset* 8，offset的取值范围为0-31。</p>
<p id="p18342527145117"><a name="p18342527145117"></a><a name="p18342527145117"></a>当数据类型为B16时，dst[i] = srcReg[offset*VL/16 + i/2]，按bit位进行计算。</p>
<p id="p234211275519"><a name="p234211275519"></a><a name="p234211275519"></a>当数据类型为B32时，dst[i] = srcReg[offset*VL/32 + i/4],，按bit位进行计算。</p>
</td>
</tr>
</tbody>
</table>

**图 1**  Offset功能示意图<a name="fig115968013431"></a>  
<!-- img2text -->
```
B32数据类型 offset = 0 时
srcReg和dst皆按bit展开

  0Byte                                  8Byte
    ↓                                      ↓
┌────────┐
│ srcReg │
└────────┘
        ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
        │ 0 │ │ 1 │ │ 0 │ │ 1 │ │ 0 │ │ 0 │ │…│     │ 0 │ │ 1 │ │ 0 │ │ 1 │     │ 1 │
        └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘ └───┘ └───┘ └───┘     └───┘
          ↘    ↓    ↘    ↘    ↘    ↘    ↘              ↘    ↘    ↘    ↘            ↘
           ↘   ↓     ↘    ↘    ↘    ↘    ↘              ↘    ↘    ↘    ↘            ↘

┌─────┐
│ dst │
└─────┘
        ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
        │ 0 │ │ 0 │ │ 0 │ │ 0 │ │ 1 │ │ 1 │ │ 1 │     │…│ │ 1 │ │ 1 │ │ 1 │     │ 1 │
        └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘ └───┘ └───┘ └───┘     └───┘


24Byte                                 32Byte
   ↓                                      ↓
B32数据类型 offset = 3 时
srcReg和dst皆按bit展开

┌────────┐
│ srcReg │
└────────┘
        ┌───┐     ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐
        │ 0 │     │ 0 │ │ 1 │ │ 0 │ │ 0 │ │ 1 │ │ 0 │     │…│ │ 0 │ │ 1 │ │ 1 │     │ 0 │
        └───┘     └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘ └───┘ └───┘ └───┘     └───┘
                    ↓    ↘    ↘    ↘    ↘    ↘              ↓    ↘    ↘    ↘           

┌─────┐
│ dst │
└─────┘
        ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐ ┌───┐ ┌───┐ ┌───┐
        │ 0 │ │ 0 │ │ 0 │ │ 0 │ │ 1 │ │ 1 │ │ 1 │     │ 1 │ │…│ │ 1 │ │ 1 │
        └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘ └───┘ └───┘ └───┘
```

说明:
- 图名：图 1 Offset功能示意图
- 上半部分文字：`B32数据类型 offset = 0 时 srcReg和dst皆按bit展开`
- 下半部分文字：`B32数据类型 offset = 3 时 srcReg和dst皆按bit展开`
- 上半部分标注位置：左侧为 `0Byte`，右侧为 `8Byte`
- 下半部分标注位置：左侧为 `24Byte`，右侧为 `32Byte`
- 图中箭头表示 srcReg 中选定bit按 offset 规则映射/展开到 dst 对应bit位置
- 由于原图存在多条扇出斜向箭头且部分bit被省略为 `...`，无法在 ASCII 中一一精确复现全部连接关系，只保留了可识别的主要映射趋势和全部文字标注

## 返回值说明<a name="section1575141714439"></a>

无

## 约束说明<a name="section11585101304320"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template <typename T>
__simd_vf__ inline void LoadAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::MaskReg mask;;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);
        AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(mask, srcAddr, offset);
        AscendC::MicroAPI::StoreAlign(dstAddr, mask, offset);
    }
}

template <typename T, int16_t offset>
__simd_vf__ inline void MaskGenWithRegTensorVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();
    AscendC::MicroAPI::LoadAlign(srcReg, srcAddr);
    AscendC::MicroAPI::MaskGenWithRegTensor<T, offset>(mask, srcReg);
    AscendC::MicroAPI::StoreAlign(dstAddr, mask);
}
```

