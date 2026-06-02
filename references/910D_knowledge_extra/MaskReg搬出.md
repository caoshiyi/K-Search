# MaskReg搬出<a name="ZH-CN_TOPIC_0000002523303882"></a>

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

Reg矢量计算数据搬运接口，适用于从MaskReg搬出到UB。

## 函数原型<a name="section620mcpsimp"></a>

```
// MaskReg搬出使用AddrReg存储偏移量
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, MaskReg& mask, AddrReg offset);

// MaskReg搬出POST_MODE_NORMAL场景
template <typename T, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, MaskReg& mask);

// MaskReg搬出POST_MODE_UPDATE场景
template <typename T, PostLiteral postMode, MaskDist dist = MaskDist::DIST_NORM>
__simd_callee__ inline void StoreAlign(__ubuf__ T*& dstAddr, MaskReg& mask, int32_t offset);

// MaskReg非对齐搬出
template <typename T>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T*& dstAddr, MaskReg& mask, UnalignRegForStore& ureg);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  MaskReg搬出使用AddrReg存储偏移量参数说明

<a name="table14132101714462"></a>
<table><thead align="left"><tr id="row19176617124620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p117621724612"><a name="p117621724612"></a><a name="p117621724612"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p417681717463"><a name="p417681717463"></a><a name="p417681717463"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p5176017164614"><a name="p5176017164614"></a><a name="p5176017164614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row0176117164610"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p12176171764613"><a name="p12176171764613"></a><a name="p12176171764613"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p5176191734615"><a name="p5176191734615"></a><a name="p5176191734615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p5614195515536"><a name="p5614195515536"></a><a name="p5614195515536"></a>支持的数据类型为b8/b16/b32。</p>
</td>
</tr>
<tr id="row1217616175468"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p5176121754619"><a name="p5176121754619"></a><a name="p5176121754619"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p12176161714468"><a name="p12176161714468"></a><a name="p12176161714468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p62011179525"><a name="p62011179525"></a><a name="p62011179525"></a>搬运模式，MaskDist类型。取值如下：</p>
<a name="ul122012176529"></a><a name="ul122012176529"></a><ul id="ul122012176529"><li>DIST_NORM，对齐约束为VL/8Byte，正常模式，搬运VL/8Byte数据。</li><li>DIST_PACK，对齐约束为VL/16Byte，下采样模式，搬运VL/16Byte数据，每间隔1bit被舍弃。</li></ul>
</td>
</tr>
<tr id="row1417671716463"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p12176201754617"><a name="p12176201754617"></a><a name="p12176201754617"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p891201384220"><a name="p891201384220"></a><a name="p891201384220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p2176141714466"><a name="p2176141714466"></a><a name="p2176141714466"></a>源操作数，类型为MaskTensor。</p>
</td>
</tr>
<tr id="row958051624416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p9580161634416"><a name="p9580161634416"></a><a name="p9580161634416"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p2581416144417"><a name="p2581416144417"></a><a name="p2581416144417"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p195811716134418"><a name="p195811716134418"></a><a name="p195811716134418"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row126641434416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1671114174418"><a name="p1671114174418"></a><a name="p1671114174418"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p567151494415"><a name="p567151494415"></a><a name="p567151494415"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p928118257494"><a name="p928118257494"></a><a name="p928118257494"></a>实际搬运UB起始地址为 srcAddr + offset。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  MaskReg搬出POST\_MODE\_NORMAL场景参数说明

<a name="table15062617384"></a>
<table><thead align="left"><tr id="row1850192663818"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p55022613381"><a name="p55022613381"></a><a name="p55022613381"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p150192683817"><a name="p150192683817"></a><a name="p150192683817"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p16505266383"><a name="p16505266383"></a><a name="p16505266383"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19501526173810"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p150426153811"><a name="p150426153811"></a><a name="p150426153811"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1450152683811"><a name="p1450152683811"></a><a name="p1450152683811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p18502267385"><a name="p18502267385"></a><a name="p18502267385"></a>支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row19503269381"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p13501226203813"><a name="p13501226203813"></a><a name="p13501226203813"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p450192643820"><a name="p450192643820"></a><a name="p450192643820"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p943573417506"><a name="p943573417506"></a><a name="p943573417506"></a>搬运模式，MaskDist类型。取值如下：</p>
<a name="ul12435334165019"></a><a name="ul12435334165019"></a><ul id="ul12435334165019"><li>DIST_NORM，对齐约束为VL/8Byte，正常模式，搬运VL/8Byte数据。</li><li>DIST_PACK，对齐约束为VL/16Byte，下采样模式，搬运VL/16Byte数据，每间隔1bit被舍弃。</li></ul>
</td>
</tr>
<tr id="row95082693817"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p450192616380"><a name="p450192616380"></a><a name="p450192616380"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1850102616386"><a name="p1850102616386"></a><a name="p1850102616386"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p650726133815"><a name="p650726133815"></a><a name="p650726133815"></a>源操作数，类型为MaskReg。</p>
</td>
</tr>
<tr id="row1850426193816"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p3501026203811"><a name="p3501026203811"></a><a name="p3501026203811"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p105022623811"><a name="p105022623811"></a><a name="p105022623811"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p5512266382"><a name="p5512266382"></a><a name="p5512266382"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  MaskReg搬出POST\_MODE\_UPDATE场景参数说明

<a name="table083611281388"></a>
<table><thead align="left"><tr id="row4836162833816"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p118361728123818"><a name="p118361728123818"></a><a name="p118361728123818"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p14837152873814"><a name="p14837152873814"></a><a name="p14837152873814"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p7837628163816"><a name="p7837628163816"></a><a name="p7837628163816"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1583718284383"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p0837928173812"><a name="p0837928173812"></a><a name="p0837928173812"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p383722813383"><a name="p383722813383"></a><a name="p383722813383"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p11837202853819"><a name="p11837202853819"></a><a name="p11837202853819"></a>支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row19837152863813"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1383742817382"><a name="p1383742817382"></a><a name="p1383742817382"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1883792812381"><a name="p1883792812381"></a><a name="p1883792812381"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p112470110512"><a name="p112470110512"></a><a name="p112470110512"></a>搬运模式，MaskDist类型。取值如下：</p>
<a name="ul1124711135110"></a><a name="ul1124711135110"></a><ul id="ul1124711135110"><li>DIST_NORM，对齐约束为VL/8Byte，正常模式，搬运VL/8Byte数据。</li><li>DIST_PACK，对齐约束为VL/16Byte，下采样模式，搬运VL/16Byte数据，每间隔1bit被舍弃。</li></ul>
</td>
</tr>
<tr id="row183752813810"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p483720285383"><a name="p483720285383"></a><a name="p483720285383"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p883792893817"><a name="p883792893817"></a><a name="p883792893817"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p1246583315216"><a name="p1246583315216"></a><a name="p1246583315216"></a>用于控制是否使能post update。</p>
<a name="ul121721550125114"></a><a name="ul121721550125114"></a><ul id="ul121721550125114"><li>POST_MODE_NORMAL，正常场景，UB操作数地址不更新。</li></ul>
<a name="ul51081954165116"></a><a name="ul51081954165116"></a><ul id="ul51081954165116"><li>POST_MODE_UPDATE，POST_MODE_UPDATE场景使用，UB地址同时作为输入和输出，每次调用会更新。</li></ul>
</td>
</tr>
<tr id="row98377283382"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1983718282389"><a name="p1983718282389"></a><a name="p1983718282389"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1983712286389"><a name="p1983712286389"></a><a name="p1983712286389"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p583772863818"><a name="p583772863818"></a><a name="p583772863818"></a>源操作数，类型为MaskTensor。</p>
</td>
</tr>
<tr id="row183712283385"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p178371228153810"><a name="p178371228153810"></a><a name="p178371228153810"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p58371928143817"><a name="p58371928143817"></a><a name="p58371928143817"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p1283713283382"><a name="p1283713283382"></a><a name="p1283713283382"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row38383284384"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p78381028183810"><a name="p78381028183810"></a><a name="p78381028183810"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1983882814386"><a name="p1983882814386"></a><a name="p1983882814386"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><div class="p" id="p342815215547"><a name="p342815215547"></a><a name="p342815215547"></a>当offset为int32_t类型时：POST_MODE_NORMAL与POST_MODE_UPDATE含义不一致。<a name="ul883822833811"></a><a name="ul883822833811"></a><ul id="ul883822833811"><li>POST_MODE_NORMAL场景：实际搬运UB起始地址为srcAddr + offset。</li><li>POST_MODE_UPDATE场景：实际搬运UB起始地址为srcAddr，搬运后执行地址更新srcAddr +=  offset。</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

**表 4**  MaskReg非对齐搬出参数说明

<a name="table11733115714442"></a>
<table><thead align="left"><tr id="row8734165774414"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p27341957134415"><a name="p27341957134415"></a><a name="p27341957134415"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.47875212478752%" id="mcps1.2.4.1.2"><p id="p177347576444"><a name="p177347576444"></a><a name="p177347576444"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.25257474252575%" id="mcps1.2.4.1.3"><p id="p1473465754410"><a name="p1473465754410"></a><a name="p1473465754410"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row187343571447"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p137342571447"><a name="p137342571447"></a><a name="p137342571447"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p1734185712449"><a name="p1734185712449"></a><a name="p1734185712449"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p7734165710446"><a name="p7734165710446"></a><a name="p7734165710446"></a>支持的数据类型为b16/b32。</p>
</td>
</tr>
<tr id="row07341357194417"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p167342057174412"><a name="p167342057174412"></a><a name="p167342057174412"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p117341157114410"><a name="p117341157114410"></a><a name="p117341157114410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p16734195714414"><a name="p16734195714414"></a><a name="p16734195714414"></a>源操作数，类型为MaskTensor。</p>
</td>
</tr>
<tr id="row673495724413"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1873435712441"><a name="p1873435712441"></a><a name="p1873435712441"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p37341557124412"><a name="p37341557124412"></a><a name="p37341557124412"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p17351657164415"><a name="p17351657164415"></a><a name="p17351657164415"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row37351557114411"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p073585744415"><a name="p073585744415"></a><a name="p073585744415"></a>ureg</p>
</td>
<td class="cellrowborder" valign="top" width="12.47875212478752%" headers="mcps1.2.4.1.2 "><p id="p157351457204415"><a name="p157351457204415"></a><a name="p157351457204415"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.25257474252575%" headers="mcps1.2.4.1.3 "><p id="p167353572447"><a name="p167353572447"></a><a name="p167353572447"></a>UnalignRegForStore，非对齐寄存器，用于保存非对齐数据，长度32B。调用完<a href="#section618mcpsimp">StoreUnAlign（MaskReg非对齐搬出接口）</a>后，需要调用<a href="连续非对齐搬出.md#section618mcpsimp">StoreUnAlignPost接口</a>，传入该非对齐寄存器，将未写出的数据写出到目的地址中。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1575141714439"></a>

无

## 约束说明<a name="section11585101304320"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template <typename T>
__simd_vf__ inline void StoreAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(count);        
        AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(mask, srcAddr, offset);
        AscendC::MicroAPI::StoreAlign(dstAddr, mask, offset);
    }
}
```

