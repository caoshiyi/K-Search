# 连续对齐搬出<a name="ZH-CN_TOPIC_0000002554423847"></a>

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

Reg矢量计算数据搬运接口，适用于从RegTensor连续对齐搬出到UB。

## 函数原型<a name="section620mcpsimp"></a>

```
// 单搬出模式 POST_MODE_NORMAL场景
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, MaskReg& mask);

// 单搬出模式 POST_MODE_UPDATE场景
template <typename T = DefaultType, PostLiteral postMode, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T*& dstAddr, U& srcReg, int32_t postUpdateStride, MaskReg& mask);

// 单搬出模式 使用AddrReg存储偏移量
template <typename T = DefaultType, StoreDist dist = StoreDist::DIST_NORM, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg, AddrReg offset, MaskReg& mask);

// 双搬出模式 POST_MODE_NORMAL场景
template <typename T = DefaultType, StoreDist dist, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg0, U& srcReg1, MaskReg& mask);

// 双搬出模式 使用AddrReg 存储偏移量
template <typename T = DefaultType, StoreDist dist, typename U>
__simd_callee__ inline void StoreAlign(__ubuf__ T* dstAddr, U& srcReg0, U& srcReg1, AddrReg offset, MaskReg& mask);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  StoreDist模板参数说明（单搬出模式）

<a name="table20720517613"></a>
<table><thead align="left"><tr id="row1389513610"><th class="cellrowborder" valign="top" width="15.479999999999999%" id="mcps1.2.4.1.1"><p id="p18895113612"><a name="p18895113612"></a><a name="p18895113612"></a><strong id="b1588517611"><a name="b1588517611"></a><a name="b1588517611"></a>StoreDist</strong></p>
</th>
<th class="cellrowborder" valign="top" width="71.36%" id="mcps1.2.4.1.2"><p id="p1286514615"><a name="p1286514615"></a><a name="p1286514615"></a><strong id="b782512613"><a name="b782512613"></a><a name="b782512613"></a>含义</strong></p>
</th>
<th class="cellrowborder" valign="top" width="13.16%" id="mcps1.2.4.1.3"><p id="p192064505562"><a name="p192064505562"></a><a name="p192064505562"></a><strong id="b182069500567"><a name="b182069500567"></a><a name="b182069500567"></a>对齐约束（Byte）</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row14815511662"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p18888192392517"><a name="p18888192392517"></a><a name="p18888192392517"></a>DIST_NORM_B8</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p48881423202511"><a name="p48881423202511"></a><a name="p48881423202511"></a>正常模式，搬运VL数据，数据类型为b8。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p14888923112510"><a name="p14888923112510"></a><a name="p14888923112510"></a>32</p>
</td>
</tr>
<tr id="row1784511268"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p10889102322516"><a name="p10889102322516"></a><a name="p10889102322516"></a>DIST_NORM_B16</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p188893235251"><a name="p188893235251"></a><a name="p188893235251"></a>正常模式，搬运VL数据，数据类型为b16。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p10889172352519"><a name="p10889172352519"></a><a name="p10889172352519"></a>32</p>
</td>
</tr>
<tr id="row285511260"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p18889623182516"><a name="p18889623182516"></a><a name="p18889623182516"></a>DIST_NORM_B32</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p108893234258"><a name="p108893234258"></a><a name="p108893234258"></a>正常模式，搬运VL数据，数据类型为b32。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p9889102314257"><a name="p9889102314257"></a><a name="p9889102314257"></a>32</p>
</td>
</tr>
<tr id="row199880479245"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p134572472253"><a name="p134572472253"></a><a name="p134572472253"></a>DIST_FIRST_ELEMENT_B8</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p20889923162512"><a name="p20889923162512"></a><a name="p20889923162512"></a>忽略mask，向dst中搬运src第一个元素，数据类型为b8。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p7889823162519"><a name="p7889823162519"></a><a name="p7889823162519"></a>1</p>
</td>
</tr>
<tr id="row998018116252"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p9906165118255"><a name="p9906165118255"></a><a name="p9906165118255"></a>DIST_FIRST_ELEMENT_B16</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p6889223122514"><a name="p6889223122514"></a><a name="p6889223122514"></a>忽略mask，向dst中搬运src第一个元素，数据类型为b16。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p0889112302515"><a name="p0889112302515"></a><a name="p0889112302515"></a>2</p>
</td>
</tr>
<tr id="row20342145722416"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p2051917553251"><a name="p2051917553251"></a><a name="p2051917553251"></a>DIST_FIRST_ELEMENT_B32</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p88891423192515"><a name="p88891423192515"></a><a name="p88891423192515"></a>忽略mask，向dst中搬运src第一个元素，数据类型为b32。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p16889923132517"><a name="p16889923132517"></a><a name="p16889923132517"></a>4</p>
</td>
</tr>
<tr id="row244620112518"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p19889112319254"><a name="p19889112319254"></a><a name="p19889112319254"></a>DIST_PACK_B16</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p1889162312259"><a name="p1889162312259"></a><a name="p1889162312259"></a>压缩模式，数据类型为b16，根据mask，将src中有效元素的低半部分bit数据连续存储于dst中</p>
<p id="p588942311251"><a name="p588942311251"></a><a name="p588942311251"></a>例：数据类型为uint16_t，mask配置为所有元素有效，dst长度需要为VL/2，执行结果如下：</p>
<p id="p168891323112513"><a name="p168891323112513"></a><a name="p168891323112513"></a>src: [0x3210, 0x7654, 0xBA98, 0xFEDC, ..., 0xFEDC, 0xBA98, 0x7654, 0x3210]</p>
<p id="p1488902312515"><a name="p1488902312515"></a><a name="p1488902312515"></a>dst: [0x5410, 0xDC98, ... 0x98DC, 0x1054]</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p198892232258"><a name="p198892232258"></a><a name="p198892232258"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row896814589240"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p089002310252"><a name="p089002310252"></a><a name="p089002310252"></a>DIST_PACK_B32</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p15890162392514"><a name="p15890162392514"></a><a name="p15890162392514"></a>压缩模式，数据类型为b32，根据mask，将src中有效元素的低半部分bit数据连续存储于dst中。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p68901523102518"><a name="p68901523102518"></a><a name="p68901523102518"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row17496503248"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p1289012239257"><a name="p1289012239257"></a><a name="p1289012239257"></a>DIST_PACK_B64</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p2890162312512"><a name="p2890162312512"></a><a name="p2890162312512"></a>压缩模式，数据类型为b64，根据mask，将src中有效元素的低半部分bit数据连续存储于dst中。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p178909235258"><a name="p178909235258"></a><a name="p178909235258"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row173695462242"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p88902236252"><a name="p88902236252"></a><a name="p88902236252"></a>DIST_PACK4_B32</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p5891142312252"><a name="p5891142312252"></a><a name="p5891142312252"></a>压缩模式，数据类型为b32，根据mask，将src中有效元素的低8bit（四分之一）数据连续存储于dst中。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p1589112352517"><a name="p1589112352517"></a><a name="p1589112352517"></a>min(32, VL/4)</p>
</td>
</tr>
<tr id="row15855119616"><td class="cellrowborder" valign="top" width="15.479999999999999%" headers="mcps1.2.4.1.1 "><p id="p389192312516"><a name="p389192312516"></a><a name="p389192312516"></a>DIST_NORM</p>
</td>
<td class="cellrowborder" valign="top" width="71.36%" headers="mcps1.2.4.1.2 "><p id="p68925235256"><a name="p68925235256"></a><a name="p68925235256"></a>正常模式，搬运VL数据，支持数据类型b8/b16/b32，根据模板T确定。</p>
</td>
<td class="cellrowborder" valign="top" width="13.16%" headers="mcps1.2.4.1.3 "><p id="p15891172319256"><a name="p15891172319256"></a><a name="p15891172319256"></a>32</p>
</td>
</tr>
</tbody>
</table>

**表 2**  StoreDist模板参数说明（双搬出模式）

<a name="table13712143342316"></a>
<table><thead align="left"><tr id="row47121533142310"><th class="cellrowborder" valign="top" width="15.290000000000001%" id="mcps1.2.4.1.1"><p id="p8712633202313"><a name="p8712633202313"></a><a name="p8712633202313"></a><strong id="b571203372318"><a name="b571203372318"></a><a name="b571203372318"></a>StoreDist</strong></p>
</th>
<th class="cellrowborder" valign="top" width="71.45%" id="mcps1.2.4.1.2"><p id="p6712233142320"><a name="p6712233142320"></a><a name="p6712233142320"></a><strong id="b1371220331233"><a name="b1371220331233"></a><a name="b1371220331233"></a>含义</strong></p>
</th>
<th class="cellrowborder" valign="top" width="13.26%" id="mcps1.2.4.1.3"><p id="p15712183312235"><a name="p15712183312235"></a><a name="p15712183312235"></a><strong id="b471243362312"><a name="b471243362312"></a><a name="b471243362312"></a>对齐约束(Byte)</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row9712193352318"><td class="cellrowborder" valign="top" width="15.290000000000001%" headers="mcps1.2.4.1.1 "><p id="p1377423715273"><a name="p1377423715273"></a><a name="p1377423715273"></a>DIST_INTLV_B8</p>
</td>
<td class="cellrowborder" valign="top" width="71.45%" headers="mcps1.2.4.1.2 "><p id="p16774163711279"><a name="p16774163711279"></a><a name="p16774163711279"></a>双搬出模式，数据类型为b8，忽略mask，将src0，src1中的元素交错存储于dst中，dst长度需要为VL*2。</p>
<p id="p1777473732716"><a name="p1777473732716"></a><a name="p1777473732716"></a>例：数据类型为uint8_t：</p>
<p id="p1077403720277"><a name="p1077403720277"></a><a name="p1077403720277"></a>src0: [0, 2, 4, 6, ... 254, 0, 2, 4, ... 252, 254]</p>
<p id="p1777473702718"><a name="p1777473702718"></a><a name="p1777473702718"></a>src1: [1, 3, 5, 7, ... 255, 1, 3, 5, ... 253, 255]</p>
<p id="p177741637172714"><a name="p177741637172714"></a><a name="p177741637172714"></a>dst: [0, 1, 2, 3, ..., 254, 255, 0, 1, 2, 3, ... 253, 254, 255]</p>
</td>
<td class="cellrowborder" valign="top" width="13.26%" headers="mcps1.2.4.1.3 "><p id="p97744373279"><a name="p97744373279"></a><a name="p97744373279"></a>32</p>
</td>
</tr>
<tr id="row10712183362315"><td class="cellrowborder" valign="top" width="15.290000000000001%" headers="mcps1.2.4.1.1 "><p id="p1377403714272"><a name="p1377403714272"></a><a name="p1377403714272"></a>DIST_INTLV_B16</p>
</td>
<td class="cellrowborder" valign="top" width="71.45%" headers="mcps1.2.4.1.2 "><p id="p1977403722714"><a name="p1977403722714"></a><a name="p1977403722714"></a>双搬出模式，数据类型为b16，忽略mask，将src0，src1中的元素交错存储于dst中，dst长度需要为VL*2。</p>
</td>
<td class="cellrowborder" valign="top" width="13.26%" headers="mcps1.2.4.1.3 "><p id="p1477443718273"><a name="p1477443718273"></a><a name="p1477443718273"></a>32</p>
</td>
</tr>
<tr id="row107134339239"><td class="cellrowborder" valign="top" width="15.290000000000001%" headers="mcps1.2.4.1.1 "><p id="p12775193772711"><a name="p12775193772711"></a><a name="p12775193772711"></a>DIST_INTLV_B32</p>
</td>
<td class="cellrowborder" valign="top" width="71.45%" headers="mcps1.2.4.1.2 "><p id="p077563715276"><a name="p077563715276"></a><a name="p077563715276"></a>双搬出模式，数据类型为b32，忽略mask，将src0，src1中的元素交错存储于dst中，dst长度需要为VL*2。</p>
</td>
<td class="cellrowborder" valign="top" width="13.26%" headers="mcps1.2.4.1.3 "><p id="p12775133792715"><a name="p12775133792715"></a><a name="p12775133792715"></a>32</p>
</td>
</tr>
</tbody>
</table>

**表 3**  单搬出模式POST\_MODE\_NORMAL场景参数说明

<a name="table208471048514"></a>
<table><thead align="left"><tr id="row784774817112"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p108471248518"><a name="p108471248518"></a><a name="p108471248518"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p14847134813115"><a name="p14847134813115"></a><a name="p14847134813115"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p1847104813110"><a name="p1847104813110"></a><a name="p1847104813110"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1897143561016"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p3177155261014"><a name="p3177155261014"></a><a name="p3177155261014"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p18177115221013"><a name="p18177115221013"></a><a name="p18177115221013"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p12177205271010"><a name="p12177205271010"></a><a name="p12177205271010"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row12812937171011"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19177165291010"><a name="p19177165291010"></a><a name="p19177165291010"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p517811526102"><a name="p517811526102"></a><a name="p517811526102"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p61786521109"><a name="p61786521109"></a><a name="p61786521109"></a>StoreDist模板参数，enum class类型，具体的取值请参考<a href="#table20720517613">表1</a>。</p>
</td>
</tr>
<tr id="row4845618151214"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1784518187125"><a name="p1784518187125"></a><a name="p1784518187125"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p188451018141218"><a name="p188451018141218"></a><a name="p188451018141218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1784512188121"><a name="p1784512188121"></a><a name="p1784512188121"></a>RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row532873619293"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19164184592918"><a name="p19164184592918"></a><a name="p19164184592918"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p141641345142912"><a name="p141641345142912"></a><a name="p141641345142912"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1416434552910"><a name="p1416434552910"></a><a name="p1416434552910"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row94281746151014"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p517813529108"><a name="p517813529108"></a><a name="p517813529108"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1617815218101"><a name="p1617815218101"></a><a name="p1617815218101"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1817805214104"><a name="p1817805214104"></a><a name="p1817805214104"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row13763101973218"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1143532313321"><a name="p1143532313321"></a><a name="p1143532313321"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p154357235328"><a name="p154357235328"></a><a name="p154357235328"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p4435152318326"><a name="p4435152318326"></a><a name="p4435152318326"></a>MaskReg类型，指示在搬运过程中哪些元素有效。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  单搬出模POST\_MODE\_UPDATE场景参数说明

<a name="table18677182661320"></a>
<table><thead align="left"><tr id="row15677126141319"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p1867715265134"><a name="p1867715265134"></a><a name="p1867715265134"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p19677726131315"><a name="p19677726131315"></a><a name="p19677726131315"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p7677226121320"><a name="p7677226121320"></a><a name="p7677226121320"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1767732619132"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p66771926131310"><a name="p66771926131310"></a><a name="p66771926131310"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1767752618139"><a name="p1767752618139"></a><a name="p1767752618139"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p4677526131318"><a name="p4677526131318"></a><a name="p4677526131318"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row19843185220142"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1841485818141"><a name="p1841485818141"></a><a name="p1841485818141"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p5414258171419"><a name="p5414258171419"></a><a name="p5414258171419"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p14697155195612"><a name="p14697155195612"></a><a name="p14697155195612"></a>用于控制是否使能post update，<a href="PostLiteral.md">PostLiteral</a>类型。</p>
</td>
</tr>
<tr id="row8677192651311"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1467782621313"><a name="p1467782621313"></a><a name="p1467782621313"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p267792671314"><a name="p267792671314"></a><a name="p267792671314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p152731742142120"><a name="p152731742142120"></a><a name="p152731742142120"></a>StoreDist模板参数，enum class类型，具体的取值请参考<a href="#table20720517613">表1</a>。</p>
</td>
</tr>
<tr id="row8678926201317"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p9678112614133"><a name="p9678112614133"></a><a name="p9678112614133"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p7678226111314"><a name="p7678226111314"></a><a name="p7678226111314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p6678122615135"><a name="p6678122615135"></a><a name="p6678122615135"></a>RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row121912131307"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1716819174306"><a name="p1716819174306"></a><a name="p1716819174306"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p11684176306"><a name="p11684176306"></a><a name="p11684176306"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p416819177309"><a name="p416819177309"></a><a name="p416819177309"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row1767892613133"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p17161251306"><a name="p17161251306"></a><a name="p17161251306"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p101617255301"><a name="p101617255301"></a><a name="p101617255301"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p31620252305"><a name="p31620252305"></a><a name="p31620252305"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row1367820264135"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p2242617191515"><a name="p2242617191515"></a><a name="p2242617191515"></a>postUpdateStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1567812263136"><a name="p1567812263136"></a><a name="p1567812263136"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p7678132641315"><a name="p7678132641315"></a><a name="p7678132641315"></a>实际搬运UB起始地址为dstAddr，搬运后自动执行地址更新：dstAddr +=  postUpdateStride。</p>
</td>
</tr>
<tr id="row199951885306"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p39951289307"><a name="p39951289307"></a><a name="p39951289307"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p169687914314"><a name="p169687914314"></a><a name="p169687914314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p7968189203118"><a name="p7968189203118"></a><a name="p7968189203118"></a>MaskReg类型，指示在搬运过程中哪些元素有效。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  单搬出模式使用AddrReg存储偏移量场景参数说明

<a name="table3732601088"></a>
<table><thead align="left"><tr id="row27329017819"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p197321801587"><a name="p197321801587"></a><a name="p197321801587"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p37321701986"><a name="p37321701986"></a><a name="p37321701986"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p4732801818"><a name="p4732801818"></a><a name="p4732801818"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row473217019812"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p573290485"><a name="p573290485"></a><a name="p573290485"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p5732901984"><a name="p5732901984"></a><a name="p5732901984"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1073211017811"><a name="p1073211017811"></a><a name="p1073211017811"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row13733501985"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p117331203817"><a name="p117331203817"></a><a name="p117331203817"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p5733110584"><a name="p5733110584"></a><a name="p5733110584"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p784621742217"><a name="p784621742217"></a><a name="p784621742217"></a>StoreDist模板参数，enum class类型，具体的取值请参考<a href="#table20720517613">表1</a>。</p>
</td>
</tr>
<tr id="row07333016818"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p12733404819"><a name="p12733404819"></a><a name="p12733404819"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p9733508815"><a name="p9733508815"></a><a name="p9733508815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p18461017132219"><a name="p18461017132219"></a><a name="p18461017132219"></a>RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row1733201683"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p197331206810"><a name="p197331206810"></a><a name="p197331206810"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p13733110688"><a name="p13733110688"></a><a name="p13733110688"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p15733170382"><a name="p15733170382"></a><a name="p15733170382"></a>目的操作数。</p>
</td>
</tr>
<tr id="row10733001883"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p197335016810"><a name="p197335016810"></a><a name="p197335016810"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p473314014816"><a name="p473314014816"></a><a name="p473314014816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p873310782"><a name="p873310782"></a><a name="p873310782"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row95513351793"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p133769582513"><a name="p133769582513"></a><a name="p133769582513"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p3376145819520"><a name="p3376145819520"></a><a name="p3376145819520"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p23761058154"><a name="p23761058154"></a><a name="p23761058154"></a>实际搬运地址UB为dstAddr + offset。</p>
</td>
</tr>
<tr id="row173317017810"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p773310589"><a name="p773310589"></a><a name="p773310589"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p37331401181"><a name="p37331401181"></a><a name="p37331401181"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p2073318017816"><a name="p2073318017816"></a><a name="p2073318017816"></a>MaskReg类型，指示在搬运过程中哪些元素有效。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  双搬出模式POST\_MODE\_NORMAL场景参数说明

<a name="table6718218153118"></a>
<table><thead align="left"><tr id="row147181318153115"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p8718718183118"><a name="p8718718183118"></a><a name="p8718718183118"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p127186181315"><a name="p127186181315"></a><a name="p127186181315"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p1471821823120"><a name="p1471821823120"></a><a name="p1471821823120"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1471801811317"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1071914183310"><a name="p1071914183310"></a><a name="p1071914183310"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1771961873116"><a name="p1771961873116"></a><a name="p1771961873116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p471931863119"><a name="p471931863119"></a><a name="p471931863119"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row27191118123118"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p8719918123110"><a name="p8719918123110"></a><a name="p8719918123110"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1719141816310"><a name="p1719141816310"></a><a name="p1719141816310"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p18667184982215"><a name="p18667184982215"></a><a name="p18667184982215"></a>StoreDist模板参数，enum class类型，具体的取值请参考<a href="#table13712143342316">表2</a>。</p>
</td>
</tr>
<tr id="row1071910183319"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1371941893120"><a name="p1371941893120"></a><a name="p1371941893120"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p571971817311"><a name="p571971817311"></a><a name="p571971817311"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p14667949102212"><a name="p14667949102212"></a><a name="p14667949102212"></a>RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row177191618133112"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p171961853112"><a name="p171961853112"></a><a name="p171961853112"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p671921803119"><a name="p671921803119"></a><a name="p671921803119"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p147198181313"><a name="p147198181313"></a><a name="p147198181313"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row107198184319"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1471931873118"><a name="p1471931873118"></a><a name="p1471931873118"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p11719818193116"><a name="p11719818193116"></a><a name="p11719818193116"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p47191718193119"><a name="p47191718193119"></a><a name="p47191718193119"></a>第一个源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row4761749143112"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p72771654153112"><a name="p72771654153112"></a><a name="p72771654153112"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p42771454183114"><a name="p42771454183114"></a><a name="p42771454183114"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p13277195418313"><a name="p13277195418313"></a><a name="p13277195418313"></a>第二个源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row55434212312"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1059304773116"><a name="p1059304773116"></a><a name="p1059304773116"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p115931647193117"><a name="p115931647193117"></a><a name="p115931647193117"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p45931047133116"><a name="p45931047133116"></a><a name="p45931047133116"></a>MaskReg类型，指示在搬运过程中哪些元素有效。</p>
</td>
</tr>
</tbody>
</table>

**表 7**  双搬出模式使用AddrReg存储偏移量场景参数说明

<a name="table1446145251013"></a>
<table><thead align="left"><tr id="row15461105221017"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p18461115213104"><a name="p18461115213104"></a><a name="p18461115213104"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p84611652161018"><a name="p84611652161018"></a><a name="p84611652161018"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p20461252131013"><a name="p20461252131013"></a><a name="p20461252131013"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row34611552111014"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p184611052131018"><a name="p184611052131018"></a><a name="p184611052131018"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1146185201010"><a name="p1146185201010"></a><a name="p1146185201010"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p16461195220101"><a name="p16461195220101"></a><a name="p16461195220101"></a>模板参数，支持的数据类型为b8/b16/b32。</p>
</td>
</tr>
<tr id="row4461852191020"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19461185291016"><a name="p19461185291016"></a><a name="p19461185291016"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1461652181013"><a name="p1461652181013"></a><a name="p1461652181013"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p4173111520234"><a name="p4173111520234"></a><a name="p4173111520234"></a>StoreDist模板参数，enum class类型，具体的取值请参考<a href="#table13712143342316">表2</a>。</p>
</td>
</tr>
<tr id="row246285215107"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p14462652181019"><a name="p14462652181019"></a><a name="p14462652181019"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p346217529103"><a name="p346217529103"></a><a name="p346217529103"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p161732157233"><a name="p161732157233"></a><a name="p161732157233"></a>RegTensor类型， 例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row446218524107"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p154621252181018"><a name="p154621252181018"></a><a name="p154621252181018"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p446255271013"><a name="p446255271013"></a><a name="p446255271013"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p18462155212102"><a name="p18462155212102"></a><a name="p18462155212102"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row1046214522101"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1546215210106"><a name="p1546215210106"></a><a name="p1546215210106"></a>srcReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p846295251010"><a name="p846295251010"></a><a name="p846295251010"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1546265219106"><a name="p1546265219106"></a><a name="p1546265219106"></a>第一个源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row11462152171010"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1462165210105"><a name="p1462165210105"></a><a name="p1462165210105"></a>srcReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p846210525106"><a name="p846210525106"></a><a name="p846210525106"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p346275231013"><a name="p346275231013"></a><a name="p346275231013"></a>第二个源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row15215115171114"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p774921971117"><a name="p774921971117"></a><a name="p774921971117"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1674911191110"><a name="p1674911191110"></a><a name="p1674911191110"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p14749719121118"><a name="p14749719121118"></a><a name="p14749719121118"></a>实际搬运地址UB为dstAddr + offset。</p>
</td>
</tr>
<tr id="row1462125217105"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1462185211013"><a name="p1462185211013"></a><a name="p1462185211013"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p446245215108"><a name="p446245215108"></a><a name="p446245215108"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p164621652121013"><a name="p164621652121013"></a><a name="p164621652121013"></a>MaskReg类型，指示在搬运过程中哪些元素有效。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1575141714439"></a>

无

## 约束说明<a name="section11585101304320"></a>

b64数据类型只支持StoreDist中的DIST\_NORM模式。

## 调用示例<a name="section642mcpsimp"></a>

```
// 单搬入/单搬出 POST_MODE_NORMAL 场景
__simd_vf__ inline void ComputeMode01(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dstSize,
    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(dstSize);
        AscendC::MicroAPI::LoadAlign(dstReg, srcAddr + i * oneRepeatSize);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}

// 单搬入/单搬出 POST_MODE_UPDATE 场景
__simd_vf__ inline void ComputeMode02(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t dstSize,
    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<T>(dstSize);
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstReg, srcAddr, oneRepeatSize);
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, dstReg, oneRepeatSize, mask);
    }
}

// 单搬入/单搬出使用 AddrReg 存储偏移量场景
__simd_vf__ inline void ComputeMode03(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();
    AscendC::MicroAPI::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadAlign(dstReg, srcAddr, aReg);
        AscendC::MicroAPI::StoreAlign(dstAddr, dstReg, aReg, mask);
    }
}

// 双搬入/双搬出 POST_MODE_NORMAL 场景
__simd_vf__ inline void ComputeMode04(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize,
    uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B8>(srcReg0, srcReg0, srcAddr + i * oneRepeatSize);
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_INTLV_B8>(dstAddr + i * oneRepeatSize, srcReg0, srcReg1, mask);
    }
}

// 双搬入/双搬出使用 AddrReg 存储偏移量场景
__simd_vf__ inline void ComputeMode05(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();
    AscendC::MicroAPI::AddrReg aReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B8>(srcReg0, srcReg1, srcAddr, aReg);
        AscendC::MicroAPI::StoreAlign<T, AscendC::MicroAPI::StoreDist::DIST_INTLV_B8>(dstAddr, srcReg0, srcReg1, aReg, mask);
    }
}
```

