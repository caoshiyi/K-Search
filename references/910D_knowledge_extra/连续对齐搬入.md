# 连续对齐搬入<a name="ZH-CN_TOPIC_0000002523304770"></a>

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

Reg矢量计算数据搬运接口，适用于从UB连续对齐搬入RegTensor。单搬入模式下，可以将数据从UB搬运到一个目的寄存器，双搬入模式下，可以将数据从UB搬运到两个目的寄存器。

## 函数原型<a name="section620mcpsimp"></a>

```
// 单搬入模式 POST_MODE_NORMAL场景
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr);

// 单搬入模式 POST_MODE_UPDATE场景
template <typename T = DefaultType, PostLiteral postMode, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T*& srcAddr, int32_t postUpdateStride);

// 单搬入模式 使用AddrReg存储偏移量
template <typename T = DefaultType, LoadDist dist = LoadDist::DIST_NORM, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg, __ubuf__ T* srcAddr, AddrReg offset);

// 双搬入模式 POST_MODE_NORMAL场景
template <typename T = DefaultType, LoadDist dist, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg0, U& dstReg1, __ubuf__ T* srcAddr);

// 双搬入模式 POST_MODE_UPDATE场景
template <typename T = DefaultType, PostLiteral postMode, LoadDist dist, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg0, U& dstReg1, __ubuf__ T*& srcAddr, int32_t postUpdateStride);

// 双搬入模式使用 AddrReg存储偏移量
template <typename T = DefaultType, LoadDist dist, typename U>
__simd_callee__ inline void LoadAlign(U& dstReg0, U& dstReg1, __ubuf__ T* srcAddr, AddrReg offset);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  LoadDist模板参数说明（单搬入模式）

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="12.920000000000002%" id="mcps1.2.4.1.1"><p id="p97799368417"><a name="p97799368417"></a><a name="p97799368417"></a><strong id="b1399495113419"><a name="b1399495113419"></a><a name="b1399495113419"></a>LoadDist</strong>取值</p>
</th>
<th class="cellrowborder" valign="top" width="66.60000000000001%" id="mcps1.2.4.1.2"><p id="p87778361341"><a name="p87778361341"></a><a name="p87778361341"></a><strong id="b81351759049"><a name="b81351759049"></a><a name="b81351759049"></a>含义</strong></p>
</th>
<th class="cellrowborder" valign="top" width="20.48%" id="mcps1.2.4.1.3"><p id="p179219511304"><a name="p179219511304"></a><a name="p179219511304"></a><strong id="b7792145183017"><a name="b7792145183017"></a><a name="b7792145183017"></a>搬运对齐约束（Byte）</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row166904181061"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p166244291365"><a name="p166244291365"></a><a name="p166244291365"></a>DIST_NORM</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p1662417291461"><a name="p1662417291461"></a><a name="p1662417291461"></a>正常模式，搬运VL数据</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p6792851173011"><a name="p6792851173011"></a><a name="p6792851173011"></a>32</p>
</td>
</tr>
<tr id="row633311144618"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p1862511291618"><a name="p1862511291618"></a><a name="p1862511291618"></a>DIST_BRC_B8</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p56256291065"><a name="p56256291065"></a><a name="p56256291065"></a>搬运一个b8类型的数据，并Broadcast到所有元素位置</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p1279265114307"><a name="p1279265114307"></a><a name="p1279265114307"></a>1</p>
</td>
</tr>
<tr id="row5258132018620"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p1462511291967"><a name="p1462511291967"></a><a name="p1462511291967"></a>DIST_BRC_B16</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p18625132919611"><a name="p18625132919611"></a><a name="p18625132919611"></a>搬运一个b16类型的数据，并Broadcast到所有元素位置</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p17921510306"><a name="p17921510306"></a><a name="p17921510306"></a>2</p>
</td>
</tr>
<tr id="row164720114610"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p5625122920619"><a name="p5625122920619"></a><a name="p5625122920619"></a>DIST_BRC_B32</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p1562572910614"><a name="p1562572910614"></a><a name="p1562572910614"></a>搬运一个b32类型的数据，并Broadcast到所有元素位置</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p97921651133015"><a name="p97921651133015"></a><a name="p97921651133015"></a>4</p>
</td>
</tr>
<tr id="row972161611617"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p196251529063"><a name="p196251529063"></a><a name="p196251529063"></a>DIST_US_B8</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p762512291563"><a name="p762512291563"></a><a name="p762512291563"></a>数据2倍上采样，加载VL/2个数据，每个输入元素重复两次，数据类型为b8</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p13792105119300"><a name="p13792105119300"></a><a name="p13792105119300"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row1591935916516"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p1462511296620"><a name="p1462511296620"></a><a name="p1462511296620"></a>DIST_US_B16</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p1162542919614"><a name="p1162542919614"></a><a name="p1162542919614"></a>数据2倍上采样，加载VL/2个数据，每个输入元素重复两次，数据类型为b16</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p1479285133016"><a name="p1479285133016"></a><a name="p1479285133016"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row145853121862"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p186251291967"><a name="p186251291967"></a><a name="p186251291967"></a>DIST_DS_B8</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p96259291762"><a name="p96259291762"></a><a name="p96259291762"></a>数据2倍下采样，加载2倍VL的数据，数据每隔一个保留，数据类型为b8</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p77927515304"><a name="p77927515304"></a><a name="p77927515304"></a>32</p>
</td>
</tr>
<tr id="row45231471367"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p0625162910611"><a name="p0625162910611"></a><a name="p0625162910611"></a>DIST_DS_B16</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p9625529869"><a name="p9625529869"></a><a name="p9625529869"></a>数据2倍下采样，加载2倍VL的数据，数据每隔一个保留，数据类型为b16</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p20792185119307"><a name="p20792185119307"></a><a name="p20792185119307"></a>32</p>
</td>
</tr>
<tr id="row154651795614"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p4626172911613"><a name="p4626172911613"></a><a name="p4626172911613"></a>DIST_UNPACK_B8</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p126261929164"><a name="p126261929164"></a><a name="p126261929164"></a>解压缩模式，按无符号整型u8加载VL/2长度数据，unpack到VL长度u16类型，中间位置补0</p>
<p id="p126261429060"><a name="p126261429060"></a><a name="p126261429060"></a>例，VL=256B：</p>
<p id="p96261329960"><a name="p96261329960"></a><a name="p96261329960"></a>src: [0x00, 0x01, 0x02, 0x03, ..., 0xFF, ...]</p>
<p id="p1662614291620"><a name="p1662614291620"></a><a name="p1662614291620"></a>dst:[0x0000, 0x0001, ..., 0x0007F]</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p10792175133011"><a name="p10792175133011"></a><a name="p10792175133011"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row832415311612"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p19626192912615"><a name="p19626192912615"></a><a name="p19626192912615"></a>DIST_UNPACK_B16</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p116266291268"><a name="p116266291268"></a><a name="p116266291268"></a>解压缩模式，按无符号整型u16加载VL/2长度数据，unpack到VL长度u32类型，中间位置补0</p>
<p id="p1062622915616"><a name="p1062622915616"></a><a name="p1062622915616"></a>例，VL=256B：</p>
<p id="p762682919620"><a name="p762682919620"></a><a name="p762682919620"></a>src: [0x0000, 0x0001, 0x0002, 0x0003, ..., 0x007F, ...]</p>
<p id="p262682911610"><a name="p262682911610"></a><a name="p262682911610"></a>dst:[0x00000000, 0x00000001, ..., 0x000003F]</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p11792351103019"><a name="p11792351103019"></a><a name="p11792351103019"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row1783657613"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p16265291966"><a name="p16265291966"></a><a name="p16265291966"></a>DIST_BLK</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p12626829468"><a name="p12626829468"></a><a name="p12626829468"></a>读取一个DataBlock（32B），并广播到VL</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p3792205163010"><a name="p3792205163010"></a><a name="p3792205163010"></a>32</p>
</td>
</tr>
<tr id="row16546017610"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p562612291565"><a name="p562612291565"></a><a name="p562612291565"></a>DIST_E2B_B16</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p1462618298617"><a name="p1462618298617"></a><a name="p1462618298617"></a>加载(VL/DataBlock)B的数据，并将每个元素（16bit）广播到一个DataBlock（32B）中。</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p11792851183018"><a name="p11792851183018"></a><a name="p11792851183018"></a>VL/16</p>
</td>
</tr>
<tr id="row820915585517"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p1626929967"><a name="p1626929967"></a><a name="p1626929967"></a>DIST_E2B_B32</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p7626122918614"><a name="p7626122918614"></a><a name="p7626122918614"></a>加载(VL/DataBlock)B的数据，并将每个元素（32bit）广播到一个DataBlock（32B）中。</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p879311510308"><a name="p879311510308"></a><a name="p879311510308"></a>VL/8</p>
</td>
</tr>
<tr id="row1251319565517"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p12626132910612"><a name="p12626132910612"></a><a name="p12626132910612"></a>DIST_UNPACK_B32</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p7626529264"><a name="p7626529264"></a><a name="p7626529264"></a>解压缩模式，按无符号整型u32加载VL/2长度数据，unpack到VL长度u64类型，中间位置补0</p>
<p id="p1762612293616"><a name="p1762612293616"></a><a name="p1762612293616"></a>例，VL=256B：</p>
<p id="p76273294610"><a name="p76273294610"></a><a name="p76273294610"></a>src: [0x00000000, 0x00000001, 0x00000002, ..., 0x000003F, ...]</p>
<p id="p16627142913612"><a name="p16627142913612"></a><a name="p16627142913612"></a>dst:[0x0000000000000000, 0x0000000000000001, ..., 0x000000000000001F]</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p979305123020"><a name="p979305123020"></a><a name="p979305123020"></a>min(32, VL/2)</p>
</td>
</tr>
<tr id="row7540135411520"><td class="cellrowborder" valign="top" width="12.920000000000002%" headers="mcps1.2.4.1.1 "><p id="p1462752911610"><a name="p1462752911610"></a><a name="p1462752911610"></a>DIST_UNPACK4_B8</p>
</td>
<td class="cellrowborder" valign="top" width="66.60000000000001%" headers="mcps1.2.4.1.2 "><p id="p562772917616"><a name="p562772917616"></a><a name="p562772917616"></a>解压缩模式，按无符号整型u8加载VL/4长度数据，unpack到VL长度u32类型，中间位置补0</p>
<p id="p36276291616"><a name="p36276291616"></a><a name="p36276291616"></a>例，VL=256B：</p>
<p id="p1462712291465"><a name="p1462712291465"></a><a name="p1462712291465"></a>src: [0x00, 0x01, 0x02, 0x03, ..., 0xFF, ...]</p>
<p id="p1962742910620"><a name="p1962742910620"></a><a name="p1962742910620"></a>dst:[0x00000000, 0x00000001, ..., 0x000003F]</p>
</td>
<td class="cellrowborder" valign="top" width="20.48%" headers="mcps1.2.4.1.3 "><p id="p87931251133018"><a name="p87931251133018"></a><a name="p87931251133018"></a>min(32, VL/4)</p>
</td>
</tr>
</tbody>
</table>

**表 2**  LoadDist模板参数说明（双搬入模式）

<a name="table20720517613"></a>
<table><thead align="left"><tr id="row1389513610"><th class="cellrowborder" valign="top" width="12.67%" id="mcps1.2.4.1.1"><p id="p18895113612"><a name="p18895113612"></a><a name="p18895113612"></a><strong id="b1588517611"><a name="b1588517611"></a><a name="b1588517611"></a>LoadDist</strong>取值</p>
</th>
<th class="cellrowborder" valign="top" width="66.77%" id="mcps1.2.4.1.2"><p id="p1286514615"><a name="p1286514615"></a><a name="p1286514615"></a><strong id="b782512613"><a name="b782512613"></a><a name="b782512613"></a>含义</strong></p>
</th>
<th class="cellrowborder" valign="top" width="20.560000000000002%" id="mcps1.2.4.1.3"><p id="p08195110611"><a name="p08195110611"></a><a name="p08195110611"></a><strong id="b6845118616"><a name="b6845118616"></a><a name="b6845118616"></a>搬运对齐约束（Byte）</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1784511268"><td class="cellrowborder" valign="top" width="12.67%" headers="mcps1.2.4.1.1 "><p id="p42259171675"><a name="p42259171675"></a><a name="p42259171675"></a>DIST_DINTLV_B8</p>
</td>
<td class="cellrowborder" valign="top" width="66.77%" headers="mcps1.2.4.1.2 "><p id="p52258171579"><a name="p52258171579"></a><a name="p52258171579"></a>双搬入模式，基于元素的交错搬运，从src中读取2*VL长度数据，将偶数索引的元素存入dst0，将奇数索引的元素存入dst1，数据类型为b8</p>
</td>
<td class="cellrowborder" valign="top" width="20.560000000000002%" headers="mcps1.2.4.1.3 "><p id="p142253171719"><a name="p142253171719"></a><a name="p142253171719"></a>32</p>
</td>
</tr>
<tr id="row285511260"><td class="cellrowborder" valign="top" width="12.67%" headers="mcps1.2.4.1.1 "><p id="p022521713716"><a name="p022521713716"></a><a name="p022521713716"></a>DIST_DINTLV_B16</p>
</td>
<td class="cellrowborder" valign="top" width="66.77%" headers="mcps1.2.4.1.2 "><p id="p122561713718"><a name="p122561713718"></a><a name="p122561713718"></a>双搬入模式，基于元素的交错搬运，从src中读取2*VL长度数据，将偶数索引的元素存入dst0，将奇数索引的元素存入dst1，数据类型为b16</p>
</td>
<td class="cellrowborder" valign="top" width="20.560000000000002%" headers="mcps1.2.4.1.3 "><p id="p9225817678"><a name="p9225817678"></a><a name="p9225817678"></a>32</p>
</td>
</tr>
<tr id="row15855119616"><td class="cellrowborder" valign="top" width="12.67%" headers="mcps1.2.4.1.1 "><p id="p8919264714"><a name="p8919264714"></a><a name="p8919264714"></a>DIST_DINTLV_B32</p>
</td>
<td class="cellrowborder" valign="top" width="66.77%" headers="mcps1.2.4.1.2 "><p id="p149142613720"><a name="p149142613720"></a><a name="p149142613720"></a>双搬入模式，基于元素的交错搬运，从src中读取2*VL长度数据，将偶数索引的元素存入dst0，将奇数索引的元素存入dst1，数据类型为b32</p>
</td>
<td class="cellrowborder" valign="top" width="20.560000000000002%" headers="mcps1.2.4.1.3 "><p id="p199117264718"><a name="p199117264718"></a><a name="p199117264718"></a>32</p>
</td>
</tr>
</tbody>
</table>

**表 3**  单搬入模式POST\_MODE\_NORMAL场景参数说明

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
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p12177205271010"><a name="p12177205271010"></a><a name="p12177205271010"></a>操作数数据类型，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row12812937171011"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p19177165291010"><a name="p19177165291010"></a><a name="p19177165291010"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p517811526102"><a name="p517811526102"></a><a name="p517811526102"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1736164825519"><a name="p1736164825519"></a><a name="p1736164825519"></a>搬运模式，LoadDist类型，具体的取值请参考<a href="#zh-cn_topic_0235751031_table33761356">表1</a>。</p>
</td>
</tr>
<tr id="row4845618151214"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1784518187125"><a name="p1784518187125"></a><a name="p1784518187125"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p188451018141218"><a name="p188451018141218"></a><a name="p188451018141218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1073619481557"><a name="p1073619481557"></a><a name="p1073619481557"></a><span id="ph13611131212578"><a name="ph13611131212578"></a><a name="ph13611131212578"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row94281746151014"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p517813529108"><a name="p517813529108"></a><a name="p517813529108"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1617815218101"><a name="p1617815218101"></a><a name="p1617815218101"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1817805214104"><a name="p1817805214104"></a><a name="p1817805214104"></a>目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row677112448101"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p101781252191016"><a name="p101781252191016"></a><a name="p101781252191016"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p117875217109"><a name="p117875217109"></a><a name="p117875217109"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p151781152101014"><a name="p151781152101014"></a><a name="p151781152101014"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  单搬入模式POST\_MODE\_UPDATE场景参数说明

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
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1259113482583"><a name="p1259113482583"></a><a name="p1259113482583"></a>搬运模式，LoadDist类型，具体的取值请参考<a href="#zh-cn_topic_0235751031_table33761356">表1</a>。</p>
</td>
</tr>
<tr id="row8678926201317"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p9678112614133"><a name="p9678112614133"></a><a name="p9678112614133"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p7678226111314"><a name="p7678226111314"></a><a name="p7678226111314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p478385355820"><a name="p478385355820"></a><a name="p478385355820"></a><span id="ph578317535589"><a name="ph578317535589"></a><a name="ph578317535589"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row1767892613133"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1167816264139"><a name="p1167816264139"></a><a name="p1167816264139"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p867816269131"><a name="p867816269131"></a><a name="p867816269131"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p167812262133"><a name="p167812262133"></a><a name="p167812262133"></a>目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row1867816265132"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p867882641312"><a name="p867882641312"></a><a name="p867882641312"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p17678182611318"><a name="p17678182611318"></a><a name="p17678182611318"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1267810260135"><a name="p1267810260135"></a><a name="p1267810260135"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row1367820264135"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p2242617191515"><a name="p2242617191515"></a><a name="p2242617191515"></a>postUpdateStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1567812263136"><a name="p1567812263136"></a><a name="p1567812263136"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p7678132641315"><a name="p7678132641315"></a><a name="p7678132641315"></a>实际搬运UB起始地址为srcAddr，搬运后执行地址更新：srcAddr +=  postUpdateStride。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  单搬入模式使用AddrReg存储偏移量场景参数说明

<a name="table153653331131"></a>
<table><thead align="left"><tr id="row1936513331318"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p1636619332034"><a name="p1636619332034"></a><a name="p1636619332034"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p17366103316319"><a name="p17366103316319"></a><a name="p17366103316319"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p9366193318317"><a name="p9366193318317"></a><a name="p9366193318317"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1636612331238"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1436613311319"><a name="p1436613311319"></a><a name="p1436613311319"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p123662033931"><a name="p123662033931"></a><a name="p123662033931"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p4366193319320"><a name="p4366193319320"></a><a name="p4366193319320"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row16366133314319"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p336615331530"><a name="p336615331530"></a><a name="p336615331530"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p336617330313"><a name="p336617330313"></a><a name="p336617330313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1853612259599"><a name="p1853612259599"></a><a name="p1853612259599"></a>搬运模式，LoadDist类型，具体的取值请参考<a href="#zh-cn_topic_0235751031_table33761356">表1</a>。</p>
</td>
</tr>
<tr id="row113661331310"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p4366113313312"><a name="p4366113313312"></a><a name="p4366113313312"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1236612339316"><a name="p1236612339316"></a><a name="p1236612339316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p14536192515598"><a name="p14536192515598"></a><a name="p14536192515598"></a><span id="ph853662585918"><a name="ph853662585918"></a><a name="ph853662585918"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row9366123313313"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p53667338313"><a name="p53667338313"></a><a name="p53667338313"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p436614331317"><a name="p436614331317"></a><a name="p436614331317"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p83661633334"><a name="p83661633334"></a><a name="p83661633334"></a>目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row2092420201016"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p183666331037"><a name="p183666331037"></a><a name="p183666331037"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p193661633030"><a name="p193661633030"></a><a name="p193661633030"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p93660331336"><a name="p93660331336"></a><a name="p93660331336"></a>实际搬运地址UB为srcAddr + offset。</p>
</td>
</tr>
<tr id="row73660331233"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p936693314314"><a name="p936693314314"></a><a name="p936693314314"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p2036617331137"><a name="p2036617331137"></a><a name="p2036617331137"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p103668331434"><a name="p103668331434"></a><a name="p103668331434"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
</tbody>
</table>

**表 6**  双搬入模式POST\_MODE\_NORMAL场景参数说明

<a name="table13414281139"></a>
<table><thead align="left"><tr id="row163413281130"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p16341102841315"><a name="p16341102841315"></a><a name="p16341102841315"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.03%" id="mcps1.2.4.1.2"><p id="p13341528181317"><a name="p13341528181317"></a><a name="p13341528181317"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.43%" id="mcps1.2.4.1.3"><p id="p634192871313"><a name="p634192871313"></a><a name="p634192871313"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row153421228191315"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1434232810137"><a name="p1434232810137"></a><a name="p1434232810137"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.03%" headers="mcps1.2.4.1.2 "><p id="p434213287136"><a name="p434213287136"></a><a name="p434213287136"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.43%" headers="mcps1.2.4.1.3 "><p id="p1934216284135"><a name="p1934216284135"></a><a name="p1934216284135"></a>模板参数，支持的数据类型为b8/b16/b32。</p>
</td>
</tr>
<tr id="row634262814131"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p43421328181312"><a name="p43421328181312"></a><a name="p43421328181312"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.03%" headers="mcps1.2.4.1.2 "><p id="p73421728141316"><a name="p73421728141316"></a><a name="p73421728141316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.43%" headers="mcps1.2.4.1.3 "><p id="p19235107145615"><a name="p19235107145615"></a><a name="p19235107145615"></a>搬运模式，LoadDist类型，具体的取值请参考<a href="#table20720517613">表2</a>。</p>
</td>
</tr>
<tr id="row14343828111314"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1343182811317"><a name="p1343182811317"></a><a name="p1343182811317"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.03%" headers="mcps1.2.4.1.2 "><p id="p8343202801316"><a name="p8343202801316"></a><a name="p8343202801316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.43%" headers="mcps1.2.4.1.3 "><p id="p1762418510413"><a name="p1762418510413"></a><a name="p1762418510413"></a><span id="ph562465113413"><a name="ph562465113413"></a><a name="ph562465113413"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row163439289132"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1334314289130"><a name="p1334314289130"></a><a name="p1334314289130"></a>dstReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.03%" headers="mcps1.2.4.1.2 "><p id="p1334362818138"><a name="p1334362818138"></a><a name="p1334362818138"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.43%" headers="mcps1.2.4.1.3 "><p id="p7343528131316"><a name="p7343528131316"></a><a name="p7343528131316"></a>第一个目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row1049493761810"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p83312042121812"><a name="p83312042121812"></a><a name="p83312042121812"></a>dstReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.03%" headers="mcps1.2.4.1.2 "><p id="p8331154220185"><a name="p8331154220185"></a><a name="p8331154220185"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.43%" headers="mcps1.2.4.1.3 "><p id="p11331154211186"><a name="p11331154211186"></a><a name="p11331154211186"></a>第二个目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row13343142851319"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1234372841319"><a name="p1234372841319"></a><a name="p1234372841319"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.03%" headers="mcps1.2.4.1.2 "><p id="p23431287135"><a name="p23431287135"></a><a name="p23431287135"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.43%" headers="mcps1.2.4.1.3 "><p id="p13343828181316"><a name="p13343828181316"></a><a name="p13343828181316"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
</tbody>
</table>

**表 7**  双搬入模式POST\_MODE\_UPDATE场景参数说明

<a name="table13703182931320"></a>
<table><thead align="left"><tr id="row12703229181316"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p17032029181318"><a name="p17032029181318"></a><a name="p17032029181318"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p5703132951314"><a name="p5703132951314"></a><a name="p5703132951314"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p5704102971316"><a name="p5704102971316"></a><a name="p5704102971316"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row18704132991313"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1070413295132"><a name="p1070413295132"></a><a name="p1070413295132"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p87041029151312"><a name="p87041029151312"></a><a name="p87041029151312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p19704152991312"><a name="p19704152991312"></a><a name="p19704152991312"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row13647514101911"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p17612111916"><a name="p17612111916"></a><a name="p17612111916"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p27612120193"><a name="p27612120193"></a><a name="p27612120193"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p12235127145610"><a name="p12235127145610"></a><a name="p12235127145610"></a>用于控制是否使能post update，<a href="PostLiteral.md">PostLiteral</a>类型。</p>
</td>
</tr>
<tr id="row1070442914131"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p3704192931312"><a name="p3704192931312"></a><a name="p3704192931312"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p67041029191315"><a name="p67041029191315"></a><a name="p67041029191315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p14964781357"><a name="p14964781357"></a><a name="p14964781357"></a>搬运模式，LoadDist类型，具体的取值请参考<a href="#table20720517613">表2</a>。</p>
</td>
</tr>
<tr id="row1370492931319"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1870432910131"><a name="p1870432910131"></a><a name="p1870432910131"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p470472910130"><a name="p470472910130"></a><a name="p470472910130"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1981417151556"><a name="p1981417151556"></a><a name="p1981417151556"></a><span id="ph78142158515"><a name="ph78142158515"></a><a name="ph78142158515"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row14704182915131"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p08228507199"><a name="p08228507199"></a><a name="p08228507199"></a>dstReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p0822145013197"><a name="p0822145013197"></a><a name="p0822145013197"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p982265091911"><a name="p982265091911"></a><a name="p982265091911"></a>第一个目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row20875164351920"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p128238500197"><a name="p128238500197"></a><a name="p128238500197"></a>dstReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p13823150171913"><a name="p13823150171913"></a><a name="p13823150171913"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1282355013194"><a name="p1282355013194"></a><a name="p1282355013194"></a>第二个目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row2704929191313"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p15704182911314"><a name="p15704182911314"></a><a name="p15704182911314"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p1070492991317"><a name="p1070492991317"></a><a name="p1070492991317"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p5704102911317"><a name="p5704102911317"></a><a name="p5704102911317"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row1870492951310"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p62261537204"><a name="p62261537204"></a><a name="p62261537204"></a>postUpdateStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p9227730207"><a name="p9227730207"></a><a name="p9227730207"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1722733162012"><a name="p1722733162012"></a><a name="p1722733162012"></a>实际搬运UB起始地址为srcAddr，搬运后执行地址更新：srcAddr +=  postUpdateStride。</p>
</td>
</tr>
</tbody>
</table>

**表 8**  双搬入模式使用AddrReg存储偏移量场景参数说明

<a name="table83748584510"></a>
<table><thead align="left"><tr id="row113755581056"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p2037514588510"><a name="p2037514588510"></a><a name="p2037514588510"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.040000000000001%" id="mcps1.2.4.1.2"><p id="p1737518581056"><a name="p1737518581056"></a><a name="p1737518581056"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41999999999999%" id="mcps1.2.4.1.3"><p id="p183755587515"><a name="p183755587515"></a><a name="p183755587515"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1937575819511"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p3375958253"><a name="p3375958253"></a><a name="p3375958253"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p173758586519"><a name="p173758586519"></a><a name="p173758586519"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1337575814511"><a name="p1337575814511"></a><a name="p1337575814511"></a>模板参数，支持的数据类型为b8/b16/b32。</p>
</td>
</tr>
<tr id="row11375258757"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p183759589511"><a name="p183759589511"></a><a name="p183759589511"></a>dist</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p23751558656"><a name="p23751558656"></a><a name="p23751558656"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p745419491658"><a name="p745419491658"></a><a name="p745419491658"></a>搬运模式，LoadDist类型，具体的取值请参考<a href="#table20720517613">表2</a>。</p>
</td>
</tr>
<tr id="row237517581556"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p53758581553"><a name="p53758581553"></a><a name="p53758581553"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p83755581950"><a name="p83755581950"></a><a name="p83755581950"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p1045520491359"><a name="p1045520491359"></a><a name="p1045520491359"></a><span id="ph84552491520"><a name="ph84552491520"></a><a name="ph84552491520"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
</td>
</tr>
<tr id="row1337515814517"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p13759581656"><a name="p13759581656"></a><a name="p13759581656"></a>dstReg0</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p183752058255"><a name="p183752058255"></a><a name="p183752058255"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p2037555819514"><a name="p2037555819514"></a><a name="p2037555819514"></a>第一个目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row93756582057"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1837555813515"><a name="p1837555813515"></a><a name="p1837555813515"></a>dstReg1</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p437518585513"><a name="p437518585513"></a><a name="p437518585513"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p123752584516"><a name="p123752584516"></a><a name="p123752584516"></a>第二个目的操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row1197755010916"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p133769582513"><a name="p133769582513"></a><a name="p133769582513"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p3376145819520"><a name="p3376145819520"></a><a name="p3376145819520"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p23761058154"><a name="p23761058154"></a><a name="p23761058154"></a>实际搬运地址UB为srcAddr + offset。</p>
</td>
</tr>
<tr id="row153751958855"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p63754581552"><a name="p63754581552"></a><a name="p63754581552"></a>srcAddr</p>
</td>
<td class="cellrowborder" valign="top" width="10.040000000000001%" headers="mcps1.2.4.1.2 "><p id="p103751581359"><a name="p103751581359"></a><a name="p103751581359"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41999999999999%" headers="mcps1.2.4.1.3 "><p id="p13375135816511"><a name="p13375135816511"></a><a name="p13375135816511"></a>源操作数在UB上的起始地址。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section1575141714439"></a>

无

## 约束说明<a name="section11585101304320"></a>

b64数据类型只支持LoadDist中的DIST\_NORM模式。

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
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B8>(srcReg0, srcReg1, srcAddr + i * oneRepeatSize);
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

