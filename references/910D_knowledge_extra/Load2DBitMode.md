# Load2DBitMode<a name="ZH-CN_TOPIC_0000002554424129"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

Load2D支持如下数据通路的搬运：

GM-\>A1; GM-\>B1; GM-\>A2; GM-\>B2;

A1-\>A2; B1-\>B2。

## 函数原型<a name="section620mcpsimp"></a>

```
template <TPosition Dst, TPosition Src, typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src,const Load2DBitModeParam& loadDataParam)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table07381635103112"></a>
<table><thead align="left"><tr id="row117393350314"><th class="cellrowborder" valign="top" width="16.55%" id="mcps1.2.3.1.1"><p id="p14739335193119"><a name="p14739335193119"></a><a name="p14739335193119"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.45%" id="mcps1.2.3.1.2"><p id="p8739203514314"><a name="p8739203514314"></a><a name="p8739203514314"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row18739935193119"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p173953516310"><a name="p173953516310"></a><a name="p173953516310"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p12739193516313"><a name="p12739193516313"></a><a name="p12739193516313"></a>源操作数和目的操作数的数据类型。</p>
<p id="p78851620162815"><a name="p78851620162815"></a><a name="p78851620162815"></a><span id="ph12885120152810"><a name="ph12885120152810"></a><a name="ph12885120152810"></a>Ascend 950PR/Ascend 950DT</span>，仅支持A1-&gt;A2、B1-&gt;B2，支持数据类型为：half/bfloat16_t/uint32_t/int32_t/float/uint8_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t</p>
</td>
</tr>
<tr id="row09851744154415"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p11985204464419"><a name="p11985204464419"></a><a name="p11985204464419"></a>Src</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p1498517449447"><a name="p1498517449447"></a><a name="p1498517449447"></a>源操作数存储的逻辑位置（TPosition），仅Load2DBitMode接口使用。</p>
</td>
</tr>
<tr id="row1130353815467"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p1230363812467"><a name="p1230363812467"></a><a name="p1230363812467"></a>Dst</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p3303153874616"><a name="p3303153874616"></a><a name="p3303153874616"></a>目的操作数存储的逻辑位置（TPosition），仅Load2DBitMode接口使用。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  通用参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.89168916891689%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.111111111111112%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.99719971997199%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p66101129164712"><a name="p66101129164712"></a><a name="p66101129164712"></a>目的操作数，类型为LocalTensor。</p>
<p id="p3610102994715"><a name="p3610102994715"></a><a name="p3610102994715"></a>数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下：</p>
<a name="ul76107290479"></a><a name="ul76107290479"></a><ul id="ul76107290479"><li>A2：ZZ格式/NZ格式；对应的分形大小为16 * (32B / sizeof(T))。</li><li>B2：ZN格式；对应的分形大小为 (32B / sizeof(T))  * 16。</li><li>A1/B1：无格式要求，一般情况下为NZ格式。NZ格式下，对应的分形大小为16 * (32B / sizeof(T))。</li></ul>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p165920401156"><a name="p165920401156"></a><a name="p165920401156"></a>源操作数，类型为LocalTensor或GlobalTensor。</p>
<p id="p192019400435"><a name="p192019400435"></a><a name="p192019400435"></a>数据类型需要与dst保持一致。</p>
</td>
</tr>
<tr id="row1767431631917"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p667418162198"><a name="p667418162198"></a><a name="p667418162198"></a>loadDataParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p11675191610195"><a name="p11675191610195"></a><a name="p11675191610195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p1667541617193"><a name="p1667541617193"></a><a name="p1667541617193"></a>LoadData参数结构体，类型为：</p>
<a name="ul207951119112217"></a><a name="ul207951119112217"></a><ul id="ul207951119112217"><li>Load2DBitModeParam，具体参考<a href="#table10539223195311">表3</a>。</li></ul>
<p id="p21811725744"><a name="p21811725744"></a><a name="p21811725744"></a>上述结构体参数定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Load2DBitModeParam类参数说明

<a name="table10539223195311"></a>
<table><thead align="left"><tr id="row05391923195317"><th class="cellrowborder" valign="top" width="18.459999999999997%" id="mcps1.2.3.1.1"><p id="p753962317532"><a name="p753962317532"></a><a name="p753962317532"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.54%" id="mcps1.2.3.1.2"><p id="p1653910236531"><a name="p1653910236531"></a><a name="p1653910236531"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row7539192395320"><td class="cellrowborder" valign="top" width="18.459999999999997%" headers="mcps1.2.3.1.1 "><p id="p11539192385319"><a name="p11539192385319"></a><a name="p11539192385319"></a>config0</p>
</td>
<td class="cellrowborder" valign="top" width="81.54%" headers="mcps1.2.3.1.2 "><p id="p453922355312"><a name="p453922355312"></a><a name="p453922355312"></a>uint64_t类型，与Load2DBitModeConfig0位域（bit-field）结构体类型参数config0BitMode组成联合体（union），初始化为0，可以使用类对象的GetConfig0()函数获取其值。</p>
</td>
</tr>
<tr id="row65398239539"><td class="cellrowborder" valign="top" width="18.459999999999997%" headers="mcps1.2.3.1.1 "><p id="p3539323195316"><a name="p3539323195316"></a><a name="p3539323195316"></a>config0BitMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.54%" headers="mcps1.2.3.1.2 "><p id="p1953902318537"><a name="p1953902318537"></a><a name="p1953902318537"></a>Load2DBitModeConfig0位域（bit-field）结构体类型，参数参考<a href="#table4109172132317">表4</a>，与config0组成联合体（union）。</p>
</td>
</tr>
<tr id="row35391323185314"><td class="cellrowborder" valign="top" width="18.459999999999997%" headers="mcps1.2.3.1.1 "><p id="p13539823125319"><a name="p13539823125319"></a><a name="p13539823125319"></a>config1</p>
</td>
<td class="cellrowborder" valign="top" width="81.54%" headers="mcps1.2.3.1.2 "><p id="p35391235531"><a name="p35391235531"></a><a name="p35391235531"></a>uint64_t类型，与Load2DBitModeConfig1位域（bit-field）结构体类型参数config1BitMode组成联合体（union），初始化为0，可以使用类对象的GetConfig1()函数获取其值。</p>
</td>
</tr>
<tr id="row8292135420158"><td class="cellrowborder" valign="top" width="18.459999999999997%" headers="mcps1.2.3.1.1 "><p id="p19292145421517"><a name="p19292145421517"></a><a name="p19292145421517"></a>config1BitMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.54%" headers="mcps1.2.3.1.2 "><p id="p152924545158"><a name="p152924545158"></a><a name="p152924545158"></a>Load2DBitModeConfig1位域（bit-field）结构体类型，参数参考<a href="#table122891852142311">表5</a>，与config1组成联合体（union）。</p>
</td>
</tr>
<tr id="row111193507272"><td class="cellrowborder" valign="top" width="18.459999999999997%" headers="mcps1.2.3.1.1 "><p id="p8120450182715"><a name="p8120450182715"></a><a name="p8120450182715"></a>ifTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="81.54%" headers="mcps1.2.3.1.2 "><p id="p1745211862812"><a name="p1745211862812"></a><a name="p1745211862812"></a>是否启用转置功能，对每个分形矩阵进行转置，默认为false。含义与LoadData2DParamsV2结构体中的同名参数含义相同，具体参考<a href="Load2D.md#table49630346128">表4</a>。</p>
<a name="ul8452918152813"></a><a name="ul8452918152813"></a><ul id="ul8452918152813"><li>true：启用</li><li>false：不启用</li></ul>
<p id="p1745221818285"><a name="p1745221818285"></a><a name="p1745221818285"></a>注意：只有A1-&gt;A2和B1-&gt;B2通路才能使能转置。使能转置功能时，支持的数据类型约束如下：</p>
<p id="p84521718112811"><a name="p84521718112811"></a><a name="p84521718112811"></a>源操作数、目的操作数支持b4、b8、b16、b32数据类型。</p>
</td>
</tr>
</tbody>
</table>

Load2DBitModeParam类参数设计思想说明：

联合体（union）是一种特殊的数据结构，允许在相同的内存位置存储不同的数据类型。union的所有成员共享同一块内存空间，大小由最大成员决定，同一时间只能使用一个成员。

位域（bit-field）是一种特殊的类成员，允许精确控制结构体中成员变量所占用的内存位数。结构体中成员变量从上到下对应内存中从低位到高位。

Load2DBitModeParam类使用union与bit-field方法，采用bit位表达参数类型，使用bit-field结构体自动处理入参的bit位数，并利用union的特性实现多参数融合传递，仅需传递一个入参即可包含全部所需信息，对应底层接口仅需要接收一个参数。同时，当需要修改参数中某一bit位的值时，仅需要通过循环和位运算即可实现，不需要重新传入参数，减少了scalar计算，实现性能提升。

Load2DBitModeParam类可以直接使用LoadData2DParamsV2结构体类型对象初始化：

```
LoadData2DParamsV2 loadDataParams;
loadDataParams.mStartPosition = 0;
loadDataParams.kStartPosition = 0;
loadDataParams.mStep = xxx;
loadDataParams.kStep = xxx;
loadDataParams.srcStride = xxx;
loadDataParams.dstStride = xxx;
loadDataParams.sid = 0;
loadDataParams.ifTranspose = false;
Load2DBitModeParam params(loadDataParams);  // 直接使用LoadData2DParamsV2结构体类型对象初始化
```

也可以使用各参数的Set函数修改参数值，并且由于使用了联合体，还可以对congfig0和config1直接进行逐bit位修改来修改参数。

**表 4**  Load2DBitModeConfig0结构体参数说明

<a name="table4109172132317"></a>
<table><thead align="left"><tr id="row1610914216237"><th class="cellrowborder" valign="top" width="18.42%" id="mcps1.2.3.1.1"><p id="p111096219231"><a name="p111096219231"></a><a name="p111096219231"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.58%" id="mcps1.2.3.1.2"><p id="p31099215237"><a name="p31099215237"></a><a name="p31099215237"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row181099215230"><td class="cellrowborder" valign="top" width="18.42%" headers="mcps1.2.3.1.1 "><p id="p11109112111239"><a name="p11109112111239"></a><a name="p11109112111239"></a>mStartPosition</p>
</td>
<td class="cellrowborder" valign="top" width="81.58%" headers="mcps1.2.3.1.2 "><p id="p47917374590"><a name="p47917374590"></a><a name="p47917374590"></a>以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16个元素。</p>
<p id="p1010911217232"><a name="p1010911217232"></a><a name="p1010911217232"></a>该参数是位域结构体的最低位参数，占用16bit，可以使用Load2DBitModeParam类对象的SetMStartPosition()函数设置其值，使用GetMStartPosition()函数获取其值，具体参考<a href="#table1123714295457">表6</a>。</p>
</td>
</tr>
<tr id="row18109162115235"><td class="cellrowborder" valign="top" width="18.42%" headers="mcps1.2.3.1.1 "><p id="p1109821122316"><a name="p1109821122316"></a><a name="p1109821122316"></a>kStartPosition</p>
</td>
<td class="cellrowborder" valign="top" width="81.58%" headers="mcps1.2.3.1.2 "><p id="p1410982182310"><a name="p1410982182310"></a><a name="p1410982182310"></a>以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B。</p>
<p id="p4494144613013"><a name="p4494144613013"></a><a name="p4494144613013"></a>该参数是位域结构体的第二低位参数，占用16bit，可以使用Load2DBitModeParam类对象的SetKStartPosition()函数设置其值，使用GetKStartPosition()函数获取其值，具体参考<a href="#table1123714295457">表6</a>。</p>
</td>
</tr>
<tr id="row13109122117234"><td class="cellrowborder" valign="top" width="18.42%" headers="mcps1.2.3.1.1 "><p id="p14109132114232"><a name="p14109132114232"></a><a name="p14109132114232"></a>mStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.58%" headers="mcps1.2.3.1.2 "><p id="p1126134912618"><a name="p1126134912618"></a><a name="p1126134912618"></a>以M*K矩阵为例，源矩阵M轴方向搬运长度，单位为16 element。取值范围：mStep∈[0, 255]。</p>
<p id="p312674972614"><a name="p312674972614"></a><a name="p312674972614"></a>通过ifTranspose参数启用转置功能时，mStep除需满足 [0, 255]的取值范围外，还需满足以下额外约束：</p>
<a name="ul1312674910269"></a><a name="ul1312674910269"></a><ul id="ul1312674910269"><li>当数据类型为b4时，mStep必须是4的倍数；</li><li>当数据类型为b8时，mStep必须是2的倍数；</li><li>当数据类型为b16时，mStep必须是1的倍数；</li><li>当数据类型为b32时，mStep无额外约束。</li></ul>
<p id="p480011560020"><a name="p480011560020"></a><a name="p480011560020"></a>该参数是位域结构体的第三低位参数，占用8bit，可以使用Load2DBitModeParam类对象的SetMStep()函数设置其值，使用GetMStep()函数获取其值，具体参考<a href="#table1123714295457">表6</a>。</p>
</td>
</tr>
<tr id="row12110221192316"><td class="cellrowborder" valign="top" width="18.42%" headers="mcps1.2.3.1.1 "><p id="p6110721182318"><a name="p6110721182318"></a><a name="p6110721182318"></a>kStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.58%" headers="mcps1.2.3.1.2 "><p id="p16755139142712"><a name="p16755139142712"></a><a name="p16755139142712"></a>以M*K矩阵为例，源矩阵K轴方向搬运长度，单位为32B。取值范围：kStep∈[0, 255]。</p>
<p id="p1375518982711"><a name="p1375518982711"></a><a name="p1375518982711"></a>通过ifTranspose参数启用转置功能时，kStep除需满足[0,255]的取值范围外，还需满足以下额外约束：</p>
<a name="ul8755119152715"></a><a name="ul8755119152715"></a><ul id="ul8755119152715"><li>当数据类型为b4、b8或b16时，kStep没有额外约束；</li><li>当数据类型为b32时，kStep必须是2的倍数。</li></ul>
<p id="p274013117111"><a name="p274013117111"></a><a name="p274013117111"></a>该参数是位域结构体的最高位参数，占用8bit，可以使用Load2DBitModeParam类对象的SetKStep()函数设置其值，使用GetKStep()函数获取其值，具体参考<a href="#table1123714295457">表6</a>。</p>
</td>
</tr>
</tbody>
</table>

Load2DBitModeConfig0结构体参数的含义与LoadData2DParamsV2结构体中的同名参数含义相同，具体参考[表4](Load2D.md#table49630346128)。

**表 5**  Load2DBitModeConfig1结构体参数说明

<a name="table122891852142311"></a>
<table><thead align="left"><tr id="row72891452172315"><th class="cellrowborder" valign="top" width="18.2%" id="mcps1.2.3.1.1"><p id="p7289152152311"><a name="p7289152152311"></a><a name="p7289152152311"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.8%" id="mcps1.2.3.1.2"><p id="p19289145215239"><a name="p19289145215239"></a><a name="p19289145215239"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row122909522234"><td class="cellrowborder" valign="top" width="18.2%" headers="mcps1.2.3.1.1 "><p id="p729016522233"><a name="p729016522233"></a><a name="p729016522233"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.8%" headers="mcps1.2.3.1.2 "><p id="p132901152162313"><a name="p132901152162313"></a><a name="p132901152162313"></a>以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B。</p>
<p id="p85201071664"><a name="p85201071664"></a><a name="p85201071664"></a>该参数是位域结构体的最低位参数，占用16bit，可以使用Load2DBitModeParam类对象的SetSrcStride()函数设置其值，使用GetSrcStride()函数获取其值，具体参考<a href="#table1123714295457">表6</a>。</p>
</td>
</tr>
<tr id="row19290125292318"><td class="cellrowborder" valign="top" width="18.2%" headers="mcps1.2.3.1.1 "><p id="p1829025232317"><a name="p1829025232317"></a><a name="p1829025232317"></a>dstStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.8%" headers="mcps1.2.3.1.2 "><p id="p52903521232"><a name="p52903521232"></a><a name="p52903521232"></a>以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B。</p>
<p id="p125064914617"><a name="p125064914617"></a><a name="p125064914617"></a>该参数是位域结构体的最高位参数，占用16bit，可以使用Load2DBitModeParam类对象的SetDstStride()函数设置其值，使用GetDstStride()函数获取其值，具体参考<a href="#table1123714295457">表6</a>。</p>
</td>
</tr>
</tbody>
</table>

Load2DBitModeConfig1结构体参数的含义与LoadData2DParamsV2结构体中的同名参数含义相同，具体参考[表4](Load2D.md#table49630346128)。

**表 6**  Load2DBitModeParam类成员函数说明

<a name="table1123714295457"></a>
<table><thead align="left"><tr id="row32383296458"><th class="cellrowborder" valign="top" width="18.22%" id="mcps1.2.3.1.1"><p id="p723882974510"><a name="p723882974510"></a><a name="p723882974510"></a>函数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.78%" id="mcps1.2.3.1.2"><p id="p17238829174516"><a name="p17238829174516"></a><a name="p17238829174516"></a>功能</p>
</th>
</tr>
</thead>
<tbody><tr id="row52381029154513"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p15238529144515"><a name="p15238529144515"></a><a name="p15238529144515"></a>void SetMStartPosition(uint32_t mStartPosition_)</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p18238102912458"><a name="p18238102912458"></a><a name="p18238102912458"></a>将Load2DBitModeConfig0结构体参数mStartPosition的值设置为mStartPosition_。</p>
</td>
</tr>
<tr id="row523832924517"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p1223816293458"><a name="p1223816293458"></a><a name="p1223816293458"></a>void SetKStartPosition(uint32_t kStartPosition_)</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p9238929104511"><a name="p9238929104511"></a><a name="p9238929104511"></a>将Load2DBitModeConfig0结构体参数kStartPosition的值设置为kStartPosition_。</p>
</td>
</tr>
<tr id="row132381229164514"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p923832917458"><a name="p923832917458"></a><a name="p923832917458"></a>void SetMStep(uint16_t mStep_)</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p12381529114519"><a name="p12381529114519"></a><a name="p12381529114519"></a>将Load2DBitModeConfig0结构体参数mStep的值设置为mStep_。</p>
</td>
</tr>
<tr id="row132381229164517"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p1023892918453"><a name="p1023892918453"></a><a name="p1023892918453"></a>void SetKStep(uint16_t kStep_)</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p6238112915452"><a name="p6238112915452"></a><a name="p6238112915452"></a>将Load2DBitModeConfig0结构体参数kStep的值设置为kStep_。</p>
</td>
</tr>
<tr id="row4238172910453"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p123811291453"><a name="p123811291453"></a><a name="p123811291453"></a>void SetSrcStride(int32_t srcStride_)</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p823872924510"><a name="p823872924510"></a><a name="p823872924510"></a>将Load2DBitModeConfig1结构体参数srcStride的值设置为srcStride_。</p>
</td>
</tr>
<tr id="row32381329134514"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p142381029154515"><a name="p142381029154515"></a><a name="p142381029154515"></a>void SetDstStride(uint16_t dstStride_)</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p2238172913459"><a name="p2238172913459"></a><a name="p2238172913459"></a>将Load2DBitModeConfig1结构体参数dstStride的值设置为dstStride_。</p>
</td>
</tr>
<tr id="row4238112913458"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p924013299459"><a name="p924013299459"></a><a name="p924013299459"></a>uint32_t GetMStartPosition() const</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p18240132924513"><a name="p18240132924513"></a><a name="p18240132924513"></a>获取Load2DBitModeConfig0结构体参数mStartPosition的值。</p>
</td>
</tr>
<tr id="row5161194194518"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p9161941114511"><a name="p9161941114511"></a><a name="p9161941114511"></a>uint32_t GetKStartPosition() const</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p191043325118"><a name="p191043325118"></a><a name="p191043325118"></a>获取Load2DBitModeConfig0结构体参数kStartPosition的值。</p>
</td>
</tr>
<tr id="row8284446144514"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p18284174614515"><a name="p18284174614515"></a><a name="p18284174614515"></a>uint16_t GetMStep() const</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p6940113212112"><a name="p6940113212112"></a><a name="p6940113212112"></a>获取Load2DBitModeConfig0结构体参数mStep的值。</p>
</td>
</tr>
<tr id="row112625144511"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p11261851194514"><a name="p11261851194514"></a><a name="p11261851194514"></a>uint16_t GetKStep() const</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p1875919332113"><a name="p1875919332113"></a><a name="p1875919332113"></a>获取Load2DBitModeConfig0结构体参数kStep的值。</p>
</td>
</tr>
<tr id="row255115574514"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p185595564518"><a name="p185595564518"></a><a name="p185595564518"></a>int32_t GetSrcStride() const</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p354914341210"><a name="p354914341210"></a><a name="p354914341210"></a>获取Load2DBitModeConfig1结构体参数srcStride的值。</p>
</td>
</tr>
<tr id="row12705744616"><td class="cellrowborder" valign="top" width="18.22%" headers="mcps1.2.3.1.1 "><p id="p27037124616"><a name="p27037124616"></a><a name="p27037124616"></a>uint16_t GetDstStride() const</p>
</td>
<td class="cellrowborder" valign="top" width="81.78%" headers="mcps1.2.3.1.2 "><p id="p44016351918"><a name="p44016351918"></a><a name="p44016351918"></a>获取Load2DBitModeConfig1结构体参数dstStride的值。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section6461234123118"></a>

```
#include "kernel_operator.h"
uint16_t C1 = 2;
uint16_t H = 4, W = 4;
uint8_t Kh = 2, Kw = 2;
uint16_t Cout = 16;
uint16_t C0 = 16;
uint8_t dilationH = 2, dilationW = 2;
uint8_t padTop = 1, padBottom = 1, padLeft = 1, padRight = 1;
uint8_t strideH = 1, strideW = 1;
uint16_t coutBlocks, ho, wo, howo, howoRound;
uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, dstSize, dstCO1Size;
uint8_t padList[4] = {padLeft, padRight, padTop, padBottom};
featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);

fmRepeat = featureMapA2Size / (16 * C0);

LoadData2DParamsV2 param = { padList, H, W, 0, 0, 0, -1, -1, strideW, strideH, Kw, Kh, dilationW, dilationH, 1, 0, fmRepeat, 0, (half)(0)};
Load2DBitModeParam paramBitMode(param); 

AscendC::LocalTensor<half> featureMapA1 = inQueueFmA1.DeQue<half>();
AscendC::LocalTensor<half> featureMapA2 = inQueueFmA2.AllocTensor<half>();
AscendC::LoadData<A2, A1, half>(featureMapA2, featureMapA1, paramBitMode);
inQueueFmA2.EnQue<half>(featureMapA2);
inQueueFmA1.FreeTensor(featureMapA1);
```

