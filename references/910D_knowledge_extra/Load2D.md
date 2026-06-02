# Load2D<a name="ZH-CN_TOPIC_0000002554424057"></a>

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

-   Load2D接口

    ```
    template <typename T>
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData2DParams& loadDataParams)
    template <typename T> 
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const LoadData2DParams& loadDataParams)
    ```

-   Load2Dv2接口

    ```
    template <typename T>
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src,const LoadData2DParamsV2& loadDataParam)
    template <typename T>
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const GlobalTensor<T>& src,const LoadData2DParamsV2& loadDataParam)
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
<a name="ul773801764017"></a><a name="ul773801764017"></a><ul id="ul773801764017"><li><strong id="b183431082710"><a name="b183431082710"></a><a name="b183431082710"></a>Load2D接口</strong><p id="p596682044015"><a name="p596682044015"></a><a name="p596682044015"></a><span id="ph433161034314"><a name="ph433161034314"></a><a name="ph433161034314"></a>Ascend 950PR/Ascend 950DT</span>，支持数据类型为：uint8_t/int8_t/uint16_t/int16_t/half/bfloat16_t/uint32_t/int32_t/float，仅支持如下数据通路: GM-&gt;A1; GM-&gt;B1; A1-&gt;A2; B1-&gt;B2。</p>
</li><li><strong id="b146611871415"><a name="b146611871415"></a><a name="b146611871415"></a>Load2Dv2接口</strong><p id="p1661635611516"><a name="p1661635611516"></a><a name="p1661635611516"></a><span id="ph166169562517"><a name="ph166169562517"></a><a name="ph166169562517"></a>Ascend 950PR/Ascend 950DT</span>，</p>
<a name="ul1847154014371"></a><a name="ul1847154014371"></a><ul id="ul1847154014371"><li>GM-&gt;A1、GM-&gt;B1时，支持数据类型为：int8_t/uint8_t/fp4x2_e2m1_t/fp4x2_e1m2_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/half/bfloat16_t/int32_t/uint32_t/float</li><li>A1-&gt;A2、B1-&gt;B2时，支持数据类型为：<p id="p1188114121312"><a name="p1188114121312"></a><a name="p1188114121312"></a>int8_t/uint8_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/half/bfloat16_t/int32_t/uint32_t/float</p>
</li></ul>
</li></ul>
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
<a name="ul207951119112217"></a><a name="ul207951119112217"></a><ul id="ul207951119112217"><li>LoadData2DParams，具体参考<a href="#table8955841508">表3</a>。</li><li>LoadData2DParamsV2，具体参考<a href="#table49630346128">表4</a>。</li></ul>
<p id="p21811725744"><a name="p21811725744"></a><a name="p21811725744"></a>上述结构体参数定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  LoadData2DParams结构体内参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1855384918180"><a name="p1855384918180"></a><a name="p1855384918180"></a>startIndex</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p35535499185"><a name="p35535499185"></a><a name="p35535499185"></a>分形矩阵ID，说明搬运起始位置为源操作数中第几个分形（0为源操作数中第1个分形矩阵）。取值范围：startIndex∈[0, 65535] 。单位：512B。默认为0。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p955315493182"><a name="p955315493182"></a><a name="p955315493182"></a>repeatTimes</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p15553949101816"><a name="p15553949101816"></a><a name="p15553949101816"></a>迭代次数，每个迭代可以处理512B数据。取值范围：repeatTimes∈[1, 255]。</p>
</td>
</tr>
<tr id="row11771625161812"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p055374920185"><a name="p055374920185"></a><a name="p055374920185"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p18553154931812"><a name="p18553154931812"></a><a name="p18553154931812"></a>相邻迭代间，源操作数前一个分形与后一个分形起始地址的间隔，单位：512B。取值范围：src_stride∈[0, 65535]。默认为0。</p>
</td>
</tr>
<tr id="row93311227171815"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1555320499185"><a name="p1555320499185"></a><a name="p1555320499185"></a>sid</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1755334901812"><a name="p1755334901812"></a><a name="p1755334901812"></a>预留参数，配置为0即可。</p>
</td>
</tr>
<tr id="row1321772919185"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p125531449181816"><a name="p125531449181816"></a><a name="p125531449181816"></a>dstGap</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1755412492183"><a name="p1755412492183"></a><a name="p1755412492183"></a>相邻迭代间，目的操作数前一个分形结束地址与后一个分形起始地址的间隔，单位：512B。取值范围：dstGap∈[0, 65535]。默认为0。</p>
</td>
</tr>
<tr id="row16697631171819"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1555415492180"><a name="p1555415492180"></a><a name="p1555415492180"></a>ifTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1955414941814"><a name="p1955414941814"></a><a name="p1955414941814"></a>是否启用转置功能，对每个分形矩阵进行转置，默认为false:</p>
<a name="ul7554154914184"></a><a name="ul7554154914184"></a><ul id="ul7554154914184"><li>true：启用</li><li>false：不启用</li></ul>
<p id="p15554134918181"><a name="p15554134918181"></a><a name="p15554134918181"></a>注意：只有A1-&gt;A2和B1-&gt;B2通路才能使能转置，使能转置功能时，源操作数、目的操作数仅支持uint16_t/int16_t/half数据类型。</p>
</td>
</tr>
<tr id="row497391184"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p16554134921816"><a name="p16554134921816"></a><a name="p16554134921816"></a>addrMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p11525142812251"><a name="p11525142812251"></a><a name="p11525142812251"></a>控制地址更新方式，默认为false：</p>
<a name="ul75591732112515"></a><a name="ul75591732112515"></a><ul id="ul75591732112515"><li>true：递减，每次迭代在前一个地址的基础上减去srcStride。</li><li>false：递增，每次迭代在前一个地址的基础上加上srcStride。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 4**  LoadData2DParamsV2结构体内参数说明

<a name="table49630346128"></a>
<table><thead align="left"><tr id="row1963934151219"><th class="cellrowborder" valign="top" width="18.37%" id="mcps1.2.3.1.1"><p id="p696313340124"><a name="p696313340124"></a><a name="p696313340124"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.63%" id="mcps1.2.3.1.2"><p id="p149631734111213"><a name="p149631734111213"></a><a name="p149631734111213"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row189631234141219"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p796333412124"><a name="p796333412124"></a><a name="p796333412124"></a>mStartPosition</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p149638341126"><a name="p149638341126"></a><a name="p149638341126"></a>以M*K矩阵为例，源矩阵M轴方向的起始位置，单位为16个元素。</p>
</td>
</tr>
<tr id="row49631734201218"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p3963123451211"><a name="p3963123451211"></a><a name="p3963123451211"></a>kStartPosition</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p13963183421215"><a name="p13963183421215"></a><a name="p13963183421215"></a>以M*K矩阵为例，源矩阵K轴方向的起始位置，单位为32B。</p>
</td>
</tr>
<tr id="row79631348122"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p11964123461216"><a name="p11964123461216"></a><a name="p11964123461216"></a>mStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p1396433451215"><a name="p1396433451215"></a><a name="p1396433451215"></a>以M*K矩阵为例，源矩阵M轴方向搬运长度，单位为16 element。取值范围：mStep∈[0, 255]。</p>
<p id="p565931111316"><a name="p565931111316"></a><a name="p565931111316"></a>通过ifTranspose参数启用转置功能时，mStep除需满足 [0, 255]的取值范围外，还需满足以下额外约束：</p>
<a name="ul1365961151310"></a><a name="ul1365961151310"></a><ul id="ul1365961151310"><li>当数据类型为b4时，mStep必须是4的倍数；</li><li>当数据类型为b8时，mStep必须是2的倍数；</li><li>当数据类型为b16时，mStep必须是1的倍数；</li><li>当数据类型为b32时，mStep无额外约束。</li></ul>
</td>
</tr>
<tr id="row11964103461217"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p159641934121217"><a name="p159641934121217"></a><a name="p159641934121217"></a>kStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p242413118297"><a name="p242413118297"></a><a name="p242413118297"></a>以M*K矩阵为例，源矩阵K轴方向搬运长度，单位为32B。取值范围：kStep∈[0, 255]。</p>
<p id="p054825521415"><a name="p054825521415"></a><a name="p054825521415"></a>通过ifTranspose参数启用转置功能时，kStep除需满足[0,255]的取值范围外，还需满足以下额外约束：</p>
<a name="ul56872534221"></a><a name="ul56872534221"></a><ul id="ul56872534221"><li>当数据类型为b4、b8或b16时，kStep没有额外约束；</li><li>当数据类型为b32时，kStep必须是2的倍数。</li></ul>
</td>
</tr>
<tr id="row12964163416120"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p9964734161213"><a name="p9964734161213"></a><a name="p9964734161213"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p18964143411127"><a name="p18964143411127"></a><a name="p18964143411127"></a>以M*K矩阵为例，源矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B。</p>
</td>
</tr>
<tr id="row139641134201218"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p119641634111212"><a name="p119641634111212"></a><a name="p119641634111212"></a>dstStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p47271817135613"><a name="p47271817135613"></a><a name="p47271817135613"></a>以M*K矩阵为例，目标矩阵K方向前一个分形起始地址与后一个分形起始地址的间隔，单位：512B。</p>
</td>
</tr>
<tr id="row1496423417127"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p179641534151210"><a name="p179641534151210"></a><a name="p179641534151210"></a>ifTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p384571192517"><a name="p384571192517"></a><a name="p384571192517"></a>是否启用转置功能，对每个分形矩阵进行转置，默认为false。</p>
<a name="ul1084512113258"></a><a name="ul1084512113258"></a><ul id="ul1084512113258"><li>true：启用</li><li>false：不启用</li></ul>
<p id="p816810169287"><a name="p816810169287"></a><a name="p816810169287"></a>注意：只有A1-&gt;A2和B1-&gt;B2通路才能使能转置。使能转置功能时，支持的数据类型约束如下：</p>
<p id="p208454112251"><a name="p208454112251"></a><a name="p208454112251"></a></p>
<p id="p332283913013"><a name="p332283913013"></a><a name="p332283913013"></a>对于<span id="ph136962433110"><a name="ph136962433110"></a><a name="ph136962433110"></a>Ascend 950PR/Ascend 950DT</span>，源操作数、目的操作数支持b4、b8、b16、b32数据类型。</p>
</td>
</tr>
<tr id="row88461040162410"><td class="cellrowborder" valign="top" width="18.37%" headers="mcps1.2.3.1.1 "><p id="p7846104042412"><a name="p7846104042412"></a><a name="p7846104042412"></a>sid</p>
</td>
<td class="cellrowborder" valign="top" width="81.63%" headers="mcps1.2.3.1.2 "><p id="p884619409244"><a name="p884619409244"></a><a name="p884619409244"></a>预留参数，配置为0即可。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 返回值说明<a name="section640mcpsimp"></a>

无

## 调用示例<a name="section6461234123118"></a>

完整使用样例请参见[LoadData样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/load_data)。

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

AscendC::LocalTensor<half> featureMapA1 = inQueueFmA1.DeQue<half>();
AscendC::LocalTensor<half> featureMapA2 = inQueueFmA2.AllocTensor<half>();

AscendC::LoadData<A2, A1, half>(featureMapA2, featureMapA1, 
{ padList, H, W, 0, 0, 0, -1, -1, strideW, strideH, Kw, Kh, dilationW, dilationH, 1, 0, fmRepeat, 0, (half)(0)});

LoadData2DParamsV2 param = { padList, H, W, 0, 0, 0, -1, -1, strideW, strideH, Kw, Kh, dilationW, dilationH, 1, 0, fmRepeat, 0, (half)(0)};
Load2DBitModeParam paramBitMode(param); 
AscendC::LoadData<A2, A1, half>(featureMapA2, featureMapA1, paramBitMode);

inQueueFmA2.EnQue<half>(featureMapA2);
inQueueFmA1.FreeTensor(featureMapA1);
```

