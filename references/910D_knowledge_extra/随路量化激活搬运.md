# 随路量化激活搬运<a name="ZH-CN_TOPIC_0000002523343624"></a>

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

## 功能说明<a name="section5649901427"></a>

支持在数据搬运过程中进行量化和Relu激活等操作，同时支持Local Memory到Global Memory通路NZ到ND格式的转换。

## 函数原型<a name="section127451128224"></a>

-   Local Memory -\> Global Memory，支持量化和Relu激活等操作，同时支持NZ到ND格式的转换

    ```
    template <typename T, typename U>
    __aicore__ inline void DataCopy(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams)
    ```

-   Local Memory -\> Local Memory，支持量化和Relu激活等操作

    ```
    template <typename T, typename U>
    __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<U>& src, const DataCopyCO12DstParams& intriParams)
    ```

> **说明：** 
>各原型支持的具体数据通路和数据类型，请参考[支持的通路和数据类型](#section2614932173011)。

## 参数说明<a name="section126288513314"></a>

**表 1**  模板参数说明

<a name="table49614585413"></a>
<table><thead align="left"><tr id="row996115820412"><th class="cellrowborder" valign="top" width="14.729999999999999%" id="mcps1.2.3.1.1"><p id="p10961458104110"><a name="p10961458104110"></a><a name="p10961458104110"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.27%" id="mcps1.2.3.1.2"><p id="p6961155817415"><a name="p6961155817415"></a><a name="p6961155817415"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row4961205811416"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p29619586417"><a name="p29619586417"></a><a name="p29619586417"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p996110583417"><a name="p996110583417"></a><a name="p996110583417"></a>目的操作数的数据类型。支持的数据类型请参考<a href="#section2614932173011">支持的通路和数据类型</a>。</p>
</td>
</tr>
<tr id="row63251529183414"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="p732512919348"><a name="p732512919348"></a><a name="p732512919348"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="p14325142915349"><a name="p14325142915349"></a><a name="p14325142915349"></a>源操作数的数据类型。支持的数据类型请参考<a href="#section2614932173011">支持的通路和数据类型</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table917mcpsimp"></a>
<table><thead align="left"><tr id="row923mcpsimp"><th class="cellrowborder" valign="top" width="15.02%" id="mcps1.2.4.1.1"><p id="p925mcpsimp"><a name="p925mcpsimp"></a><a name="p925mcpsimp"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="9.86%" id="mcps1.2.4.1.2"><p id="p927mcpsimp"><a name="p927mcpsimp"></a><a name="p927mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.12%" id="mcps1.2.4.1.3"><p id="p929mcpsimp"><a name="p929mcpsimp"></a><a name="p929mcpsimp"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row930mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p2925016172518"><a name="p2925016172518"></a><a name="p2925016172518"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p199251416112517"><a name="p199251416112517"></a><a name="p199251416112517"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p37551021906"><a name="p37551021906"></a><a name="p37551021906"></a>目的操作数，类型为LocalTensor或GlobalTensor。</p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p2055414382913"><a name="p2055414382913"></a><a name="p2055414382913"></a>源操作数，类型为LocalTensor。</p>
</td>
</tr>
<tr id="row997554013220"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p13976540132215"><a name="p13976540132215"></a><a name="p13976540132215"></a>intriParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.4.1.2 "><p id="p139761340102213"><a name="p139761340102213"></a><a name="p139761340102213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.12%" headers="mcps1.2.4.1.3 "><p id="p99761407226"><a name="p99761407226"></a><a name="p99761407226"></a>搬运参数，类型为<a href="#table35908519282">DataCopyCO12DstParams</a>。</p>
<p id="p58852119618"><a name="p58852119618"></a><a name="p58852119618"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_data_copy.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  DataCopyCO12DstParams结构体参数定义（C0取值：一般情况下，C0 = 16；使能channelSplit（channel切分）时，C0 = 8）

<a name="table35908519282"></a>
<table><thead align="left"><tr id="row558675117280"><th class="cellrowborder" valign="top" width="18.44%" id="mcps1.2.3.1.1"><p id="p258615172811"><a name="p258615172811"></a><a name="p258615172811"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.56%" id="mcps1.2.3.1.2"><p id="p658645162811"><a name="p658645162811"></a><a name="p658645162811"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row14586751142811"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1058645110289"><a name="p1058645110289"></a><a name="p1058645110289"></a>nSize</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p058635172812"><a name="p058635172812"></a><a name="p058635172812"></a>src横向方向的size大小。</p>
<a name="ul85861951162813"></a><a name="ul85861951162813"></a><ul id="ul85861951162813"><li>不使能NZ2ND功能，必须为C0的倍数，此时连续传输数据块的个数为nSize / C0。</li><li>使能NZ2ND功能，不受限制。</li></ul>
</td>
</tr>
<tr id="row1158795120285"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p195868515281"><a name="p195868515281"></a><a name="p195868515281"></a>mSize</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p1586165111289"><a name="p1586165111289"></a><a name="p1586165111289"></a>src纵向方向的size大小。</p>
<a name="ul135871651162818"></a><a name="ul135871651162818"></a><ul id="ul135871651162818"><li>不使能NZ2ND功能，连续传输数据块的大小为mSize * C0个元素的长度。</li></ul>
<a name="ul35871851112818"></a><a name="ul35871851112818"></a><ul id="ul35871851112818"><li>使能NZ2ND功能，NZ/ND矩阵的大小为mSize * nSize。</li></ul>
</td>
</tr>
<tr id="row95874518288"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1158775152815"><a name="p1158775152815"></a><a name="p1158775152815"></a>dstStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><a name="ul658714512286"></a><a name="ul658714512286"></a><ul id="ul658714512286"><li>不使能NZ2ND功能<p id="p13413101019019"><a name="p13413101019019"></a><a name="p13413101019019"></a>dst相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔），取值不为0。单位为DataBlock（32字节）。</p>
</li></ul>
<a name="ul19587851142815"></a><a name="ul19587851142815"></a><ul id="ul19587851142815"><li>使能NZ2ND功能<p id="p9587115117281"><a name="p9587115117281"></a><a name="p9587115117281"></a>dst同一ND矩阵的相邻行的偏移（头与头），取值不为0， 单位为元素。</p>
</li></ul>
</td>
</tr>
<tr id="row15871851142820"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1558745112812"><a name="p1558745112812"></a><a name="p1558745112812"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><a name="ul1158711519284"></a><a name="ul1158711519284"></a><ul id="ul1158711519284"><li>不使能NZ2ND功能<p id="p058785162816"><a name="p058785162816"></a><a name="p058785162816"></a>src相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔），必须为16的倍数。取值范围：srcStride∈[0, 65535]， 单位：C0_Size(C0 * sizeof(U)，U为src的数据类型)。</p>
</li><li>使能NZ2ND功能<p id="p1458755120280"><a name="p1458755120280"></a><a name="p1458755120280"></a>src同一NZ矩阵的相邻Z排布的偏移（头与头），必须为16的倍数，取值范围：srcStride∈[0, 65535]，单位C0_size。</p>
</li></ul>
</td>
</tr>
<tr id="row11588175116280"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p058755113284"><a name="p058755113284"></a><a name="p058755113284"></a>quantPre</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p958845172814"><a name="p958845172814"></a><a name="p958845172814"></a>用于控制量化模式，QuantMode_t类型，具体定义如下。默认值为QuantMode_t::NoQuant，即不使能量化功能。</p>
<div class="p" id="p158813518287"><a name="p158813518287"></a><a name="p158813518287"></a>配置为scalar量化时，需要调用<a href="SetFixpipePreQuantFlag.md">SetFixpipePreQuantFlag</a>接口来设置scalar量化参数；配置为tensor量化时，需要调用<a href="SetFixPipeConfig.md">SetFixPipeConfig</a>来设置tensor量化参数。<a name="screen1658855117287"></a><a name="screen1658855117287"></a><pre class="screen" codetype="Cpp" id="screen1658855117287">enum QuantMode_t
{
    NoQuant,      // 不使能量化功能
    F322F16,      // float量化成half, scalar量化
    F322BF16,     // float量化成bfloat16_t, scalar量化
    DEQF16,       // int32_t量化成half, scalar量化
    VDEQF16,      // int32_t量化成half，tensor量化
    QF322B8_PRE,  // float量化成int8_t/uint8_t，scalar量化
    VQF322B8_PRE, // float量化成int8_t/uint8_t，tensor量化
    REQ8,         // int32_t量化成int8_t/uint8_t，scalar量化
    VREQ8,        // int32_t量化成int8_t/uint8_t，tensor量化
};</pre>
</div>
</td>
</tr>
<tr id="row16589125162817"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1258865114282"><a name="p1258865114282"></a><a name="p1258865114282"></a>reluPre</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p12589195182811"><a name="p12589195182811"></a><a name="p12589195182811"></a>用于配置relu操作的模式，类型为uint8_t，取值如下：</p>
<a name="ul45892512285"></a><a name="ul45892512285"></a><ul id="ul45892512285"><li>0：不使能relu</li><li>1：Normal relu</li></ul>
</td>
</tr>
<tr id="row125892051182817"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p1758914518286"><a name="p1758914518286"></a><a name="p1758914518286"></a>channelSplit</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p9589115162816"><a name="p9589115162816"></a><a name="p9589115162816"></a>类型为bool，配置是否使能channel切分，对于float类型的dst生效。</p>
<a name="ul1758913514289"></a><a name="ul1758913514289"></a><ul id="ul1758913514289"><li>false：不使能</li><li>true：使能</li></ul>
</td>
</tr>
<tr id="row05907514285"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p758945162812"><a name="p758945162812"></a><a name="p758945162812"></a>nz2ndEn</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p105868160369"><a name="p105868160369"></a><a name="p105868160369"></a>类型为bool，配置是否使能NZ2ND的格式转换，仅在CO1 -&gt; GM通路生效。</p>
<p id="p058917510285"><a name="p058917510285"></a><a name="p058917510285"></a>如果要使能NZ2ND的功能需要同步调用<a href="SetFixpipeNz2ndFlag.md">SetFixpipeNz2ndFlag</a>来设置格式转换的相关配置信息。</p>
<a name="ul75891951112811"></a><a name="ul75891951112811"></a><ul id="ul75891951112811"><li>false：不使能</li><li>true：使能</li></ul>
</td>
</tr>
<tr id="row191139435013"><td class="cellrowborder" valign="top" width="18.44%" headers="mcps1.2.3.1.1 "><p id="p22986503015"><a name="p22986503015"></a><a name="p22986503015"></a>sid</p>
</td>
<td class="cellrowborder" valign="top" width="81.56%" headers="mcps1.2.3.1.2 "><p id="p22986503016"><a name="p22986503016"></a><a name="p22986503016"></a>预留参数，为后续的功能做保留，开发者暂时无需关注。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section182481936173316"></a>

无

## 约束说明<a name="section455344833317"></a>

无

## 支持的通路和数据类型<a name="section2614932173011"></a>

下文的数据通路均通过逻辑位置[TPosition](TPosition.md#table5376122715308)来表达，并注明了对应的物理通路。TPosition与物理内存的映射关系见[表1](通用说明和约束.md#table07372185712)。

**表 4**  Local Memory -\> Global Memory具体通路和支持的数据类型

<a name="table11342410534"></a>
<table><thead align="left"><tr id="row7372435317"><th class="cellrowborder" valign="top" width="27.38%" id="mcps1.2.5.1.1"><p id="p203192445318"><a name="p203192445318"></a><a name="p203192445318"></a>支持型号</p>
</th>
<th class="cellrowborder" valign="top" width="13.489999999999998%" id="mcps1.2.5.1.2"><p id="p4312410536"><a name="p4312410536"></a><a name="p4312410536"></a>数据通路</p>
</th>
<th class="cellrowborder" valign="top" width="23.990000000000002%" id="mcps1.2.5.1.3"><p id="p6382417536"><a name="p6382417536"></a><a name="p6382417536"></a>源操作数的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="35.14%" id="mcps1.2.5.1.4"><p id="p226732755818"><a name="p226732755818"></a><a name="p226732755818"></a>目的操作数的数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row768718143414"><td class="cellrowborder" rowspan="2" valign="top" width="27.38%" headers="mcps1.2.5.1.1 "><p id="p059173617236"><a name="p059173617236"></a><a name="p059173617236"></a><span id="ph14591936192320"><a name="ph14591936192320"></a><a name="ph14591936192320"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="13.489999999999998%" headers="mcps1.2.5.1.2 "><p id="p1868711810344"><a name="p1868711810344"></a><a name="p1868711810344"></a>CO1 -&gt; GM（L0C Buffer -&gt; GM）</p>
</td>
<td class="cellrowborder" valign="top" width="23.990000000000002%" headers="mcps1.2.5.1.3 "><p id="p164642173719"><a name="p164642173719"></a><a name="p164642173719"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="35.14%" headers="mcps1.2.5.1.4 "><p id="p2045413461840"><a name="p2045413461840"></a><a name="p2045413461840"></a>uint8_t、int8_t、half、bfloat16_t、float</p>
</td>
</tr>
<tr id="row2040631320344"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1043117693719"><a name="p1043117693719"></a><a name="p1043117693719"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p17212393419"><a name="p17212393419"></a><a name="p17212393419"></a>uint8_t、int8_t、half、int16_t、int32_t</p>
</td>
</tr>
</tbody>
</table>

**表 5**  Local Memory -\> Local Memory具体通路和支持的数据类型

<a name="table1289914372562"></a>
<table><thead align="left"><tr id="row389923712561"><th class="cellrowborder" valign="top" width="26.980000000000004%" id="mcps1.2.5.1.1"><p id="p48990375567"><a name="p48990375567"></a><a name="p48990375567"></a>支持型号</p>
</th>
<th class="cellrowborder" valign="top" width="13.990000000000002%" id="mcps1.2.5.1.2"><p id="p10899037175616"><a name="p10899037175616"></a><a name="p10899037175616"></a>数据通路</p>
</th>
<th class="cellrowborder" valign="top" width="23.650000000000002%" id="mcps1.2.5.1.3"><p id="p489963717565"><a name="p489963717565"></a><a name="p489963717565"></a>源操作数的数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="35.38%" id="mcps1.2.5.1.4"><p id="p1011117501286"><a name="p1011117501286"></a><a name="p1011117501286"></a>目的操作数的数据类型</p>
</th>
</tr>
</thead>
<tbody><tr id="row1187182320389"><td class="cellrowborder" rowspan="2" valign="top" width="26.980000000000004%" headers="mcps1.2.5.1.1 "><p id="p1167920498384"><a name="p1167920498384"></a><a name="p1167920498384"></a><span id="ph5679104963810"><a name="ph5679104963810"></a><a name="ph5679104963810"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" width="13.990000000000002%" headers="mcps1.2.5.1.2 "><p id="p16797496386"><a name="p16797496386"></a><a name="p16797496386"></a>CO1 -&gt; A1（L0C Buffer -&gt; L1 Buffer）</p>
</td>
<td class="cellrowborder" valign="top" width="23.650000000000002%" headers="mcps1.2.5.1.3 "><p id="p18679204915381"><a name="p18679204915381"></a><a name="p18679204915381"></a>float</p>
</td>
<td class="cellrowborder" valign="top" width="35.38%" headers="mcps1.2.5.1.4 "><p id="p12363951468"><a name="p12363951468"></a><a name="p12363951468"></a>uint8_t、int8_t、half、bfloat16_t、float</p>
</td>
</tr>
<tr id="row2050020340381"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p768094983818"><a name="p768094983818"></a><a name="p768094983818"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p11669510362"><a name="p11669510362"></a><a name="p11669510362"></a>uint8_t、int8_t、half、int16_t、int32_t</p>
</td>
</tr>
</tbody>
</table>

## 调用示例<a name="section2409153316111"></a>

-   随路格式转换数据搬运，通路：CO1-\>A1、CO1-\>GM

    示例：Mmad含有矩阵乘偏置，左矩阵和右矩阵的数据类型为int8\_t，结果矩阵的数据类型为int32\_t。量化模式DEQF16，scalar量化参数为0.5，将Mmad计算出的结果由int32\_t量化成half并搬出。

    ```
    #ifdef ASCENDC_CPU_DEBUG
    #include "tikicpulib.h"
    #endif
    #include "kernel_operator.h"
    #include "../../instrs/common_utils/register_utils.h"
    SET_G_CORE_TYPE_IS_AIC
    template <typename dst_T, typename fmap_T, typename weight_T, typename dstCO1_T> class KernelCubeDataCopy{
    public:
        __aicore__ inline KernelCubeDataCopy(uint16_t CoutIn, uint8_t dilationHIn, uint8_t dilationWIn, QuantMode_t deqModeIn)
        {
            // ceiling of 16
            Cout = CoutIn;
            dilationH = dilationHIn;
            dilationW = dilationWIn;
            C0 = 32 / sizeof(fmap_T);
            C1 = channelSize / C0;
            coutBlocks = (Cout + 16 - 1) / 16;
            ho = H - dilationH * (Kh - 1);
            wo = W - dilationW * (Kw - 1);
            howo = ho * wo;
            howoRound = ((howo + 16 - 1) / 16) * 16;
            featureMapA1Size = C1 * H * W * C0;      // shape: [C1, H, W, C0]
            weightA1Size = C1 * Kh * Kw * Cout * C0; // shape: [C1, Kh, Kw, Cout, C0]
            featureMapA2Size = howoRound * (C1 * Kh * Kw * C0);
            weightB2Size = (C1 * Kh * Kw * C0) * coutBlocks * 16;
            m = howo;
            k = C1 * Kh * Kw * C0;
            n = Cout;
            biasSize = Cout;                  // shape: [Cout]
            dstSize = coutBlocks * howo * 16; // shape: [coutBlocks, howo, 16]
            dstCO1Size = coutBlocks * howoRound * 16;
            fmRepeat = featureMapA2Size / (16 * C0);
            weRepeat = weightB2Size / (16 * C0);
            deqMode = deqModeIn;
        }
        __aicore__ inline void Init(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm, __gm__ uint8_t* biasGm, __gm__ uint8_t* deqGm, __gm__ uint8_t* dstGm)
        {
            fmGlobal.SetGlobalBuffer((__gm__ fmap_T*)fmGm);
            weGlobal.SetGlobalBuffer((__gm__ weight_T*)weGm);
            biasGlobal.SetGlobalBuffer((__gm__ dstCO1_T*)biasGm);
            deqGlobal.SetGlobalBuffer((__gm__ uint64_t*)deqGm);
            dstGlobal.SetGlobalBuffer((__gm__ dst_T*)dstGm);
            pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(fmap_T));
            pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(fmap_T));
            pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(weight_T));
            pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(weight_T));
            pipe.InitBuffer(inQueueBiasA1, 1, biasSize * sizeof(dstCO1_T));
            pipe.InitBuffer(inQueueDeqA1, 1, dstCO1Size * sizeof(uint64_t));
            pipe.InitBuffer(inQueueDeqFB, 1, dstCO1Size * sizeof(uint64_t));
            pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(dstCO1_T));
            pipe.InitBuffer(outQueueA1, 1, dstCO1Size * sizeof(dst_T));
         }
        __aicore__ inline void Process()
        {
            CopyIn();
            Split();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.AllocTensor<fmap_T>();
            AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.AllocTensor<weight_T>();
            AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.AllocTensor<dstCO1_T>();
            AscendC::DataCopy(featureMapA1, fmGlobal, { 1, static_cast<uint16_t>(featureMapA1Size * sizeof(fmap_T) / 32), 0, 0 });
            AscendC::DataCopy(weightB1, weGlobal, { 1, static_cast<uint16_t>(weightA1Size * sizeof(weight_T) / 32), 0, 0 });
            AscendC::DataCopy(biasA1, biasGlobal, { 1, static_cast<uint16_t>(biasSize * sizeof(dstCO1_T) / 32), 0, 0 });
            inQueueFmA1.EnQue(featureMapA1);
            inQueueWeB1.EnQue(weightB1);
            inQueueBiasA1.EnQue(biasA1);
        }
        __aicore__ inline void Split()
        {
            AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
            AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
            AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
            AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();
            uint8_t padList[] = {0, 0, 0, 0};
            // load3dv2
            AscendC::LoadData(featureMapA2, featureMapA1, { padList, H, W, channelSize, k, howoRound, 0, 0, 1, 1, Kw, Kh, dilationW, dilationH, false, false, 0 });
            // load2d
            AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });
            inQueueFmA2.EnQue<fmap_T>(featureMapA2);
            inQueueWeB2.EnQue<weight_T>(weightB2);
            inQueueFmA1.FreeTensor(featureMapA1);
            inQueueWeB1.FreeTensor(weightB1);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.DeQue<fmap_T>();
            AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.DeQue<weight_T>();
            AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.AllocTensor<dstCO1_T>();
            AscendC::LocalTensor<dstCO1_T> biasA1 = inQueueBiasA1.DeQue<dstCO1_T>();
            // C = A * B + bias
            // m: 左矩阵Height, k: 左矩阵Width, n: 右矩阵Width
            AscendC::Mmad(dstCO1, featureMapA2, weightB2, biasA1, { m, n, k, true, 0, false, false, false });
            outQueueCO1.EnQue<dstCO1_T>(dstCO1);
            inQueueFmA2.FreeTensor(featureMapA2);
            inQueueWeB2.FreeTensor(weightB2);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<dstCO1_T> dstCO1 = outQueueCO1.DeQue<dstCO1_T>();
            AscendC::LocalTensor<dst_T> dstA1 = outQueueA1.DeQue<dst_T>();
            // 使能DEQF16量化，量化参数设置为0.5
            float tmp = (float)0.5;
            // 将float的tmp转换成uint64_t的deqScalar
            uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
            bool nz2ndEn = false;
            // nz2nd不使能时，nSize必须为16的倍数
            uint16_t nSize = coutBlocks * 16;
            uint16_t mSize = m;
            // srcStride必须为16的倍数
            uint16_t srcStride = (m + 16 - 1) / 16 * 16;
            // nz2nd不使能时，dstStride为burst头到头的距离，且为32B对齐
            uint32_t dstStride = m * sizeof(dst_T) * 16 / 32;
            if (nz2ndEn) {
                // nd矩阵的数量为1，src_nd_stride和dst_nd_stride填1
                AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
                // nz2nd使能时，nSize可以不为16的倍数，与Mmad的n保持一致
                nSize = n;
                // nz2nd使能时，dstStride表示同一nd矩阵的相邻连续行的间隔，与n保持一致
                dstStride = nSize;
            };
            // 不使能relu与channelSplit
            AscendC::DataCopyCO12DstParams intriParams(nSize, mSize, dstStride, srcStride, deqMode, 0, false, nz2ndEn);
          
            // mov l0c to gm, deq scalar quant
            AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::DataCopy(dstGlobal, dstCO1, intriParams);
            // // mov l0c to gm, deq tensor quant
            // // 需要额外申请deq tensor的gm空间，将值搬运到workA1
            // AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
            // // deq tensor的size
            // uint16_t deqSize = 128;
            // AscendC::DataCopy(workA1, deqGlobal, deqSize);
            // // deq tensor在fix上的地址
            // AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>();
            // // l1->fix, burst_len unit is 128Bytes
            // uint16_t fbufBurstLen = deqSize / 128;
            // AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
            // AscendC::DataCopy(deqFB, workA1, dataCopyParams);
            // // 设置量化tensor
            // AscendC::SetFixPipeConfig(deqFB);
            // AscendC::PipeBarrier<PIPE_FIX>();
            // AscendC::DataCopy(dstGlobal, dstCO1, intriParams);
            // inQueueDeqA1.FreeTensor(workA1);
            // inQueueDeqFB.FreeTensor(deqFB);
            // // mov l0c to l1, deq scalar quant, and then mov l1 to gm
            // AscendC::SetFixpipePreQuantFlag(deqScalar);  // 设置量化参数
            // AscendC::PipeBarrier<PIPE_FIX>();
            // AscendC::DataCopy(dstA1, dstCO1, intriParams);
            // AscendC::DataCopy(dstGlobal, dstA1, dstCO1Size);
            // // mov l0c to l1, deq tensor quant, and then mov l1 to gm
            // AscendC::LocalTensor<uint64_t> workA1 = inQueueDeqA1.AllocTensor<uint64_t>();
            // uint16_t deqSize = 128;
            // AscendC::DataCopy(workA1, deqGlobal, deqSize);
            // AscendC::LocalTensor<uint64_t> deqFB = inQueueDeqFB.AllocTensor<uint64_t>();
            // uint16_t fbufBurstLen = deqSize / 128;
            // AscendC::DataCopyParams dataCopyParams(1, fbufBurstLen, 0, 0);
            // AscendC::DataCopy(deqFB, workA1, dataCopyParams);
            // // 设置量化tensor
            // AscendC::SetFixPipeConfig(deqFB);
            // AscendC::PipeBarrier<PIPE_FIX>();
            // AscendC::DataCopy(dstA1, dstCO1, intriParams);
            // AscendC::DataCopy(dstGlobal, dstA1, dstCO1Size);
            // inQueueDeqA1.FreeTensor(workA1);
            // inQueueDeqFB.FreeTensor(deqFB);
            // outQueueCO1.FreeTensor(dstCO1);
            // outQueueA1.FreeTensor(dstA1);
        }
    private:
        AscendC::TPipe pipe;
        // feature map queue
        AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
        AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
        // weight queue
        AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
        AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
        // bias queue
        AscendC::TQue<AscendC::TPosition::A1, 1> inQueueBiasA1;
        // deq tensor queue
        AscendC::TQue<AscendC::TPosition::A1, 1> inQueueDeqA1;
        // fb dst of deq tensor
        AscendC::TQue<AscendC::TPosition::C2PIPE2GM, 1> inQueueDeqFB;
        // dst queue
        AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
        AscendC::TQue<AscendC::TPosition::A1, 1> outQueueA1;
        AscendC::GlobalTensor<fmap_T> fmGlobal;
        AscendC::GlobalTensor<weight_T> weGlobal;
        AscendC::GlobalTensor<dst_T> dstGlobal;
        AscendC::GlobalTensor<uint64_t> deqGlobal;
        AscendC::GlobalTensor<dstCO1_T> biasGlobal;
        AscendC::GlobalTensor<half> eleWiseGlobal;
        uint16_t channelSize = 32;
        uint16_t H = 4, W = 4;
        uint8_t Kh = 2, Kw = 2;
        uint16_t Cout;
        uint16_t C0, C1;
        uint8_t dilationH, dilationW;
        uint16_t coutBlocks, ho, wo, howo, howoRound;
        uint32_t featureMapA1Size, weightA1Size, featureMapA2Size, weightB2Size, biasSize, dstSize, dstCO1Size;
        uint16_t m, k, n;
        uint8_t fmRepeat, weRepeat;
        QuantMode_t deqMode = QuantMode_t::NoQuant;
    };
    #define KERNEL_CUBE_DATACOPY(dst_type, fmap_type, weight_type, dstCO1_type, CoutIn, dilationHIn, dilationWIn, deqModeIn)  \
        extern "C" __global__ __aicore__ void cube_datacopy_kernel_##fmap_type(__gm__ uint8_t* fmGm, __gm__ uint8_t* weGm,    \
            __gm__ uint8_t* biasGm, __gm__ uint8_t* deqGm, __gm__ uint8_t* dstGm)                                             \
        {                                                                                                                     \
            if (g_coreType == AscendC::AIV) {                                                                                 \
                return;                                                                                                       \
            }                                                                                                                 \
            KernelCubeDataCopy<dst_type, fmap_type, weight_type, dstCO1_type> op(CoutIn, dilationHIn, dilationWIn,            \
                deqModeIn);                                                                                                   \
            op.Init(fmGm, weGm, biasGm, deqGm, dstGm);                                                                        \
            op.Process();                                                                                                     \
        }
    KERNEL_CUBE_DATACOPY(half, int8_t, int8_t, int32_t, 128, 1, 1, QuantMode_t::DEQF16);
    ```

