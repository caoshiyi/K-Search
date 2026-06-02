# Select（灵活标量位置）<a name="ZH-CN_TOPIC_0000002523303866"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

给定两个源操作数src0和src1，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时从src1选取。

对于tensor高维切分计算接口，支持根据mask参数对上述选取结果，再次进行过滤，有效位填入最终的dst，无效位则保持dst原始值。例如：src0为\[1,2,3,4,5,6,7,8\]，src1为\[9,10,11,12,13,14,15,16\]，selMask为\[0,0,0,0,1,1,1,1\]，mask为\[1,1,1,1,0,0,0,0\]，dst原始值为\[-1,-2,-3,-4,-5,-6,-7,-8\]，则根据selMask的比特位选取后的结果dst\_temp为：\[9,10,11,12,5,6,7,8\]，然后再根据mask进行过滤，dst的最终输出结果为\[9,10,11,12,-5,-6,-7,-8\]。

本选择功能支持三种模式：

-   模式0：根据selMask在两个tensor中选取元素。selMask中有效数据的个数存在限制，具体取决于源操作数的数据类型。在每一轮迭代中，根据selMask的有效位数据进行选择操作，每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。
-   模式1：根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。多轮迭代时，每轮迭代连续使用selMask的不同部分。
-   模式2：根据selMask在两个tensor中选取元素，selMask无有效数据限制。多轮迭代时，每轮迭代连续使用selMask的不同部分。

针对模式1，提供灵活标量位置的接口。

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
    __aicore__ inline void Select(const T2& dst, const LocalTensor<T1>& selMask, const T3& src0, const T4& src1, SELMODE selMode, uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, bool isSetMask = true, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
        __aicore__ inline void Select(const T2& dst, const LocalTensor<T1>& selMask, const T3& src0, const T4& src1, SELMODE selMode, uint64_t mask[], uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T0 = BinaryDefaultType, typename T1 = BinaryDefaultType, bool isSetMask = true, const BinaryConfig &config = DEFAULT_BINARY_CONFIG, typename T2, typename T3, typename T4>
        __aicore__ inline void Select(const T2& dst, const LocalTensor<T1>& selMask, const T3& src0, const T4& src1, SELMODE selMode, uint64_t mask, uint8_t repeatTime, const BinaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1183814716169"></a>
<table><thead align="left"><tr id="row0838154741615"><th class="cellrowborder" valign="top" width="16.580000000000002%" id="mcps1.2.3.1.1"><p id="p683818472167"><a name="p683818472167"></a><a name="p683818472167"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.42%" id="mcps1.2.3.1.2"><p id="p208381447161619"><a name="p208381447161619"></a><a name="p208381447161619"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row166091324112514"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p186091247251"><a name="p186091247251"></a><a name="p186091247251"></a>T0</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p186091824132516"><a name="p186091824132516"></a><a name="p186091824132516"></a>源操作数和目的操作数的数据类型。</p>
<p id="p2094981215419"><a name="p2094981215419"></a><a name="p2094981215419"></a>特别地，对于灵活标量位置接口，为预留参数，暂未启用，为后续的功能扩展做保留，需要指定时，传入默认值BinaryDefaultType即可。</p>
</td>
</tr>
<tr id="row135451522112520"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p1354612213257"><a name="p1354612213257"></a><a name="p1354612213257"></a>T1</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p14571141782416"><a name="p14571141782416"></a><a name="p14571141782416"></a>selMask的数据类型。</p>
</td>
</tr>
<tr id="row18381947151619"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p683894716163"><a name="p683894716163"></a><a name="p683894716163"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p134861920172013"><a name="p134861920172013"></a><a name="p134861920172013"></a>预留参数，保持默认值即可。如需使用在接口外部设置mask的功能，可以调用不传入mask参数的接口来实现。</p>
</td>
</tr>
<tr id="row3400166112410"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p7401463247"><a name="p7401463247"></a><a name="p7401463247"></a>selMode</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p840106102420"><a name="p840106102420"></a><a name="p840106102420"></a>同<a href="#table8955841508">表2 参数说明</a>中的selMode参数说明。</p>
</td>
</tr>
<tr id="row1010412442456"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p914318523429"><a name="p914318523429"></a><a name="p914318523429"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p19365134614341"><a name="p19365134614341"></a><a name="p19365134614341"></a>类型为BinaryConfig，当标量为LocalTensor单点元素类型时生效，用于指定单点元素操作数位置。默认值DEFAULT_BINARY_CONFIG，表示右操作数为标量。</p>
<a name="screen13143195284213"></a><a name="screen13143195284213"></a><pre class="screen" codetype="Cpp" id="screen13143195284213">struct BinaryConfig {
    int8_t scalarTensorIndex = 1; // 用于指定标量为LocalTensor单点元素时标量的位置，0表示左操作数，1表示右操作数
};
constexpr BinaryConfig DEFAULT_BINARY_CONFIG = {1};</pre>
</td>
</tr>
<tr id="row125610473454"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p7144195244214"><a name="p7144195244214"></a><a name="p7144195244214"></a>T2</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p3172440173612"><a name="p3172440173612"></a><a name="p3172440173612"></a>LocalTensor类型，根据输入参数dst自动推导相应的数据类型，开发者无需配置该参数，保证dst满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row16559144914459"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p51440526428"><a name="p51440526428"></a><a name="p51440526428"></a>T3</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p151721640153610"><a name="p151721640153610"></a><a name="p151721640153610"></a>LocalTensor类型或标量类型，根据输入参数src0自动推导相应的数据类型，开发者无需配置该参数，保证src0满足数据类型的约束即可。</p>
</td>
</tr>
<tr id="row4209185214512"><td class="cellrowborder" valign="top" width="16.580000000000002%" headers="mcps1.2.3.1.1 "><p id="p7144352114214"><a name="p7144352114214"></a><a name="p7144352114214"></a>T4</p>
</td>
<td class="cellrowborder" valign="top" width="83.42%" headers="mcps1.2.3.1.2 "><p id="p11721940193620"><a name="p11721940193620"></a><a name="p11721940193620"></a>LocalTensor类型或标量类型，根据输入参数src1自动推导相应的数据类型，开发者无需配置该参数，保证src1满足数据类型的约束即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="16.38163816381638%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.24112411241124%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.37723772377238%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p19576531173410"><a name="p19576531173410"></a><a name="p19576531173410"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p16576163119347"><a name="p16576163119347"></a><a name="p16576163119347"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="p785913161268"><a name="p785913161268"></a><a name="p785913161268"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p775563112310"><a name="p775563112310"></a><a name="p775563112310"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p087755215110"><a name="p087755215110"></a><a name="p087755215110"></a><span id="ph1886081495211"><a name="ph1886081495211"></a><a name="ph1886081495211"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/complex32/int64_t/uint64_t/complex64</p>
</td>
</tr>
<tr id="row59491678111"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p29491174110"><a name="p29491174110"></a><a name="p29491174110"></a>selMask</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p171691412131"><a name="p171691412131"></a><a name="p171691412131"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="p933512419919"><a name="p933512419919"></a><a name="p933512419919"></a>选取mask。</p>
<p id="p1634988196"><a name="p1634988196"></a><a name="p1634988196"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p962521419203"><a name="p962521419203"></a><a name="p962521419203"></a><span id="ph7168111512201"><a name="ph7168111512201"></a><a name="ph7168111512201"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t/uint16_t/uint32_t/uint64_t。</p>
<p id="p448916309617"><a name="p448916309617"></a><a name="p448916309617"></a>每个比特位表示1个元素的选取，当selMask的比特位为1时，从src0中选取元素；比特位为0时，从src1中选取元素。</p>
<p id="p1199617464116"><a name="p1199617464116"></a><a name="p1199617464116"></a>selMode为模式0时，在每一轮迭代中，根据selMask的有效位数据进行选择操作，每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。selMode为模式1/2时，多次迭代对selMask连续消耗。</p>
<a name="ul126291640131014"></a><a name="ul126291640131014"></a><ul id="ul126291640131014"><li>模式0：根据selMask在两个tensor中选取元素，selMask有位数限制，不管迭代多少次，每次迭代都只根据截取后的固定位数的selMask进行选择。当源操作数的数据类型为8位时，selMask前256比特位有效；当源操作数的数据类型为16位时，selMask前128比特位有效；源操作数的数据类型为32位时，selMask前64比特位有效；源操作数的数据类型为64位时，selMask前32比特位有效。</li><li>模式1：根据selMask在1个tensor和1个scalar标量中选取元素。支持多次迭代，选取方式为，根据selMask的设置值，如果selMask比特值为1，则选择src0内的同位置数值，如果selMask比特值为0，则选择标量值。selMask连续存放，当源操作数的数据类型为8位时，一次比较获取selMask256bit长度的数据；当源操作数的数据类型为16位时，一次比较获取selMask128bit长度的数据；源操作数的数据类型为32位时，一次比较获取selMask64bit长度的数据；源操作数的数据类型为64位时，一次比较获取selMask32bit长度的数据。</li><li>模式2：根据selMask在两个tensor中选取元素。支持多次迭代，选取方式为，根据selMask的设置值，如果selMask比特值为1，则选择src0内的同位置数值，如果selMask比特值为0，则选择src1内的同位置数值。selMask连续存放，当源操作数的数据类型为8位时，一次比较获取selMask256bit长度的数据；当源操作数的数据类型为16位时，一次比较获取selMask128bit长度的数据；源操作数的数据类型为32位时，一次比较获取selMask64bit长度的数据；源操作数的数据类型为64位时，一次比较获取selMask32bit长度的数据。</li></ul>
</td>
</tr>
<tr id="row69959512410"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p165761231123417"><a name="p165761231123417"></a><a name="p165761231123417"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p757693163410"><a name="p757693163410"></a><a name="p757693163410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><div class="p" id="p62581922192218"><a name="p62581922192218"></a><a name="p62581922192218"></a>灵活标量位置接口中源操作数。<a name="ul3585928979"></a><a name="ul3585928979"></a><ul id="ul3585928979"><li>类型为LocalTensor时，支持当作矢量操作数或标量单点元素，支持的TPosition为VECIN/VECCALC/VECOUT。<p id="p205584233295"><a name="p205584233295"></a><a name="p205584233295"></a><span id="ph2558172315291"><a name="ph2558172315291"></a><a name="ph2558172315291"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p12558323142913"><a name="p12558323142913"></a><a name="p12558323142913"></a><span id="ph7558182319297"><a name="ph7558182319297"></a><a name="ph7558182319297"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/complex32/int64_t/uint64_t/complex64</p>
</li><li>类型为标量时：<p id="p11558142332918"><a name="p11558142332918"></a><a name="p11558142332918"></a><span id="ph75580236298"><a name="ph75580236298"></a><a name="ph75580236298"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/complex32/int64_t/uint64_t/complex64</p>
</li></ul>
</div>
<p id="p136651831996"><a name="p136651831996"></a><a name="p136651831996"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p10576203116349"><a name="p10576203116349"></a><a name="p10576203116349"></a>src1</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p1157693123417"><a name="p1157693123417"></a><a name="p1157693123417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><div class="p" id="p17783113192218"><a name="p17783113192218"></a><a name="p17783113192218"></a>灵活标量位置接口中源操作数。<a name="ul825610308105"></a><a name="ul825610308105"></a><ul id="ul825610308105"><li>类型为LocalTensor时，支持当作矢量操作数或标量单点元素，支持的TPosition为VECIN/VECCALC/VECOUT。<p id="p29171220133019"><a name="p29171220133019"></a><a name="p29171220133019"></a><span id="ph99171820133010"><a name="ph99171820133010"></a><a name="ph99171820133010"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p149171920183012"><a name="p149171920183012"></a><a name="p149171920183012"></a><span id="ph129172208302"><a name="ph129172208302"></a><a name="ph129172208302"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/complex32/int64_t/uint64_t/complex64</p>
</li><li>类型为标量时：<p id="p1791719202308"><a name="p1791719202308"></a><a name="p1791719202308"></a><span id="ph89171020193018"><a name="ph89171020193018"></a><a name="ph89171020193018"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/half/bfloat16_t/float/int32_t/uint32_t/complex32/int64_t/uint64_t/complex64</p>
</li></ul>
</div>
<p id="p14981134719107"><a name="p14981134719107"></a><a name="p14981134719107"></a>数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p10974181411356"><a name="p10974181411356"></a><a name="p10974181411356"></a>selMode</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p1797491412352"><a name="p1797491412352"></a><a name="p1797491412352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="p16706144115550"><a name="p16706144115550"></a><a name="p16706144115550"></a>指令模式，SELMODE类型，取值如下：</p>
<a name="screen6704150175512"></a><a name="screen6704150175512"></a><pre class="screen" codetype="Cpp" id="screen6704150175512">enum class SELMODE : uint8_t {
    VSEL_CMPMASK_SPR = 0, 
    VSEL_TENSOR_SCALAR_MODE,
    VSEL_TENSOR_TENSOR_MODE,
};</pre>
<a name="ul167871527102517"></a><a name="ul167871527102517"></a><ul id="ul167871527102517"><li>模式0：取值为VSEL_CMPMASK_SPR。根据selMask在两个tensor中选取元素。selMask中有效数据的个数存在限制，<span>具体取决于源操作数的数据类型。</span>在每一轮迭代中，根据selMask的有效位数据进行选择操作，每一轮迭代采用的selMask，均为相同数值，即selMask的有效数值。</li><li>模式1：取值为VSEL_TENSOR_SCALAR_MODE。根据selMask在1个tensor和1个scalar标量中选取元素，selMask无有效数据限制。多轮迭代时，<span>每轮迭代连续使用</span>selMask<span>的不同部分。</span></li><li>模式2：取值为VSEL_TENSOR_TENSOR_MODE。根据selMask在两个tensor中选取元素，selMask无有效数据限制。多轮迭代时，<span>每轮迭代连续使用</span>selMask<span>的不同部分。</span></li></ul>
</td>
</tr>
<tr id="row769135514428"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p175358462154"><a name="p175358462154"></a><a name="p175358462154"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p0554313181312"><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><a name="zh-cn_topic_0000002523303824_p0554313181312"></a><span id="zh-cn_topic_0000002523303824_ph793119540147"><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a name="zh-cn_topic_0000002523303824_ph793119540147"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><a name="zh-cn_topic_0000002523303824_ul1255411133132"></a><ul id="zh-cn_topic_0000002523303824_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><a name="zh-cn_topic_0000002523303824_ul18554121313135"></a><ul id="zh-cn_topic_0000002523303824_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
</td>
</tr>
<tr id="row12284152699"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p557663119345"><a name="p557663119345"></a><a name="p557663119345"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p195761631163416"><a name="p195761631163416"></a><a name="p195761631163416"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="p0448621183615"><a name="p0448621183615"></a><a name="p0448621183615"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row191624543910"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1325595674818"><a name="p1325595674818"></a><a name="p1325595674818"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p172551556134814"><a name="p172551556134814"></a><a name="p172551556134814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002554343931_p12596185919348"><a name="zh-cn_topic_0000002554343931_p12596185919348"></a><a name="zh-cn_topic_0000002554343931_p12596185919348"></a>控制操作数地址步长的参数。<a href="BinaryRepeatParams.md">BinaryRepeatParams</a>类型，包含操作数相邻迭代间相同datablock的地址步长，操作数同一迭代内不同datablock的地址步长等参数。</p>
<p id="zh-cn_topic_0000002554343931_p1156819418442"><a name="zh-cn_topic_0000002554343931_p1156819418442"></a><a name="zh-cn_topic_0000002554343931_p1156819418442"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>；同一迭代内DataBlock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row17825055390"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p158803470265"><a name="p158803470265"></a><a name="p158803470265"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.24112411241124%" headers="mcps1.2.4.1.2 "><p id="p7880174717266"><a name="p7880174717266"></a><a name="p7880174717266"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.37723772377238%" headers="mcps1.2.4.1.3 "><p id="p122367392431"><a name="p122367392431"></a><a name="p122367392431"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section198548421851"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   调用灵活标量位置接口且源操作数为LocalTensor单点元素的场景，不支持源操作数和目的操作数地址重叠。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   针对Ascend 950PR/Ascend 950DT，int8\_t/uint8\_t/uint64\_t/int64\_t/complex32/complex64数据类型仅支持tensor前n个数据计算接口。
-   左操作数及右操作数中，必须有一个为矢量；当前不支持左右操作数同时为标量。
-   本接口传入LocalTensor单点数据作为标量时，idx参数需要传入编译期已知的常量，传入变量时需要声明为constexpr。
-   模式1场景使用灵活标量位置接口时需要填写模板参数config避免接口匹配到其他模式。

## 调用示例<a name="section642mcpsimp"></a>

-   Select-tensor前n个数据计算样例（模式1）

    ```
    // 灵活标量位置，src1Local[0]作为标量
    static constexpr AscendC::BinaryConfig config = { 1 };
    AscendC::Select<BinaryDefaultType, uint8_t, config>(dstLocal, maskLocal, src0Local, src1Local[0], AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, dataSize);
    
    // 灵活标量位置，src0Local[0]作为标量
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Select<BinaryDefaultType, uint8_t, config>(dstLocal, maskLocal, src0Local[0], src1Local, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, dataSize);
    ```

