# And<a name="ZH-CN_TOPIC_0000002554424507"></a>

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

本节介绍两种接口，分别用于对RegTensor和MaskReg进行操作：

-   对RegTensor操作：

    根据mask对输入数据srcReg0、srcReg1执行按元素与\( & \)操作，将结果写入dstReg。计算公式如下：

    <!-- img2text -->
$$
dst_i = src0_i \,\&\, src1_i
$$

-   对MaskReg操作：

    将两个输入MaskReg的有效bit进行逻辑与运算得到新的MaskReg。

## 函数原型<a name="section620mcpsimp"></a>

-   对RegTensor操作

    ```
    template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
    __simd_callee__ inline void And(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
    ```

-   对MaskReg操作

    ```
    __simd_callee__ inline void And(MaskReg& dst, MaskReg& src0, MaskReg& src1, MaskReg& mask)
    ```

## 参数说明<a name="section622mcpsimp"></a>

-   对RegTensor操作

    **表 1**  模板参数说明

    <a name="table172551105016"></a>
    <table><thead align="left"><tr id="row152514120505"><th class="cellrowborder" valign="top" width="18.2%" id="mcps1.2.3.1.1"><p id="p1325101195013"><a name="p1325101195013"></a><a name="p1325101195013"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="81.8%" id="mcps1.2.3.1.2"><p id="p12520118504"><a name="p12520118504"></a><a name="p12520118504"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row92541205014"><td class="cellrowborder" valign="top" width="18.2%" headers="mcps1.2.3.1.1 "><p id="p11250110504"><a name="p11250110504"></a><a name="p11250110504"></a>T</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.8%" headers="mcps1.2.3.1.2 "><p id="p625141195017"><a name="p625141195017"></a><a name="p625141195017"></a>操作数数据类型。</p>
    <p id="p1225115501"><a name="p1225115501"></a><a name="p1225115501"></a><span id="ph11257125017"><a name="ph11257125017"></a><a name="ph11257125017"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool/uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/uint64_t/int64_t</p>
    </td>
    </tr>
    <tr id="row12514135013"><td class="cellrowborder" valign="top" width="18.2%" headers="mcps1.2.3.1.1 "><p id="p52512195012"><a name="p52512195012"></a><a name="p52512195012"></a>mode</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.8%" headers="mcps1.2.3.1.2 "><p id="p22511112504"><a name="p22511112504"></a><a name="p22511112504"></a>选择MERGING模式或ZEROING模式。</p>
    <a name="ul10251514504"></a><a name="ul10251514504"></a><ul id="ul10251514504"><li>ZEROING, mask未筛选的元素在dst中置零。</li><li>MERGING, 当前不支持。</li></ul>
    </td>
    </tr>
    <tr id="row152591105013"><td class="cellrowborder" valign="top" width="18.2%" headers="mcps1.2.3.1.1 "><p id="p72531125019"><a name="p72531125019"></a><a name="p72531125019"></a>U</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.8%" headers="mcps1.2.3.1.2 "><p id="p725171135010"><a name="p725171135010"></a><a name="p725171135010"></a>srcReg0/srcReg1/dstReg RegTensor类型， 例如RegTensor&lt;uint32_t&gt;，由编译器自动推导，用户不需要填写。</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 2**  参数说明

    <a name="table14261514502"></a>
    <table><thead align="left"><tr id="row3262116508"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p6267118507"><a name="p6267118507"></a><a name="p6267118507"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p2026191195013"><a name="p2026191195013"></a><a name="p2026191195013"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p126619509"><a name="p126619509"></a><a name="p126619509"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row82617110505"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p426161165010"><a name="p426161165010"></a><a name="p426161165010"></a>dstReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1826171205012"><a name="p1826171205012"></a><a name="p1826171205012"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p20261145015"><a name="p20261145015"></a><a name="p20261145015"></a>目的操作数。</p>
    <p id="p22619115503"><a name="p22619115503"></a><a name="p22619115503"></a><span id="ph2261411508"><a name="ph2261411508"></a><a name="ph2261411508"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
    </td>
    </tr>
    <tr id="row626161145015"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p152610185010"><a name="p152610185010"></a><a name="p152610185010"></a>srcReg0</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p122611118506"><a name="p122611118506"></a><a name="p122611118506"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p16264155017"><a name="p16264155017"></a><a name="p16264155017"></a>源操作数。</p>
    <p id="p626151115013"><a name="p626151115013"></a><a name="p626151115013"></a><span id="ph18263113507"><a name="ph18263113507"></a><a name="ph18263113507"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
    <p id="p132651105015"><a name="p132651105015"></a><a name="p132651105015"></a>两个源操作数的数据类型需要与目的操作数保持一致。</p>
    </td>
    </tr>
    <tr id="row42610115502"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p2269110507"><a name="p2269110507"></a><a name="p2269110507"></a>srcReg1</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p7261815500"><a name="p7261815500"></a><a name="p7261815500"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1026171135019"><a name="p1026171135019"></a><a name="p1026171135019"></a>源操作数。</p>
    <p id="p926116509"><a name="p926116509"></a><a name="p926116509"></a><span id="ph32691135015"><a name="ph32691135015"></a><a name="ph32691135015"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
    <p id="p1261916506"><a name="p1261916506"></a><a name="p1261916506"></a>两个源操作数的数据类型需要与目的操作数保持一致。</p>
    </td>
    </tr>
    <tr id="row5268195011"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p12641145013"><a name="p12641145013"></a><a name="p12641145013"></a>mask</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p126171175011"><a name="p126171175011"></a><a name="p126171175011"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11261112501"><a name="p11261112501"></a><a name="p11261112501"></a><span id="ph72621155015"><a name="ph72621155015"></a><a name="ph72621155015"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
    </td>
    </tr>
    </tbody>
    </table>

-   对MaskReg操作

    <a name="table17445957175011"></a>
    <table><thead align="left"><tr id="row134451857195011"><th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.1"><p id="p16445135775019"><a name="p16445135775019"></a><a name="p16445135775019"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="50%" id="mcps1.1.3.1.2"><p id="p1744511579501"><a name="p1744511579501"></a><a name="p1744511579501"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row74451857145019"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1944512571507"><a name="p1944512571507"></a><a name="p1944512571507"></a>dst</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p1744565718504"><a name="p1744565718504"></a><a name="p1744565718504"></a>目的操作数。</p>
    </td>
    </tr>
    <tr id="row144451857175020"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p59747391278"><a name="p59747391278"></a><a name="p59747391278"></a>src0</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p53355414286"><a name="p53355414286"></a><a name="p53355414286"></a>源操作数。</p>
    </td>
    </tr>
    <tr id="row2521428183011"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p58511219123111"><a name="p58511219123111"></a><a name="p58511219123111"></a>src1</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p175262810308"><a name="p175262810308"></a><a name="p175262810308"></a>源操作数。</p>
    </td>
    </tr>
    <tr id="row3926164202714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.1 "><p id="p1892618426271"><a name="p1892618426271"></a><a name="p1892618426271"></a>mask</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.1.3.1.2 "><p id="p3926174212712"><a name="p3926174212712"></a><a name="p3926174212712"></a>指示在计算过程中哪些bit有效。</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

-   对RegTensor操作

    ```
    template <typename T>
    __simd_vf__ inline void AndVF(__ubuf__ T* dstAddr, __ubuf__ T* src0Addr, __ubuf__ T* src1Addr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg0;
        AscendC::MicroAPI::RegTensor<T> srcReg1;
        AscendC::MicroAPI::RegTensor<T> dstReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg0, src0Addr + i * oneRepeatSize);
            AscendC::MicroAPI::LoadAlign(srcReg1, src1Addr + i * oneRepeatSize);        
            AscendC::MicroAPI::And(dstReg, srcReg0, srcReg1, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

-   对MaskReg操作

    ```
    template <typename T>
    __simd_vf__ inline void AndVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg;
        AscendC::MicroAPI::MaskReg src0 = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALLF>();
        AscendC::MicroAPI::MaskReg src1 = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg dst;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
            AscendC::MicroAPI::And(dst, src0, src1, mask);
            AscendC::MicroAPI::Adds(srcReg, srcReg, 0, dst);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask);
        }
    }
    ```

