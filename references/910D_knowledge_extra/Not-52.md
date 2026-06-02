# Not<a name="ZH-CN_TOPIC_0000002523343958"></a>

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

    对输入srcReg中的每个有效数据按位取反，将结果写入dstReg。

-   对MaskReg操作：

    对输入src中的每个有效位（bit）取反，将结果写入dst。

## 函数原型<a name="section620mcpsimp"></a>

-   对RegTensor进行操作

    ```
    template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename U>
    __simd_callee__ inline void Not(U& dstReg, U& srcReg, MaskReg& mask)
    ```

-   对MaskReg进行操作

    ```
    __simd_callee__ inline void Not(MaskReg& dst, MaskReg& src, MaskReg& mask)
    ```

## 参数说明<a name="section622mcpsimp"></a>

-   对RegTensor进行操作

    **表 1**  模板参数说明

    <a name="table4835205712588"></a>
    <table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.61%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="81.39%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数数据类型。</p>
    <p id="p7373174413584"><a name="p7373174413584"></a><a name="p7373174413584"></a><span id="ph1137394417584"><a name="ph1137394417584"></a><a name="ph1137394417584"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/int64_t/uint64_t/half/float</p>
    </td>
    </tr>
    <tr id="row18835145716587"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p1756419170189"><a name="p1756419170189"></a><a name="p1756419170189"></a>mode</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p77520541653"><a name="p77520541653"></a><a name="p77520541653"></a>选择MERGING模式或ZEROING模式。</p>
    <a name="ul1163765616511"></a><a name="ul1163765616511"></a><ul id="ul1163765616511"><li>ZEROING，mask未筛选的元素在dst中置零。</li><li>MERGING，当前不支持。</li></ul>
    </td>
    </tr>
    <tr id="row2424151625011"><td class="cellrowborder" valign="top" width="18.61%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a>U</p>
    </td>
    <td class="cellrowborder" valign="top" width="81.39%" headers="mcps1.2.3.1.2 "><p id="p15901115914145"><a name="p15901115914145"></a><a name="p15901115914145"></a><span id="ph19851723182011"><a name="ph19851723182011"></a><a name="ph19851723182011"></a>目的操作数的RegTensor类型，例如RegTensor&lt;half&gt;，由编译器自动推导，用户不需要填写。</span></p>
    </td>
    </tr>
    </tbody>
    </table>

    **表 2**  参数说明

    <a name="table147028618289"></a>
    <table><thead align="left"><tr id="row19702156192811"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="p1670217682820"><a name="p1670217682820"></a><a name="p1670217682820"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p20702464283"><a name="p20702464283"></a><a name="p20702464283"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="p870215616281"><a name="p870215616281"></a><a name="p870215616281"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1970206162814"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p157029620283"><a name="p157029620283"></a><a name="p157029620283"></a>dstReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p187024618284"><a name="p187024618284"></a><a name="p187024618284"></a>输出</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p5702164284"><a name="p5702164284"></a><a name="p5702164284"></a>目的操作数。</p>
    <p id="p27023615289"><a name="p27023615289"></a><a name="p27023615289"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
    </td>
    </tr>
    <tr id="row970220672819"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6702469281"><a name="p6702469281"></a><a name="p6702469281"></a>srcReg</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p57029614285"><a name="p57029614285"></a><a name="p57029614285"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p177029622816"><a name="p177029622816"></a><a name="p177029622816"></a>源操作数。</p>
    <p id="p137026616286"><a name="p137026616286"></a><a name="p137026616286"></a>类型为<a href="RegTensor.md">RegTensor</a>。</p>
    <p id="p670212611287"><a name="p670212611287"></a><a name="p670212611287"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
    </td>
    </tr>
    <tr id="row167021569289"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p37029612280"><a name="p37029612280"></a><a name="p37029612280"></a>mask</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1970217602814"><a name="p1970217602814"></a><a name="p1970217602814"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p10702663286"><a name="p10702663286"></a><a name="p10702663286"></a><span id="ph167021767286"><a name="ph167021767286"></a><a name="ph167021767286"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
    </td>
    </tr>
    </tbody>
    </table>

-   对MaskReg进行操作

    **表 3**  参数说明

    <a name="table13822195442813"></a>
    <table><thead align="left"><tr id="row28221954102815"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="p1482275462812"><a name="p1482275462812"></a><a name="p1482275462812"></a>参数名</p>
    </th>
    <th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="p18822554202817"><a name="p18822554202817"></a><a name="p18822554202817"></a>描述</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row13822185410281"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p188221554112813"><a name="p188221554112813"></a><a name="p188221554112813"></a>dst</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p198221454142819"><a name="p198221454142819"></a><a name="p198221454142819"></a>目的操作数。</p>
    </td>
    </tr>
    <tr id="row12822175452814"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p59747391278"><a name="p59747391278"></a><a name="p59747391278"></a>src</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p53355414286"><a name="p53355414286"></a><a name="p53355414286"></a>源操作数。</p>
    </td>
    </tr>
    <tr id="row3926164202714"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p1892618426271"><a name="p1892618426271"></a><a name="p1892618426271"></a>mask</p>
    </td>
    <td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p3926174212712"><a name="p3926174212712"></a><a name="p3926174212712"></a>指示在计算过程中哪些bit有效。</p>
    </td>
    </tr>
    </tbody>
    </table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

-   对RegTensor进行操作

    ```
    template <typename T>
    __simd_vf__ inline void NotVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, 
     uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg;
        AscendC::MicroAPI::RegTensor<T> dstReg;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
            AscendC::MicroAPI::Not(dstReg, srcReg, mask);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
        }
    }
    ```

-   对MaskReg进行操作

    ```
    template <typename T>
    __simd_vf__ inline void NotVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
    {
        AscendC::MicroAPI::RegTensor<T> srcReg;
        AscendC::MicroAPI::MaskReg src = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALLF>();
        AscendC::MicroAPI::MaskReg dst;
        AscendC::MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * oneRepeatSize);
            AscendC::MicroAPI::Not(dst, src, mask);
            AscendC::MicroAPI::Adds(srcReg, srcReg, 0, dst);
            AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, srcReg, mask);
        }
    }
    ```

