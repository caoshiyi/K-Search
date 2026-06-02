# 连续非对齐搬出<a name="ZH-CN_TOPIC_0000002554423787"></a>

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

Reg矢量计算数据搬运接口，适用于从RegTensor连续非对齐搬出到UB。

## 函数原型<a name="section620mcpsimp"></a>

```
// 场景1:使用uint32_t作为存储偏移量，使用post mode进行搬运，每调用一次接口更新目的操作数在UB上的地址
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T*& dstAddr, U& srcReg, UnalignRegForStore& ureg, uint32_t postUpdateStride); 

// 处理非对齐搬运的尾块，适用于场景1
template <typename T, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE>
__simd_callee__ inline void StoreUnAlignPost(__ubuf__ T*& dstAddr, UnalignRegForStore& ureg, int32_t postUpdateStride);

// 场景2：使用 AddrReg 存储偏移量进行搬运，使用post mode进行搬运，每调用一次接口更新目的操作数在UB上的地址
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T*& dstAddr, U& srcReg, UnalignRegForStore& ureg, AddrReg& areg);
// 处理非对齐搬运的尾块，适用于场景2
template <typename T>
__simd_callee__ inline void StoreUnAlignPost(__ubuf__ T*& dstAddr, UnalignRegForStore& ureg, AddrReg& areg);

// 场景3：配合Squeeze使用，连续非对齐搬出，SqueezeReg矢量计算API会存储有效元素的总字节数到AR特殊寄存器，使用此接口会将AR寄存器中效元素个数作为存储偏移量
template <typename T = DefaultType, PostLiteral postMode = PostLiteral::POST_MODE_UPDATE, typename U>
__simd_callee__ inline void StoreUnAlign(__ubuf__ T* dstAddr, U& srcReg, UnalignRegForStore& ureg);

// 处理非对齐搬运的尾块，适用于场景3
template <typename T>
__simd_callee__ inline void StoreUnAlignPost(__ubuf__ T* dstAddr, UnalignRegForStore& ureg);
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  连续非对齐搬出使用int32\_t存储偏移量场景

<a name="table14132101714462"></a>
<table><thead align="left"><tr id="row19176617124620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p117621724612"><a name="p117621724612"></a><a name="p117621724612"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.468753124687531%" id="mcps1.2.4.1.2"><p id="p417681717463"><a name="p417681717463"></a><a name="p417681717463"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.26257374262573%" id="mcps1.2.4.1.3"><p id="p5176017164614"><a name="p5176017164614"></a><a name="p5176017164614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1091516521711"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p123381318914"><a name="p123381318914"></a><a name="p123381318914"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1733820311492"><a name="p1733820311492"></a><a name="p1733820311492"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p53381931597"><a name="p53381931597"></a><a name="p53381931597"></a>模板参数，支持的数据类型为b8/b16/b32/b64。</p>
</td>
</tr>
<tr id="row25831652671"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p533814311498"><a name="p533814311498"></a><a name="p533814311498"></a>postMode</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1338031399"><a name="p1338031399"></a><a name="p1338031399"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p14697155195612"><a name="p14697155195612"></a><a name="p14697155195612"></a>用于控制是否使能post update，<a href="PostLiteral.md">PostLiteral</a>类型。</p>
</td>
</tr>
<tr id="row08095552714"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p20338831596"><a name="p20338831596"></a><a name="p20338831596"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p13393312913"><a name="p13393312913"></a><a name="p13393312913"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1433913312093"><a name="p1433913312093"></a><a name="p1433913312093"></a>模板参数，模板参数T支持的数据类型对应的RegTensor类型。</p>
</td>
</tr>
<tr id="row7879164218227"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p48791042192216"><a name="p48791042192216"></a><a name="p48791042192216"></a>ureg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p987974211223"><a name="p987974211223"></a><a name="p987974211223"></a>输出/输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p2087964292215"><a name="p2087964292215"></a><a name="p2087964292215"></a>UnalignRegForStore，非对齐寄存器，用于保存非对齐数据，长度32B。StoreUnAlign函数中作为输入/输出，StoreUnAlignPost函数中作为输入。</p>
</td>
</tr>
<tr id="row1417671716463"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p1039635944011"><a name="p1039635944011"></a><a name="p1039635944011"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1317616173468"><a name="p1317616173468"></a><a name="p1317616173468"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p2176141714466"><a name="p2176141714466"></a><a name="p2176141714466"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row958051624416"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p9580161634416"><a name="p9580161634416"></a><a name="p9580161634416"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p2581416144417"><a name="p2581416144417"></a><a name="p2581416144417"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p195811716134418"><a name="p195811716134418"></a><a name="p195811716134418"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row4397131094210"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p639881018426"><a name="p639881018426"></a><a name="p639881018426"></a>postUpdateStride</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1343482217424"><a name="p1343482217424"></a><a name="p1343482217424"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p20878111944912"><a name="p20878111944912"></a><a name="p20878111944912"></a>POST_MODE_NORMAL与POST_MODE_UPDATE场景下postUpdateStride含义不一致。</p>
<a name="ul71413941812"></a><a name="ul71413941812"></a><ul id="ul71413941812"><li>POST_MODE_NORMAL场景：实际搬运UB起始地址为dstAddr，搬运数量为postUpdateStride。</li><li>POST_MODE_UPDATE场景：实际搬运UB起始地址为dstAddr，搬运数量为postUpdateStride，搬运后执行地址更新dstAddr +=  postUpdateStride。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  连续非对齐搬出使用AddrReg存储偏移量场景

<a name="table19491132434910"></a>
<table><thead align="left"><tr id="row4491142474916"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p124911824124916"><a name="p124911824124916"></a><a name="p124911824124916"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.468753124687531%" id="mcps1.2.4.1.2"><p id="p1849110246490"><a name="p1849110246490"></a><a name="p1849110246490"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.26257374262573%" id="mcps1.2.4.1.3"><p id="p1249112242496"><a name="p1249112242496"></a><a name="p1249112242496"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1049210247496"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p7492152411499"><a name="p7492152411499"></a><a name="p7492152411499"></a>ureg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p11492724144915"><a name="p11492724144915"></a><a name="p11492724144915"></a>输出/输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p11492162424913"><a name="p11492162424913"></a><a name="p11492162424913"></a>UnalignRegForStore，非对齐寄存器，用于保存非对齐数据，长度32B。StoreUnAlign函数中作为输入/输出，StoreUnAlignPost函数中作为输入。</p>
</td>
</tr>
<tr id="row44921248492"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p34929248493"><a name="p34929248493"></a><a name="p34929248493"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p9492142418498"><a name="p9492142418498"></a><a name="p9492142418498"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p12492152410491"><a name="p12492152410491"></a><a name="p12492152410491"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
<tr id="row10492192415497"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p2049252419491"><a name="p2049252419491"></a><a name="p2049252419491"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p104921724184920"><a name="p104921724184920"></a><a name="p104921724184920"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p2049262417496"><a name="p2049262417496"></a><a name="p2049262417496"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row194928244493"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p9492142412492"><a name="p9492142412492"></a><a name="p9492142412492"></a>areg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p949272414920"><a name="p949272414920"></a><a name="p949272414920"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p1757614581216"><a name="p1757614581216"></a><a name="p1757614581216"></a>实际搬运UB起始地址为 AddrReg数据类型，存储地址偏移量offset。实际搬运UB起始地址为srcAddr + offset。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  配合Squeeze使用，连续非对齐搬出，使用AR寄存器中效元素个数作为存储偏移量

<a name="table19766162354619"></a>
<table><thead align="left"><tr id="row1476672354620"><th class="cellrowborder" valign="top" width="13.268673132686734%" id="mcps1.2.4.1.1"><p id="p16766112315461"><a name="p16766112315461"></a><a name="p16766112315461"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.468753124687531%" id="mcps1.2.4.1.2"><p id="p147661823104614"><a name="p147661823104614"></a><a name="p147661823104614"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.26257374262573%" id="mcps1.2.4.1.3"><p id="p13766923134614"><a name="p13766923134614"></a><a name="p13766923134614"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row19605752134618"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p3766723114617"><a name="p3766723114617"></a><a name="p3766723114617"></a>dstAddr</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p17661238463"><a name="p17661238463"></a><a name="p17661238463"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p3766182315465"><a name="p3766182315465"></a><a name="p3766182315465"></a>目的操作数在UB上的起始地址。</p>
</td>
</tr>
<tr id="row1476642317464"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p576652394619"><a name="p576652394619"></a><a name="p576652394619"></a>ureg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p1376652344614"><a name="p1376652344614"></a><a name="p1376652344614"></a>输出/输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p127661923134615"><a name="p127661923134615"></a><a name="p127661923134615"></a>UnalignRegForStore，非对齐寄存器，用于保存非对齐数据，长度32B。StoreUnAlign函数中作为输入/输出，StoreUnAlignPost函数中作为输入。</p>
</td>
</tr>
<tr id="row1276652312469"><td class="cellrowborder" valign="top" width="13.268673132686734%" headers="mcps1.2.4.1.1 "><p id="p2766123154619"><a name="p2766123154619"></a><a name="p2766123154619"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="12.468753124687531%" headers="mcps1.2.4.1.2 "><p id="p676622314469"><a name="p676622314469"></a><a name="p676622314469"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.26257374262573%" headers="mcps1.2.4.1.3 "><p id="p8766132374613"><a name="p8766132374613"></a><a name="p8766132374613"></a>源操作数，类型为RegTensor。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section1575141714439"></a>

-   该接口中的dstAddr不需要32B对齐。
-   StoreUnAlign与StoreUnAlignPost接口组合使用。

## 调用示例<a name="section642mcpsimp"></a>

```
// 连续非对齐搬出使用 uint32_t 存储偏移量场景
template <typename T>
__simd_vf__ inline void StoreUnAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t postUpdateStride, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::UnalignRegForLoad ureg0;
    AscendC::MicroAPI::UnalignRegForStore ureg1;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadUnAlignPre(ureg0, srcAddr + i * postUpdateStride);
        AscendC::MicroAPI::LoadUnAlign(srcReg, ureg0, srcAddr + i * postUpdateStride);
        AscendC::MicroAPI::StoreUnAlign(dstAddr, srcReg, ureg1, postUpdateStride);
    }
    AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg1, 0);
}

// 连续非对齐搬出使用 AddrReg 存储偏移量场景
template <typename T>
__simd_vf__ inline void StoreUnAlignVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;    
    AscendC::MicroAPI::UnalignRegForLoad ureg0;
    AscendC::MicroAPI::UnalignRegForStore ureg1;
    AscendC::MicroAPI::AddrReg aReg;
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
        aReg = AscendC::MicroAPI::CreateAddrReg<T>(i, oneRepeatSize);
        AscendC::MicroAPI::LoadUnAlignPre(ureg0, srcAddr, aReg);
        AscendC::MicroAPI::LoadUnAlign(srcReg, ureg0, srcAddr, aReg, 0);
        AscendC::MicroAPI::StoreUnAlign(dstAddr, srcReg, ureg1, aReg);
    }
    AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg1, aReg);
}

// 配合Squeeze使用，连续非对齐搬出，SqueezeReg矢量计算API会存储有效元素的总字节数到AR寄存器，使用AR寄存器中效元素个数作为存储偏移量
template <typename T>
__aicore__ inline void SqueezeVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> srcReg0;
    AscendC::MicroAPI::RegTensor<T> srcReg1;
    AscendC::MicroAPI::UnalignRegForStore ureg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::H>();
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(srcReg0, srcAddr, oneRepeatSize);
        AscendC::MicroAPI::Squeeze<T, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(srcReg1, srcReg0, mask);
        AscendC::MicroAPI::StoreUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, srcReg1, ureg);
     }
     AscendC::MicroAPI::StoreUnAlignPost(dstAddr, ureg);
}
```

