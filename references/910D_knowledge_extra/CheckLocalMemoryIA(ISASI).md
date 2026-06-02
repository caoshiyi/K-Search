# CheckLocalMemoryIA\(ISASI\)<a name="ZH-CN_TOPIC_0000002523304212"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1743412418717"><a name="p1743412418717"></a><a name="p1743412418717"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

check设定范围内的UB读写行为，如果有设定范围的读写行为则会出现EXCEPTION报错，无设定范围的读写行为则不会报错。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void CheckLocalMemoryIA(const CheckLocalMemoryIAParam& checkParams)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="21.64216421642164%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="16.55165516551655%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="61.8061806180618%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1075785651510"><td class="cellrowborder" valign="top" width="21.64216421642164%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>checkParams</p>
</td>
<td class="cellrowborder" valign="top" width="16.55165516551655%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="61.8061806180618%" headers="mcps1.2.4.1.3 "><p id="p17376814155615"><a name="p17376814155615"></a><a name="p17376814155615"></a>用于配置对UB访问的检查行为，类型为CheckLocalMemoryIAParam。</p>
<p id="p395104375712"><a name="p395104375712"></a><a name="p395104375712"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p12287014111614"><a name="p12287014111614"></a><a name="p12287014111614"></a>参数说明请参考<a href="#table15780447181917">表2</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  CheckLocalMemoryIAParam结构体内参数说明

<a name="table15780447181917"></a>
<table><thead align="left"><tr id="row0780947111915"><th class="cellrowborder" valign="top" width="15.229999999999999%" id="mcps1.2.3.1.1"><p id="p1780124771913"><a name="p1780124771913"></a><a name="p1780124771913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="84.77%" id="mcps1.2.3.1.2"><p id="p1578014718198"><a name="p1578014718198"></a><a name="p1578014718198"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row10780647151919"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p6340835122118"><a name="p6340835122118"></a><a name="p6340835122118"></a>enableBit</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p12340173514212"><a name="p12340173514212"></a><a name="p12340173514212"></a>配置的异常寄存器，取值范围：enableBit∈[0,3]，默认为0。</p>
<a name="ul1497210915510"></a><a name="ul1497210915510"></a><ul id="ul1497210915510"><li>0：异常寄存器0。</li><li>1：异常寄存器1。</li><li>2：异常寄存器2。</li><li>3：异常寄存器3。</li></ul>
</td>
</tr>
<tr id="row6780947191919"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p1934033512213"><a name="p1934033512213"></a><a name="p1934033512213"></a>startAddr</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p1634012352218"><a name="p1634012352218"></a><a name="p1634012352218"></a>Check的起始地址，32B对齐，取值范围：startAddr∈[0, 65535]，默认值为0。比如，可通过LocalTensor.GetPhyAddr()/32来获取startAddr。</p>
</td>
</tr>
<tr id="row1078074711194"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p334033518217"><a name="p334033518217"></a><a name="p334033518217"></a>endAddr</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p1734053517219"><a name="p1734053517219"></a><a name="p1734053517219"></a>Check的结束地址，32B对齐，取值范围：endAddr∈[0, 65535] 。默认值为0。</p>
</td>
</tr>
<tr id="row1761285762117"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p0306336224"><a name="p0306336224"></a><a name="p0306336224"></a>isScalarRead</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p193068319220"><a name="p193068319220"></a><a name="p193068319220"></a>Check标量读访问。</p>
<a name="ul490711252012"></a><a name="ul490711252012"></a><ul id="ul490711252012"><li>false：不开启，默认为false。</li><li>true：开启。</li></ul>
</td>
</tr>
<tr id="row1751545416214"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p88841831173112"><a name="p88841831173112"></a><a name="p88841831173112"></a>isScalarWrite</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p197361730144619"><a name="p197361730144619"></a><a name="p197361730144619"></a>Check标量写访问。</p>
<a name="ul1337350504"></a><a name="ul1337350504"></a><ul id="ul1337350504"><li>false：不开启，默认为false。</li><li>true：开启。</li></ul>
</td>
</tr>
<tr id="row1155115212117"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p1343003353112"><a name="p1343003353112"></a><a name="p1343003353112"></a>isVectorRead</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p5522145964513"><a name="p5522145964513"></a><a name="p5522145964513"></a>Check矢量读访问。</p>
<a name="ul934615561408"></a><a name="ul934615561408"></a><ul id="ul934615561408"><li>false：不开启，默认为false。</li><li>true：开启。</li></ul>
</td>
</tr>
<tr id="row159851016414"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p11568173463113"><a name="p11568173463113"></a><a name="p11568173463113"></a>isVectorWrite</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p14467154341914"><a name="p14467154341914"></a><a name="p14467154341914"></a>Check矢量写访问。</p>
<a name="ul3634123617"></a><a name="ul3634123617"></a><ul id="ul3634123617"><li>false：不开启，默认为false。</li><li>true：开启。</li></ul>
</td>
</tr>
<tr id="row1715510455472"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p7155114564712"><a name="p7155114564712"></a><a name="p7155114564712"></a>isMteRead</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p15155645184718"><a name="p15155645184718"></a><a name="p15155645184718"></a>Check Mte读访问。</p>
<a name="ul370715819"></a><a name="ul370715819"></a><ul id="ul370715819"><li>false：不开启，默认为false。</li><li>true：开启。</li></ul>
</td>
</tr>
<tr id="row86657494476"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p866610493478"><a name="p866610493478"></a><a name="p866610493478"></a>isMteWrite</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p123196511386"><a name="p123196511386"></a><a name="p123196511386"></a>Check Mte写访问。</p>
<a name="ul1889013711115"></a><a name="ul1889013711115"></a><ul id="ul1889013711115"><li>false：不开启，默认为false。</li><li>true：开启。</li></ul>
</td>
</tr>
<tr id="row8835135993514"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p2083665923512"><a name="p2083665923512"></a><a name="p2083665923512"></a>isEnable</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p18836135916359"><a name="p18836135916359"></a><a name="p18836135916359"></a>是否使能enableBit参数配置的异常寄存器。</p>
<a name="ul1094851015117"></a><a name="ul1094851015117"></a><ul id="ul1094851015117"><li>false：不使能，默认为false。</li><li>true：使能。</li></ul>
</td>
</tr>
<tr id="row1510111409485"><td class="cellrowborder" valign="top" width="15.229999999999999%" headers="mcps1.2.3.1.1 "><p id="p21011540194818"><a name="p21011540194818"></a><a name="p21011540194818"></a>reserved</p>
</td>
<td class="cellrowborder" valign="top" width="84.77%" headers="mcps1.2.3.1.2 "><p id="p1601381273"><a name="p1601381273"></a><a name="p1601381273"></a>预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   startAddr/endAddr的单位是32B，check的范围不包含startAddr，包含endAddr，即\(startAddr，endAddr\]。
-   每次调用完该接口需要进行复位（配置isEnable为false进行复位）；
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

该示例check矢量写访问是否在设定的\(startAddr, endAddr\]范围内。当前示例check到矢量写在设定的范围内，结果会报错（ACL\_ERROR\_RT\_VECTOR\_CORE\_EXCEPTION）。

```
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc0, inQueueSrc1;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst
pipe.InitBuffer(inQueueSrc0, 1, 512 * sizeof(half));
pipe.InitBuffer(inQueueSrc1, 1, 512 * sizeof(half));
pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
AscendC::LocalTensor<half> src0Local = inQueueSrc0.DeQue<half>();
AscendC::LocalTensor<half> src1Local = inQueueSrc1.DeQue<half>();
AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
AscendC::CheckLocalMemoryIA({ 0, (uint32_t)(dstLocal.GetPhyAddr() / 32),(uint32_t)((dstLocal.GetPhyAddr() + 512 * sizeof(half)) / 32), false, false, false, true, false, false,
true });
```

