# SetCtrlSpr\(ISASI\)<a name="ZH-CN_TOPIC_0000002554423795"></a>

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

对CTRL寄存器（控制寄存器）的特定比特位进行设置。

## 函数原型<a name="section620mcpsimp"></a>

```
template <int8_t startBit, int8_t endBit>
__aicore__ static inline void SetCtrlSpr(int64_t value)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="18.27%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.73%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="18.27%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>startBit</p>
</td>
<td class="cellrowborder" valign="top" width="81.73%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>起始比特位索引。</p>
</td>
</tr>
<tr id="row193420910534"><td class="cellrowborder" valign="top" width="18.27%" headers="mcps1.2.3.1.1 "><p id="p183417975310"><a name="p183417975310"></a><a name="p183417975310"></a>endBit</p>
</td>
<td class="cellrowborder" valign="top" width="81.73%" headers="mcps1.2.3.1.2 "><p id="p23412916537"><a name="p23412916537"></a><a name="p23412916537"></a>终止比特位索引。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.89%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.7%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.89%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>value</p>
</td>
<td class="cellrowborder" valign="top" width="10.7%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>起止比特位上新设置的值。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  常用CTRL寄存器比特位说明

<a name="table1660715170215"></a>
<table><thead align="left"><tr id="row16607417102114"><th class="cellrowborder" valign="top" width="17.68%" id="mcps1.2.5.1.1"><p id="p1760710179217"><a name="p1760710179217"></a><a name="p1760710179217"></a>CTRL寄存器比特位</p>
</th>
<th class="cellrowborder" valign="top" width="46.46%" id="mcps1.2.5.1.2"><p id="p10607131710212"><a name="p10607131710212"></a><a name="p10607131710212"></a>功能</p>
</th>
<th class="cellrowborder" valign="top" width="8.129999999999999%" id="mcps1.2.5.1.3"><p id="p1608717112119"><a name="p1608717112119"></a><a name="p1608717112119"></a>默认值</p>
</th>
<th class="cellrowborder" valign="top" width="27.73%" id="mcps1.2.5.1.4"><p id="p8316195315151"><a name="p8316195315151"></a><a name="p8316195315151"></a>配合使用的API</p>
</th>
</tr>
</thead>
<tbody><tr id="row3608617112116"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p196081617142115"><a name="p196081617142115"></a><a name="p196081617142115"></a>CTRL[8:6]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p78671445104116"><a name="p78671445104116"></a><a name="p78671445104116"></a><span>用于控制数据从L0C Buffer/Unified Buffer/L1 Buffer搬运至Global Memory时原子操作的使能及数据类型选择。</span></p>
<a name="ul10745131444510"></a><a name="ul10745131444510"></a><ul id="ul10745131444510"><li>3'b000：不使能原子操作；</li><li>3'b001：使能原子操作，数据类型为float；</li><li>3'b010：使能原子操作，数据类型为half；</li><li>3'b011：使能原子操作，数据类型为int16_t；</li><li>3'b100：使能原子操作，数据类型为int32_t；</li><li>3'b101：使能原子操作，数据类型为int8_t；</li><li>3'b110：使能原子操作，数据类型为bfloat16_t。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p9608121718218"><a name="p9608121718218"></a><a name="p9608121718218"></a>3'b000</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p13316135318153"><a name="p13316135318153"></a><a name="p13316135318153"></a>不涉及</p>
</td>
</tr>
<tr id="row18608717182115"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p1660817171216"><a name="p1660817171216"></a><a name="p1660817171216"></a>CTRL[10:9]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p1240518716248"><a name="p1240518716248"></a><a name="p1240518716248"></a><span>用于控制原子操作的类型，仅在CTRL[8:6]使能原子操作时生效。</span></p>
<a name="ul2413114819453"></a><a name="ul2413114819453"></a><ul id="ul2413114819453"><li>2'b00：选择ADD操作；</li><li>2'b01：选择MAX操作；</li><li>2'b10：选择MIN操作。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p860811176215"><a name="p860811176215"></a><a name="p860811176215"></a>2'b00</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p1031625315159"><a name="p1031625315159"></a><a name="p1031625315159"></a>不涉及</p>
</td>
</tr>
<tr id="row10586917104012"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p14668112512405"><a name="p14668112512405"></a><a name="p14668112512405"></a>CTRL[45]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p185872178403"><a name="p185872178403"></a><a name="p185872178403"></a>用于控制左右矩阵数据做Mmad计算时的处理方式。</p>
<a name="ul195931042174112"></a><a name="ul195931042174112"></a><ul id="ul195931042174112"><li>1'b0：按照原数据类型进行处理；</li><li>1'b1：左右矩阵数据均为fp8_e4m3fn_t时，数据视为hifloat8_t进行矩阵乘法计算。其他场景按照原数据类型进行处理。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p1270813334438"><a name="p1270813334438"></a><a name="p1270813334438"></a>1'b0</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p1947132194417"><a name="p1947132194417"></a><a name="p1947132194417"></a>不涉及</p>
</td>
</tr>
<tr id="row560871715215"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p182361236192111"><a name="p182361236192111"></a><a name="p182361236192111"></a>CTRL[48]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p4162174512467"><a name="p4162174512467"></a><a name="p4162174512467"></a><span>用于控制浮点数计算和浮点数精度转换时的饱和模式，仅在CTRL[60]使能时生效。</span></p>
<a name="ul1584405315476"></a><a name="ul1584405315476"></a><ul id="ul1584405315476"><li>1'b0：饱和模式，INF输出会被饱和为&plusmn;MAX， NaN输出会被饱和为0；</li><li>1'b1：非饱和模式，INF/NAN保持原输出。</li></ul>
<p id="p96554915494"><a name="p96554915494"></a><a name="p96554915494"></a>该控制位仅支持如下数据类型：</p>
<a name="ul106598917497"></a><a name="ul106598917497"></a><ul id="ul106598917497"><li>浮点数计算时支持half数据类型；</li><li>浮点数精度转换时支持如下数据类型：hifloat8_t、fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t、half、bfloat16_t。</li></ul>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p13608131742117"><a name="p13608131742117"></a><a name="p13608131742117"></a>1'b0</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p1595913426185"><a name="p1595913426185"></a><a name="p1595913426185"></a>配合使用的API：</p>
<a name="ul11961542161817"></a><a name="ul11961542161817"></a><ul id="ul11961542161817"><li>矢量计算API</li><li>原子操作API</li><li>精度转换指令</li></ul>
<p id="p1192712117193"><a name="p1192712117193"></a><a name="p1192712117193"></a>使用约束：</p>
<a name="ul1092115147198"></a><a name="ul1092115147198"></a><ul id="ul1092115147198"><li>需要满足数据类型限制。</li><li>执行原子操作过程中，如果需要重新配置该控制位，需要调用<a href="DataCacheCleanAndInvalid.md">DataCacheCleanAndInvalid</a>先清除当前Cache Line状态并将当前数据写出，防止饱和模式变更影响当前数据。具体调用示例可参考<a href="#li31971525192313">原子操作中，half类型配置全局非饱和模式示例。</a></li></ul>
</td>
</tr>
<tr id="row6608417152114"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p4793193912113"><a name="p4793193912113"></a><a name="p4793193912113"></a>CTRL[50]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p14895105819547"><a name="p14895105819547"></a><a name="p14895105819547"></a>用于控制浮点数精度转换时的NAN饱和模式，在CTRL[48]设置为饱和模式时生效。</p>
<p id="p123841545102313"><a name="p123841545102313"></a><a name="p123841545102313"></a>1'b0：NAN输出会被转换为0.0；</p>
<p id="p143843455234"><a name="p143843455234"></a><a name="p143843455234"></a>1'b1：NAN输出会保持NAN。</p>
<p id="p1192141319562"><a name="p1192141319562"></a><a name="p1192141319562"></a>该控制位仅支持如下数据类型：</p>
<p id="p144771653181917"><a name="p144771653181917"></a><a name="p144771653181917"></a>fp8_e8m0_t、fp8_e5m2_t、fp8_e4m3fn_t。</p>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p786415863820"><a name="p786415863820"></a><a name="p786415863820"></a>1'b0</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p327168122519"><a name="p327168122519"></a><a name="p327168122519"></a>精度转换指令（需要满足数据类型限制）。</p>
</td>
</tr>
<tr id="row6608191782111"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p19814342102116"><a name="p19814342102116"></a><a name="p19814342102116"></a>CTRL[53]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p138541157162310"><a name="p138541157162310"></a><a name="p138541157162310"></a><span>用于控制整数计算指令的饱和模式。</span></p>
<p id="p447421220593"><a name="p447421220593"></a><a name="p447421220593"></a>1'b0：截断模式，<span>溢出值按目标数据类型位数截断</span>，保留低位，舍弃高位；</p>
<p id="p17854125718234"><a name="p17854125718234"></a><a name="p17854125718234"></a>1'b1：饱和模式，溢出值饱和到&plusmn;MAX。</p>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p255715910385"><a name="p255715910385"></a><a name="p255715910385"></a>1'b0</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p157416467568"><a name="p157416467568"></a><a name="p157416467568"></a>矢量计算API（输入输出数据类型为整数）。</p>
</td>
</tr>
<tr id="row2608181772116"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p14301145182116"><a name="p14301145182116"></a><a name="p14301145182116"></a>CTRL[59]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p1253953213237"><a name="p1253953213237"></a><a name="p1253953213237"></a><span>用于控制浮点数转整数或整数转整数时的精度转换饱和模式，</span><span>仅在CTRL[60]使能时生效。</span></p>
<p id="p8290184953117"><a name="p8290184953117"></a><a name="p8290184953117"></a>1'b0：饱和模式：溢出值饱和到&plusmn;MAX；</p>
<p id="p1453915325233"><a name="p1453915325233"></a><a name="p1453915325233"></a>1'b1：截断模式：<span>溢出值按目标数据类型位数截断</span>，保留低位，舍弃高位。</p>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p11850170113913"><a name="p11850170113913"></a><a name="p11850170113913"></a>1'b0</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p02711842518"><a name="p02711842518"></a><a name="p02711842518"></a>精度转换指令。</p>
</td>
</tr>
<tr id="row166082173214"><td class="cellrowborder" valign="top" width="17.68%" headers="mcps1.2.5.1.1 "><p id="p4999154792118"><a name="p4999154792118"></a><a name="p4999154792118"></a>CTRL[60]</p>
</td>
<td class="cellrowborder" valign="top" width="46.46%" headers="mcps1.2.5.1.2 "><p id="p196618489316"><a name="p196618489316"></a><a name="p196618489316"></a><span>用于控制饱和模式的全局生效方式。</span></p>
<p id="p15891145142319"><a name="p15891145142319"></a><a name="p15891145142319"></a>1'b0：单指令设置饱和；</p>
<p id="p19891155122314"><a name="p19891155122314"></a><a name="p19891155122314"></a>1'b1：全局设置饱和。</p>
</td>
<td class="cellrowborder" valign="top" width="8.129999999999999%" headers="mcps1.2.5.1.3 "><p id="p45777114396"><a name="p45777114396"></a><a name="p45777114396"></a>1'b1</p>
</td>
<td class="cellrowborder" valign="top" width="27.73%" headers="mcps1.2.5.1.4 "><p id="p2671143015297"><a name="p2671143015297"></a><a name="p2671143015297"></a><span>该控制位可与Reg矢量计算API </span><a href="Cast-65.md">Cast</a><span>配合使用，或与CTRL[48]、CTRL[59]</span><span>配合使用，具体配置信息参考</span><a href="#table231122118201">表4</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  饱和模式全局或单指令生效配置表

<a name="table231122118201"></a>
<table><thead align="left"><tr id="row13111212208"><th class="cellrowborder" valign="top" width="17.580000000000002%" id="mcps1.2.4.1.1"><p id="p1531721192010"><a name="p1531721192010"></a><a name="p1531721192010"></a>全局使能位</p>
</th>
<th class="cellrowborder" valign="top" width="36.01%" id="mcps1.2.4.1.2"><p id="p12317213209"><a name="p12317213209"></a><a name="p12317213209"></a>控制位</p>
</th>
<th class="cellrowborder" valign="top" width="46.410000000000004%" id="mcps1.2.4.1.3"><p id="p831172114201"><a name="p831172114201"></a><a name="p831172114201"></a>功能描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row831182182017"><td class="cellrowborder" rowspan="2" valign="top" width="17.580000000000002%" headers="mcps1.2.4.1.1 "><p id="p17319219202"><a name="p17319219202"></a><a name="p17319219202"></a>CTRL[60] = 1'b0</p>
</td>
<td class="cellrowborder" valign="top" width="36.01%" headers="mcps1.2.4.1.2 "><p id="p94086452053"><a name="p94086452053"></a><a name="p94086452053"></a>Reg矢量计算Cast API的trait模板参数中satMode设置为SatMode::NO_SAT。</p>
</td>
<td class="cellrowborder" valign="top" width="46.410000000000004%" headers="mcps1.2.4.1.3 "><p id="p23118216209"><a name="p23118216209"></a><a name="p23118216209"></a>单指令非饱和模式。</p>
</td>
</tr>
<tr id="row168751913114217"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1189636133313"><a name="p1189636133313"></a><a name="p1189636133313"></a>Reg矢量计算Cast API的trait模板参数中satMode设置为</p>
<p id="p1989218228613"><a name="p1989218228613"></a><a name="p1989218228613"></a>SatMode::SAT。</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1487615130429"><a name="p1487615130429"></a><a name="p1487615130429"></a>单指令饱和模式。</p>
</td>
</tr>
<tr id="row1031142192013"><td class="cellrowborder" rowspan="4" valign="top" width="17.580000000000002%" headers="mcps1.2.4.1.1 "><p id="p1943154410414"><a name="p1943154410414"></a><a name="p1943154410414"></a>CTRL[60] = 1'b1</p>
</td>
<td class="cellrowborder" valign="top" width="36.01%" headers="mcps1.2.4.1.2 "><p id="p2233173311495"><a name="p2233173311495"></a><a name="p2233173311495"></a>CTRL[48] = 1'b1</p>
</td>
<td class="cellrowborder" valign="top" width="46.410000000000004%" headers="mcps1.2.4.1.3 "><p id="p9318219207"><a name="p9318219207"></a><a name="p9318219207"></a>全局非饱和模式（<span>浮点数计算和浮点数精度转换</span>）。</p>
</td>
</tr>
<tr id="row6560627174217"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p8560172717422"><a name="p8560172717422"></a><a name="p8560172717422"></a>CTRL[48] = 1'b0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7718171825020"><a name="p7718171825020"></a><a name="p7718171825020"></a>全局饱和模式（<span>浮点数计算和浮点数精度转换</span>）。</p>
</td>
</tr>
<tr id="row129291930134210"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p15929153012420"><a name="p15929153012420"></a><a name="p15929153012420"></a>CTRL[59] = 1'b1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p105203246501"><a name="p105203246501"></a><a name="p105203246501"></a>全局非饱和模式（<span>浮点数转整数或整数转整数时的精度转换</span>）。</p>
</td>
</tr>
<tr id="row4548334204218"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p13548153484212"><a name="p13548153484212"></a><a name="p13548153484212"></a>CTRL[59] = 1'b0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p921793617509"><a name="p921793617509"></a><a name="p921793617509"></a>全局饱和模式<span>（浮点数转整数或整数转整数时的精度转换）</span>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   仅支持CTRL\[8:6\]、CTRL\[10:9\]、CTRL\[45\]、CTRL\[48\]、CTRL\[50\]、CTRL\[53\]、CTRL\[59\]、CTRL\[60\]比特位。
-   对于CTRL\[8:6\]和CTRL\[10:9\]的设置，已封装原子操作API，建议通过这些原子操作API进行配置。
    -   [SetAtomicType](SetAtomicType.md)
    -   [DisableDmaAtomic](DisableDmaAtomic.md)
    -   [SetAtomicAdd](SetAtomicAdd.md)
    -   [SetAtomicMax](SetAtomicMax(ISASI).md)
    -   [SetAtomicMin](SetAtomicMin(ISASI).md)

## 调用示例<a name="section11279242185011"></a>

-   如下示例中使能原子操作模式，数据类型为float。

    ```
    SetCtrlSpr<6, 8>(1);
    ```

-   <a name="li31971525192313"></a>原子操作中，half类型配置全局非饱和模式示例。

    ```
    SetCtrlSpr<6, 8>(2);
    SetAtomicAdd<half>();
    DataCacheCleanAndInvalid<half, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_ATOMIC>(dstTensor);
    SetCtrlSpr<48, 48>(1);
    ...
    ```

