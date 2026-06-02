# MmadWithSparse<a name="ZH-CN_TOPIC_0000002554344519"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵， 右矩阵B为稠密矩阵 。对于矩阵A，在MmadWithSparse计算时完成稠密化；对于矩阵B，在计算执行前的输入数据准备时自行完成稠密化（按照下文中介绍的稠密算法进行稠密化），所以输入本接口的B矩阵为稠密矩阵。B稠密矩阵需要通过调用[LoadDataWithSparse](LoadDataWithSparse.md)载入，同时加载索引矩阵，索引矩阵在矩阵B稠密化的过程中生成，再用于A矩阵的稠密化。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = int32_t, typename U = int8_t, typename Std::enable_if<Std::is_same<PrimT<T>, int32_t>::value, bool>::type = true, typename Std::enable_if<Std::is_same<PrimT<U>, int8_t>::value, bool>::type = true>
__aicore__ inline void MmadWithSparse(const LocalTensor<T>& dst, const LocalTensor<U>& fm, const LocalTensor<U>& filter, const MmadParams& mmadParams)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="13.34%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="86.66%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="13.34%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="86.66%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>dst的数据类型。</p>
</td>
</tr>
<tr id="row14306152011913"><td class="cellrowborder" valign="top" width="13.34%" headers="mcps1.2.3.1.1 "><p id="p1030652081918"><a name="p1030652081918"></a><a name="p1030652081918"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="86.66%" headers="mcps1.2.3.1.2 "><p id="p14306152041916"><a name="p14306152041916"></a><a name="p14306152041916"></a>fm、filter的数据类型。</p>
<a name="ul3987732122120"></a><a name="ul3987732122120"></a><ul id="ul3987732122120"><li>当dst、fm、filter为基础数据类型时， T必须为int32_t类型，U必须为int8_t类型，否则编译失败。</li></ul>
<a name="ul17416131483320"></a><a name="ul17416131483320"></a><ul id="ul17416131483320"><li>当dst、fm、filter为<a href="TensorTrait.md">TensorTrait</a>类型时，T的LiteType必须为int32_t类型，U的LiteType必须为int8_t类型，否则编译失败。</li></ul>
<p id="p882014184212"><a name="p882014184212"></a><a name="p882014184212"></a>最后两个模板参数仅用于上述数据类型检查，用户无需关注。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.35103510351035%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.98759875987598%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p19287714181617"><a name="p19287714181617"></a><a name="p19287714181617"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p192871614151615"><a name="p192871614151615"></a><a name="p192871614151615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p16287121461618"><a name="p16287121461618"></a><a name="p16287121461618"></a>目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。</p>
<p id="p5170152413011"><a name="p5170152413011"></a><a name="p5170152413011"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要256个元素（<span id="ph12315302272"><a name="ph12315302272"></a><a name="ph12315302272"></a>1024字节</span>）对齐。</span></p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>fm</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p0287191420164"><a name="p0287191420164"></a><a name="p0287191420164"></a>源操作数，左矩阵A，类型为LocalTensor，支持的TPosition为A2。</p>
<p id="p122771447172412"><a name="p122771447172412"></a><a name="p122771447172412"></a><span id="ph14913134718242"><a name="ph14913134718242"></a><a name="ph14913134718242"></a>LocalTensor的起始地址需要512字节对齐。</span></p>
</td>
</tr>
<tr id="row9486215111718"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1648712150175"><a name="p1648712150175"></a><a name="p1648712150175"></a>filter</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p19487171515178"><a name="p19487171515178"></a><a name="p19487171515178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p3487131516175"><a name="p3487131516175"></a><a name="p3487131516175"></a>源操作数，右矩阵B，类型为LocalTensor，支持的TPosition为B2。</p>
<p id="p104786272518"><a name="p104786272518"></a><a name="p104786272518"></a><span id="ph04781226252"><a name="ph04781226252"></a><a name="ph04781226252"></a>LocalTensor的起始地址需要512字节对齐。</span></p>
</td>
</tr>
<tr id="row1075785651510"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>mmadParams</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p152541610124614"><a name="p152541610124614"></a><a name="p152541610124614"></a>矩阵乘相关参数，类型为MmadParams。</p>
<p id="p17376814155615"><a name="p17376814155615"></a><a name="p17376814155615"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p12287014111614"><a name="p12287014111614"></a><a name="p12287014111614"></a>参数说明请参考<a href="Mmad.md#table15780447181917">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   原始稀疏矩阵B每4个元素中应保证最多2个非零元素，如果存在3个或更多非零元素，则仅使用前2个非零元素。
-   当M、K、N中的任意一个值为0时，该指令不会被执行。

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 稠密算法说明<a name="section19443175724215"></a>

假设原始稀疏矩阵B的每4个元素中至少有2个零，稠密化后的矩阵B是一个在每4个元素中过滤掉2个零的稠密矩阵。矩阵B稠密化的过程中生成索引矩阵，过程如下：对于稀疏矩阵B中的每4个元素，将在index矩阵中生成2个2位索引，并按照以下规则进行编码。索引必须在\{0, 1, 2\}范围内。

-   第一个索引用于指示前3个元素中第1个非零元素的相对位置。
-   第二个索引用于指示第2个非零元素在后3个元素中的相对位置。

具体可参考下表。其中，“-”表示算法不关心该位置上的值，因为其会被过滤。

<a name="table07754782315"></a>
<table><thead align="left"><tr id="row108341479237"><th class="cellrowborder" valign="top" width="23.212321232123216%" id="mcps1.1.8.1.1"><p id="p1583477172315"><a name="p1583477172315"></a><a name="p1583477172315"></a>示例</p>
</th>
<th class="cellrowborder" valign="top" width="10.491049104910491%" id="mcps1.1.8.1.2"><p id="p88342074230"><a name="p88342074230"></a><a name="p88342074230"></a>ele0</p>
</th>
<th class="cellrowborder" valign="top" width="10.491049104910491%" id="mcps1.1.8.1.3"><p id="p483419712316"><a name="p483419712316"></a><a name="p483419712316"></a>ele1</p>
</th>
<th class="cellrowborder" valign="top" width="9.150915091509152%" id="mcps1.1.8.1.4"><p id="p16834177122310"><a name="p16834177122310"></a><a name="p16834177122310"></a>ele2</p>
</th>
<th class="cellrowborder" valign="top" width="10.491049104910491%" id="mcps1.1.8.1.5"><p id="p48341274235"><a name="p48341274235"></a><a name="p48341274235"></a>ele3</p>
</th>
<th class="cellrowborder" valign="top" width="18.081808180818083%" id="mcps1.1.8.1.6"><p id="p28341732311"><a name="p28341732311"></a><a name="p28341732311"></a>Index_a[i]</p>
</th>
<th class="cellrowborder" valign="top" width="18.081808180818083%" id="mcps1.1.8.1.7"><p id="p68341571237"><a name="p68341571237"></a><a name="p68341571237"></a>Index_b[i]</p>
</th>
</tr>
</thead>
<tbody><tr id="row15834187202317"><td class="cellrowborder" rowspan="6" valign="top" width="23.212321232123216%" headers="mcps1.1.8.1.1 "><p id="p3834207112311"><a name="p3834207112311"></a><a name="p3834207112311"></a>Two non-zero elements</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.2 "><p id="p383417722319"><a name="p383417722319"></a><a name="p383417722319"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.3 "><p id="p19834117122313"><a name="p19834117122313"></a><a name="p19834117122313"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="9.150915091509152%" headers="mcps1.1.8.1.4 "><p id="p1483412715234"><a name="p1483412715234"></a><a name="p1483412715234"></a>X</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.5 "><p id="p383417152318"><a name="p383417152318"></a><a name="p383417152318"></a>Y</p>
</td>
<td class="cellrowborder" valign="top" width="18.081808180818083%" headers="mcps1.1.8.1.6 "><p id="p0834879233"><a name="p0834879233"></a><a name="p0834879233"></a>2’b10</p>
</td>
<td class="cellrowborder" valign="top" width="18.081808180818083%" headers="mcps1.1.8.1.7 "><p id="p88345712232"><a name="p88345712232"></a><a name="p88345712232"></a>2’b10</p>
</td>
</tr>
<tr id="row38341715232"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p683487192315"><a name="p683487192315"></a><a name="p683487192315"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p1983416719235"><a name="p1983416719235"></a><a name="p1983416719235"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p178346720235"><a name="p178346720235"></a><a name="p178346720235"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p98341377231"><a name="p98341377231"></a><a name="p98341377231"></a>Y</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p1383419742320"><a name="p1383419742320"></a><a name="p1383419742320"></a>2’b01</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p118341672238"><a name="p118341672238"></a><a name="p118341672238"></a>2’b10</p>
</td>
</tr>
<tr id="row1283537182312"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p78351711232"><a name="p78351711232"></a><a name="p78351711232"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p198359712237"><a name="p198359712237"></a><a name="p198359712237"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p4835137192314"><a name="p4835137192314"></a><a name="p4835137192314"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p18350712319"><a name="p18350712319"></a><a name="p18350712319"></a>Y</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p208354772313"><a name="p208354772313"></a><a name="p208354772313"></a>2’b00</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p88359712237"><a name="p88359712237"></a><a name="p88359712237"></a>2’b10</p>
</td>
</tr>
<tr id="row18351873238"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p148351577232"><a name="p148351577232"></a><a name="p148351577232"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p683587132318"><a name="p683587132318"></a><a name="p683587132318"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p1583514732311"><a name="p1583514732311"></a><a name="p1583514732311"></a>Y</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p98356713238"><a name="p98356713238"></a><a name="p98356713238"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p583527102311"><a name="p583527102311"></a><a name="p583527102311"></a>2’b01</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p128351175234"><a name="p128351175234"></a><a name="p128351175234"></a>2’b01</p>
</td>
</tr>
<tr id="row68357717233"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p17835117162318"><a name="p17835117162318"></a><a name="p17835117162318"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p48359711235"><a name="p48359711235"></a><a name="p48359711235"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p15835376236"><a name="p15835376236"></a><a name="p15835376236"></a>Y</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p78358712238"><a name="p78358712238"></a><a name="p78358712238"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p18835179231"><a name="p18835179231"></a><a name="p18835179231"></a>2’b00</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p6835276236"><a name="p6835276236"></a><a name="p6835276236"></a>2’b01</p>
</td>
</tr>
<tr id="row483515713236"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p168351476239"><a name="p168351476239"></a><a name="p168351476239"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p183519716231"><a name="p183519716231"></a><a name="p183519716231"></a>Y</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p5835375231"><a name="p5835375231"></a><a name="p5835375231"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p178359715234"><a name="p178359715234"></a><a name="p178359715234"></a>-</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p6835371234"><a name="p6835371234"></a><a name="p6835371234"></a>2’b00</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p108356742310"><a name="p108356742310"></a><a name="p108356742310"></a>2’b00</p>
</td>
</tr>
<tr id="row38351718232"><td class="cellrowborder" rowspan="4" valign="top" width="23.212321232123216%" headers="mcps1.1.8.1.1 "><p id="p18356782318"><a name="p18356782318"></a><a name="p18356782318"></a>One non-zero element</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.2 "><p id="p14835177122311"><a name="p14835177122311"></a><a name="p14835177122311"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.3 "><p id="p12835177233"><a name="p12835177233"></a><a name="p12835177233"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="9.150915091509152%" headers="mcps1.1.8.1.4 "><p id="p188355722312"><a name="p188355722312"></a><a name="p188355722312"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.5 "><p id="p383511711230"><a name="p383511711230"></a><a name="p383511711230"></a>X</p>
</td>
<td class="cellrowborder" valign="top" width="18.081808180818083%" headers="mcps1.1.8.1.6 "><p id="p1883657152313"><a name="p1883657152313"></a><a name="p1883657152313"></a>2’b00</p>
</td>
<td class="cellrowborder" valign="top" width="18.081808180818083%" headers="mcps1.1.8.1.7 "><p id="p198361719234"><a name="p198361719234"></a><a name="p198361719234"></a>2’b10</p>
</td>
</tr>
<tr id="row1983610762312"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p0836137112320"><a name="p0836137112320"></a><a name="p0836137112320"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p1083614792317"><a name="p1083614792317"></a><a name="p1083614792317"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p1183611711234"><a name="p1183611711234"></a><a name="p1183611711234"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p08361377230"><a name="p08361377230"></a><a name="p08361377230"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p208361071238"><a name="p208361071238"></a><a name="p208361071238"></a>2’b10</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p583647112313"><a name="p583647112313"></a><a name="p583647112313"></a>2’b00</p>
</td>
</tr>
<tr id="row283612742314"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p383657142311"><a name="p383657142311"></a><a name="p383657142311"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p78361474235"><a name="p78361474235"></a><a name="p78361474235"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p1483637172312"><a name="p1483637172312"></a><a name="p1483637172312"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p118368720239"><a name="p118368720239"></a><a name="p118368720239"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p168363719232"><a name="p168363719232"></a><a name="p168363719232"></a>2’b01</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p1783620717239"><a name="p1783620717239"></a><a name="p1783620717239"></a>2’b00</p>
</td>
</tr>
<tr id="row5836107162312"><td class="cellrowborder" valign="top" headers="mcps1.1.8.1.1 "><p id="p483617172310"><a name="p483617172310"></a><a name="p483617172310"></a>X</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.2 "><p id="p11836197172319"><a name="p11836197172319"></a><a name="p11836197172319"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.3 "><p id="p383618772313"><a name="p383618772313"></a><a name="p383618772313"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.4 "><p id="p3836207192310"><a name="p3836207192310"></a><a name="p3836207192310"></a>0</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.5 "><p id="p5836157132310"><a name="p5836157132310"></a><a name="p5836157132310"></a>2’b00</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.1.8.1.6 "><p id="p7836127132317"><a name="p7836127132317"></a><a name="p7836127132317"></a>2’b00</p>
</td>
</tr>
<tr id="row13836127132310"><td class="cellrowborder" valign="top" width="23.212321232123216%" headers="mcps1.1.8.1.1 "><p id="p5836278236"><a name="p5836278236"></a><a name="p5836278236"></a>All zero</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.2 "><p id="p10836197142315"><a name="p10836197142315"></a><a name="p10836197142315"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.3 "><p id="p1983619720233"><a name="p1983619720233"></a><a name="p1983619720233"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="9.150915091509152%" headers="mcps1.1.8.1.4 "><p id="p16836875233"><a name="p16836875233"></a><a name="p16836875233"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="10.491049104910491%" headers="mcps1.1.8.1.5 "><p id="p1836179237"><a name="p1836179237"></a><a name="p1836179237"></a>0</p>
</td>
<td class="cellrowborder" valign="top" width="18.081808180818083%" headers="mcps1.1.8.1.6 "><p id="p68366718231"><a name="p68366718231"></a><a name="p68366718231"></a>2’b00</p>
</td>
<td class="cellrowborder" valign="top" width="18.081808180818083%" headers="mcps1.1.8.1.7 "><p id="p16836079231"><a name="p16836079231"></a><a name="p16836079231"></a>2’b00</p>
</td>
</tr>
</tbody>
</table>

该索引矩阵用于A矩阵的稠密化，根据索引矩阵从MatrixA中的4个元素中选择2个元素参与计算，如下图所示：

<!-- img2text -->
```text
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  ele0   │  ele1   │  ele2   │  ele3   │  ele4   │  ele5   │  ele6   │  ele7   │
└────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘
     │         │         │         │         │         │         │         │
     │    ┌────▼────┐    │    ┌────▼────┐    │    ┌────▼────┐    │    ┌────▼────┐
     ├───►│ 0  1  2 │◄───┼───►│ 0  1  2 │◄───┼───►│ 0  1  2 │◄───┼───►│ 0  1  2 │
     │    └────┬────┘    │    └────┬────┘    │    └────┬────┘    │    └────┬────┘
     │     index[0]      │     index[1]      │     index[2]      │     index[3]
     │                   │                   │                   │
     │                   │                   │                   │
     │         ┌─────────▼─────────┬─────────▼─────────┬─────────▼─────────┬─────────▼─────────┐
     │         │      out[0]       │      out[1]       │      out[2]       │      out[3]       │
     └────────►└───────────────────┴───────────────────┴───────────────────┴───────────────────┘

连接关系:
ele0, ele1, ele2 → index[0]
ele1, ele2, ele3 → index[1]
ele4, ele5, ele6 → index[2]
ele5, ele6, ele7 → index[3]

index[0] → out[0]
index[1] → out[1]
index[1] → out[2]
index[2] → out[2]
index[2] → out[3]
index[3] → out[3]
```

## 调用示例<a name="section642mcpsimp"></a>

完整使用样例请参见[MmadWithSparse样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/01_matrix/mmad_with_sparse)。

```
#include "kernel_operator.h"
int srcOffset = 0;
int dstOffset = 0;
AscendC::LocalTensor<int8_t> a1Local = inQueueA1.DeQue<int8_t>();
AscendC::LocalTensor<int8_t> a2Local = inQueueA2.AllocTensor<int8_t>();

AscendC::LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = kBlocks * mBlocks;
loadDataParams.srcStride = 1;
loadDataParams.ifTranspose = false;

AscendC::LoadData(a2Local, a1Local, loadDataParams);

inQueueA2.EnQue<int8_t>(a2Local);
inQueueA1.FreeTensor(a1Local);

AscendC::LocalTensor<int8_t> b2Local = inQueueB2.AllocTensor<int8_t>();

// transform nz to zn
AscendC::LoadData2DParams loadDataParams;
loadDataParams.repeatTimes = kBlocks * nBlocks / 2;
loadDataParams.srcStride = 0;
loadDataParams.ifTranspose = false;

AscendC::LoadDataWithSparse(b2Local, b1Local, idxb1Local, loadDataParams);

inQueueB2.EnQue<int8_t>(b2Local);

AscendC::LocalTensor<int8_t> b2Local = inQueueB2.DeQue<int8_t>();
AscendC::LocalTensor<int32_t> c1Local = outQueueCO1.AllocTensor<int32_t>();

uint32 m = 16;
uint32 k = 64;
uint32 n = 16;
AscendC::MmadWithSparse(c1Local, a2Local, b2Local, { m, n, k, false, 0, false, false, false });
```

