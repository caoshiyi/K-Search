# LoadImageToLocal<a name="ZH-CN_TOPIC_0000002523344272"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.879999999999995%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.120000000000005%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.879999999999995%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.120000000000005%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将图像数据从Global Memory搬运到Local Memory。 搬运过程中可以完成图像预处理操作：包括图像翻转，改变图像尺寸（抠图，裁边，缩放，伸展），以及色域转换，类型转换等。图像预处理的相关参数通过[SetAippFunctions](SetAippFunctions.md)进行配置。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void LoadImageToLocal(const LocalTensor<T>& dst, const LoadImageToLocalParams& loadDataParams)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.371637163716375%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.341134113411341%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.28722872287229%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p13547261512"><a name="p13547261512"></a><a name="p13547261512"></a>目的操作数，类型为LocalTensor。</p>
<p id="p13212054173014"><a name="p13212054173014"></a><a name="p13212054173014"></a><span id="ph14913134718242"><a name="ph14913134718242"></a><a name="ph14913134718242"></a>LocalTensor的起始地址</span>需要保证32字节对齐。</p>
<p id="p592392813811"><a name="p592392813811"></a><a name="p592392813811"></a><span id="ph931120291384"><a name="ph931120291384"></a><a name="ph931120291384"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t、int8_t、half；支持的TPosition为VECIN、VECCALC、VECOUT。</p>
</td>
</tr>
<tr id="row1767431631917"><td class="cellrowborder" valign="top" width="16.371637163716375%" headers="mcps1.2.4.1.1 "><p id="p667418162198"><a name="p667418162198"></a><a name="p667418162198"></a>loadDataParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.341134113411341%" headers="mcps1.2.4.1.2 "><p id="p11675191610195"><a name="p11675191610195"></a><a name="p11675191610195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.28722872287229%" headers="mcps1.2.4.1.3 "><p id="p1732273213169"><a name="p1732273213169"></a><a name="p1732273213169"></a>LoadData参数结构体，类型为LoadImageToLocalParams。</p>
<p id="p395104375712"><a name="p395104375712"></a><a name="p395104375712"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p1667541617193"><a name="p1667541617193"></a><a name="p1667541617193"></a>参数说明参考<a href="#table8955841508">表2</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  LoadImageToLocalParams结构体内参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="16.46164616461646%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.29112911291129%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24722472247225%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p1855384918180"><a name="p1855384918180"></a><a name="p1855384918180"></a>horizSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p75532491189"><a name="p75532491189"></a><a name="p75532491189"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p35535499185"><a name="p35535499185"></a><a name="p35535499185"></a>从源图中加载图片的水平宽度，单位为像素，取值范围：horizSize∈[2, 4095] 。</p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p955315493182"><a name="p955315493182"></a><a name="p955315493182"></a>vertSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p755314991818"><a name="p755314991818"></a><a name="p755314991818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p15553949101816"><a name="p15553949101816"></a><a name="p15553949101816"></a>从源图中加载图片的垂直高度，单位为像素，取值范围：vertSize∈[2, 4095]。</p>
</td>
</tr>
<tr id="row11771625161812"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p055374920185"><a name="p055374920185"></a><a name="p055374920185"></a>horizStartPos</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p6553449121814"><a name="p6553449121814"></a><a name="p6553449121814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p65651740538"><a name="p65651740538"></a><a name="p65651740538"></a>加载图片在源图片上的水平起始地址，单位为像素，取值范围：horizStartPos∈[0, 4095] 。默认为0。</p>
<p id="p88061141137"><a name="p88061141137"></a><a name="p88061141137"></a>注意：当输入图片为YUV420SP、XRGB8888， RGB888和YUV400格式时，该参数需要是偶数。</p>
</td>
</tr>
<tr id="row93311227171815"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p1555320499185"><a name="p1555320499185"></a><a name="p1555320499185"></a>vertStartPos</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p1355314971811"><a name="p1355314971811"></a><a name="p1355314971811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p234518101766"><a name="p234518101766"></a><a name="p234518101766"></a>加载图片在源图片上的垂直起始地址，单位为像素，取值范围：vertStartPos∈[0, 4095] 。默认为0。</p>
<p id="p83171329446"><a name="p83171329446"></a><a name="p83171329446"></a>注意：当输入图片为YUV420SP格式时，该参数需要是偶数。</p>
</td>
</tr>
<tr id="row1321772919185"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p125531449181816"><a name="p125531449181816"></a><a name="p125531449181816"></a>srcHorizSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p105545496187"><a name="p105545496187"></a><a name="p105545496187"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p1130219124815"><a name="p1130219124815"></a><a name="p1130219124815"></a>源图像水平宽度 ，单位为像素，取值范围：srcHorizSize∈[2, 4095] 。</p>
<p id="p061124212720"><a name="p061124212720"></a><a name="p061124212720"></a>注意：当输入图片为YUV420SP格式时，该参数需要是偶数。</p>
</td>
</tr>
<tr id="row16697631171819"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p1555415492180"><a name="p1555415492180"></a><a name="p1555415492180"></a>topPadSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p15554164914188"><a name="p15554164914188"></a><a name="p15554164914188"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p835216479810"><a name="p835216479810"></a><a name="p835216479810"></a>目的图像顶部填充的像素数 ，取值范围：topPadSize∈[0, 32] ，默认为0。进行数据填充时使用，需要先调用<a href="SetAippFunctions.md">SetAippFunctions</a>通过<a href="SetAippFunctions.md#table8955841508">AippPaddingParams</a>配置填充的数值，再通过topPadSize、botPadSize、leftPadSize、rightPadSize配置填充的大小范围。</p>
</td>
</tr>
<tr id="row497391184"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p16554134921816"><a name="p16554134921816"></a><a name="p16554134921816"></a>botPadSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p19554114910183"><a name="p19554114910183"></a><a name="p19554114910183"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p155541749141820"><a name="p155541749141820"></a><a name="p155541749141820"></a>目的图像底部填充的像素数，取值范围：botPadSize∈[0, 32] ，默认为0。</p>
</td>
</tr>
<tr id="row113654148595"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p1536531455915"><a name="p1536531455915"></a><a name="p1536531455915"></a>leftPadSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p536531411596"><a name="p536531411596"></a><a name="p536531411596"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p1871912112910"><a name="p1871912112910"></a><a name="p1871912112910"></a>目的图像左边填充的像素数，取值范围：leftPadSize∈[0, 32] ，默认为0。</p>
</td>
</tr>
<tr id="row12646161755910"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p4646151745920"><a name="p4646151745920"></a><a name="p4646151745920"></a>rightPadSize</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p864681715594"><a name="p864681715594"></a><a name="p864681715594"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p95861946994"><a name="p95861946994"></a><a name="p95861946994"></a>目的图像右边填充的像素数，取值范围：rightPadSize∈[0, 32] ，默认为0。</p>
</td>
</tr>
<tr id="row1597316591537"><td class="cellrowborder" valign="top" width="16.46164616461646%" headers="mcps1.2.4.1.1 "><p id="p1535104110411"><a name="p1535104110411"></a><a name="p1535104110411"></a>sid</p>
</td>
<td class="cellrowborder" valign="top" width="11.29112911291129%" headers="mcps1.2.4.1.2 "><p id="p1797355916319"><a name="p1797355916319"></a><a name="p1797355916319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24722472247225%" headers="mcps1.2.4.1.3 "><p id="p6264205416479"><a name="p6264205416479"></a><a name="p6264205416479"></a>预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   加载到dst的图片的大小加padding的大小必须小于等于所在存储空间的大小。
-   当通过[SetAippFunctions](SetAippFunctions.md)配置padding模式为块填充模式或者镜像块填充模式时，因为padding的数据来自于抠出的图片，左右padding的长度（leftPadSize、rightPadSize）必须小于或等于抠图的水平长度（horizSize），上下padding的长度（topPadSize、botPadSize）必须小于或等于抠图的垂直的长度（vertSize）。

## 返回值说明<a name="section640mcpsimp"></a>

无

