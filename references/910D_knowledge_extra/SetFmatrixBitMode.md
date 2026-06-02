# SetFmatrixBitMode<a name="ZH-CN_TOPIC_0000002554344699"></a>

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

用于调用[Load3Dv1/Load3Dv2](LoadData.md)时设置FeatureMap的属性描述。Load3Dv1/Load3Dv2的模板参数isSetFMatrix设置为false时，表示Load3Dv1/Load3Dv2传入的FeatureMap的属性（包括l1H、l1W、padList，参数介绍参考[表4 LoadData3DParamsV1结构体内参数说明](Load3D.md#table679014222918)、[表5 LoadData3DParamsV2结构体内参数说明](Load3D.md#table193501032193419)）描述不生效，开发者需要通过该接口进行设置。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void SetFmatrix(const SetFMatrixBitModeParams& param, const FmatrixMode& fmatrixMode)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.661366136613662%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.35103510351035%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.98759875987598%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1075785651510"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p0633113433313"><a name="p0633113433313"></a><a name="p0633113433313"></a>fmatrixMode</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p206333347331"><a name="p206333347331"></a><a name="p206333347331"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p8416141817453"><a name="p8416141817453"></a><a name="p8416141817453"></a>用于控制LoadData指令从left还是right寄存器获取信息。FmatrixMode类型，定义如下。当前只支持FMATRIX_LEFT，左右矩阵均使用该配置。</p>
<a name="screen1488815152539"></a><a name="screen1488815152539"></a><pre class="screen" codetype="Cpp" id="screen1488815152539">enum class FmatrixMode : uint8_t {
    FMATRIX_LEFT = 0,
    FMATRIX_RIGHT = 1,
}; </pre>
</td>
</tr>
<tr id="row1268415574259"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1068419577255"><a name="p1068419577255"></a><a name="p1068419577255"></a>param</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p146851157152516"><a name="p146851157152516"></a><a name="p146851157152516"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p1968516574259"><a name="p1968516574259"></a><a name="p1968516574259"></a>类型为SetFMatrixBitMode，具体参考<a href="#table85031523118">表2</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  SetFMatrixBitMode类参数说明

<a name="table85031523118"></a>
<table><thead align="left"><tr id="row1750310523111"><th class="cellrowborder" valign="top" width="13.639999999999999%" id="mcps1.2.3.1.1"><p id="p135037521514"><a name="p135037521514"></a><a name="p135037521514"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="86.36%" id="mcps1.2.3.1.2"><p id="p850313521516"><a name="p850313521516"></a><a name="p850313521516"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row4503155219119"><td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.3.1.1 "><p id="p125037521110"><a name="p125037521110"></a><a name="p125037521110"></a>config0</p>
</td>
<td class="cellrowborder" valign="top" width="86.36%" headers="mcps1.2.3.1.2 "><p id="p25041852617"><a name="p25041852617"></a><a name="p25041852617"></a>uint64_t类型，与SetFMatrixBitModeConfig0位域（bit-field）结构体类型参数config0BitMode组成联合体（union），初始化为0，可以使用类对象的GetConfig0()函数获取其值。</p>
</td>
</tr>
<tr id="row195047520120"><td class="cellrowborder" valign="top" width="13.639999999999999%" headers="mcps1.2.3.1.1 "><p id="p1950414521912"><a name="p1950414521912"></a><a name="p1950414521912"></a>config0BitMode</p>
</td>
<td class="cellrowborder" valign="top" width="86.36%" headers="mcps1.2.3.1.2 "><p id="p19504552014"><a name="p19504552014"></a><a name="p19504552014"></a>SetFMatrixBitModeConfig0位域（bit-field）结构体类型，参数参考<a href="#table1162220101434">表3</a>，与config0组成联合体（union）。</p>
</td>
</tr>
</tbody>
</table>

SetFMatrixBitMode类参数设计思想说明：

联合体（union）是一种特殊的数据结构，允许在相同的内存位置存储不同的数据类型。union的所有成员共享同一块内存空间，大小由最大成员决定，同一时间只能使用一个成员。

位域（bit-field）是一种特殊的类成员，允许精确控制结构体中成员变量所占用的内存位数。结构体中成员变量从上到下对应内存中从低位到高位。

SetFMatrixBitMode类使用union与bit-field方法，采用bit位表达参数类型，使用bit-field结构体自动处理入参的bit位数，并利用union的特性实现多参数融合传递，仅需传递一个入参即可包含全部所需信息，对应底层接口仅需要接收一个参数。同时，当需要修改参数中某一bit位的值时，仅需要通过循环和位运算即可实现，不需要重新传入参数，减少了scalar计算，实现性能提升。

SetFMatrixBitMode类可以直接使用LoadData3DParamsV2结构体类型对象初始化：

```
template <typename T>
__aicore__ inline SetFMatrixBitModeParams(const LoadData3DParamsV2<T> &loadData3DParams_);
```

也可以使用各参数的Set函数修改参数值，并且由于使用了联合体，还可以对congfig0直接进行逐bit位修改来修改参数。

**表 3**  SetFMatrixBitModeConfig0结构体参数说明

<a name="table1162220101434"></a>
<table><thead align="left"><tr id="row1362218101636"><th class="cellrowborder" valign="top" width="13.59%" id="mcps1.2.3.1.1"><p id="p3622610938"><a name="p3622610938"></a><a name="p3622610938"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="86.41%" id="mcps1.2.3.1.2"><p id="p162220109316"><a name="p162220109316"></a><a name="p162220109316"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row17622171010314"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="p116222104312"><a name="p116222104312"></a><a name="p116222104312"></a>l1H</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="p66226105319"><a name="p66226105319"></a><a name="p66226105319"></a>源操作数height，取值范围：l1H∈[1, 32767]。</p>
<p id="p5493731155812"><a name="p5493731155812"></a><a name="p5493731155812"></a>该参数是位域结构体的最低位参数，占用16bit，可以使用SetFMatrixBitMode类对象的SetL1H()函数设置其值。</p>
</td>
</tr>
<tr id="row2622110633"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="p1262261015314"><a name="p1262261015314"></a><a name="p1262261015314"></a>l1W</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="p1362251018311"><a name="p1362251018311"></a><a name="p1362251018311"></a>源操作数width，取值范围：l1W∈[1, 32767] 。</p>
<p id="p144611270018"><a name="p144611270018"></a><a name="p144611270018"></a>该参数是位域结构体的第二低位参数，占用16bit，可以使用SetFMatrixBitMode类对象的SetL1W()函数设置其值。</p>
</td>
</tr>
<tr id="row106221310138"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="p96226100312"><a name="p96226100312"></a><a name="p96226100312"></a>padList0</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="p1162311020311"><a name="p1162311020311"></a><a name="p1162311020311"></a>对应<a href="SetFmatrix.md#table8955841508">表1</a>中padding列表中的 padding_left值，取值范围：[0,255]。默认为0。</p>
<p id="p4271630904"><a name="p4271630904"></a><a name="p4271630904"></a>该参数是位域结构体的第三低位参数，占用8bit，可以使用SetFMatrixBitMode类对象的SetPadList()函数设置其值。</p>
</td>
</tr>
<tr id="row1262310101733"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="p662316101032"><a name="p662316101032"></a><a name="p662316101032"></a>padList1</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="p136236103318"><a name="p136236103318"></a><a name="p136236103318"></a>对应<a href="SetFmatrix.md#table8955841508">表1</a>中padding列表中的 padding_right值，取值范围：[0,255]。默认为0。</p>
<p id="p1515652916113"><a name="p1515652916113"></a><a name="p1515652916113"></a>该参数是位域结构体的第四低位参数，占用8bit，可以使用SetFMatrixBitMode类对象的SetPadList()函数设置其值。</p>
</td>
</tr>
<tr id="row9623151012313"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="p562316101037"><a name="p562316101037"></a><a name="p562316101037"></a>padList2</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="p462351010315"><a name="p462351010315"></a><a name="p462351010315"></a>对应<a href="SetFmatrix.md#table8955841508">表1</a>中padding列表中的 padding_top值，取值范围：[0,255]。默认为0。</p>
<p id="p125213311115"><a name="p125213311115"></a><a name="p125213311115"></a>该参数是位域结构体的第五低位参数，占用8bit，可以使用SetFMatrixBitMode类对象的SetPadList()函数设置其值。</p>
</td>
</tr>
<tr id="row1362314108319"><td class="cellrowborder" valign="top" width="13.59%" headers="mcps1.2.3.1.1 "><p id="p1462316101734"><a name="p1462316101734"></a><a name="p1462316101734"></a>padList3</p>
</td>
<td class="cellrowborder" valign="top" width="86.41%" headers="mcps1.2.3.1.2 "><p id="p2062318102037"><a name="p2062318102037"></a><a name="p2062318102037"></a>对应<a href="SetFmatrix.md#table8955841508">表1</a>中padding列表中的 padding_bottom值，取值范围：[0,255]。默认为0。</p>
<p id="p321918350118"><a name="p321918350118"></a><a name="p321918350118"></a>该参数是位域结构体的最高位参数，占用8bit，可以使用SetFMatrixBitMode类对象的SetPadList()函数设置其值。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   该接口需要配合load3Dv1/load3Dv2接口一起使用，需要在load3Dv1/load3Dv2接口之前调用。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::TPipe pipe;

AscendC::TQue<AscendC::TPosition::A1, 1> inQueueFmA1;
AscendC::TQue<AscendC::TPosition::A2, 1> inQueueFmA2;
// weight queue
AscendC::TQue<AscendC::TPosition::B1, 1> inQueueWeB1;
AscendC::TQue<AscendC::TPosition::B2, 1> inQueueWeB2;
pipe.InitBuffer(inQueueFmA1, 1, featureMapA1Size * sizeof(fmap_T));
pipe.InitBuffer(inQueueFmA2, 1, featureMapA2Size * sizeof(fmap_T));
pipe.InitBuffer(inQueueWeB1, 1, weightA1Size * sizeof(weight_T));
pipe.InitBuffer(inQueueWeB2, 1, weightB2Size * sizeof(weight_T));
pipe.InitBuffer(outQueueCO1, 1, dstCO1Size * sizeof(dstCO1_T));

AscendC::LocalTensor<fmap_T> featureMapA1 = inQueueFmA1.DeQue<fmap_T>();
AscendC::LocalTensor<weight_T> weightB1 = inQueueWeB1.DeQue<weight_T>();
AscendC::LocalTensor<fmap_T> featureMapA2 = inQueueFmA2.AllocTensor<fmap_T>();
AscendC::LocalTensor<weight_T> weightB2 = inQueueWeB2.AllocTensor<weight_T>();
uint16_t channelSize = 32;
uint16_t H = 4, W = 4;
uint8_t Kh = 2, Kw = 2;
uint16_t Cout = 16;
uint16_t C0, C1;
uint8_t dilationH = 2, dilationW = 2;

uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
AscendC::SetFmatrix(H, W, padList, FmatrixMode::FMATRIX_LEFT);
/*  
SetFMatrixBitModeParams param;
param.SetL1H(H);
param.SetL1W(W);
param.SetPadList(padList);
AscendC::SetFmatrix(param, FmatrixMode::FMATRIX_LEFT);
*/ 
AscendC::SetLoadDataPaddingValue(0);
AscendC::SetLoadDataRepeat({0, 1, 0});
AscendC::SetLoadDataBoundary((uint32_t)0);
static constexpr AscendC::IsResetLoad3dConfig LOAD3D_CONFIG = {false,false};
AscendC::LoadData<fmap_T, LOAD3D_CONFIG>(featureMapA2, featureMapA1,
    { padList, H, W, channelSize, k, howoRound, 0, 0, 1, 1, Kw, Kh, dilationW, dilationH, false, false, 0 });
AscendC::LoadData(weightB2, weightB1, { 0, weRepeat, 1, 0, 0, false, 0 });

inQueueFmA2.EnQue<fmap_T>(featureMapA2);
inQueueWeB2.EnQue<weight_T>(weightB2);
inQueueFmA1.FreeTensor(featureMapA1);
inQueueWeB1.FreeTensor(weightB1);
```

