# SetFmatrix<a name="ZH-CN_TOPIC_0000002523343708"></a>

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
__aicore__ inline void SetFmatrix(uint16_t l1H, uint16_t l1W, const uint8_t padList[4], const FmatrixMode& fmatrixMode)
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
<tbody><tr id="row4956154125018"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p86321534103316"><a name="p86321534103316"></a><a name="p86321534103316"></a>l1H</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p3632334183313"><a name="p3632334183313"></a><a name="p3632334183313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p116321234123319"><a name="p116321234123319"></a><a name="p116321234123319"></a>源操作数height，取值范围：l1H∈[1, 32767]。</p>
</td>
</tr>
<tr id="row9486215111718"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p1633153423316"><a name="p1633153423316"></a><a name="p1633153423316"></a>l1W</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p176332345336"><a name="p176332345336"></a><a name="p176332345336"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p96331334153318"><a name="p96331334153318"></a><a name="p96331334153318"></a>源操作数width，取值范围：l1W∈[1, 32767] 。</p>
</td>
</tr>
<tr id="row3609135711444"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p4791358204412"><a name="p4791358204412"></a><a name="p4791358204412"></a>padList</p>
</td>
<td class="cellrowborder" valign="top" width="10.35103510351035%" headers="mcps1.2.4.1.2 "><p id="p1479117584448"><a name="p1479117584448"></a><a name="p1479117584448"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.98759875987598%" headers="mcps1.2.4.1.3 "><p id="p1379165819443"><a name="p1379165819443"></a><a name="p1379165819443"></a>padding列表 [padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。</p>
</td>
</tr>
<tr id="row1075785651510"><td class="cellrowborder" valign="top" width="13.661366136613662%" headers="mcps1.2.4.1.1 "><p id="p0633113433313"><a name="p0633113433313"></a><a name="p0633113433313"></a>fmatrixMode</p>
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

