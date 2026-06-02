# Conv2D（废弃）<a name="ZH-CN_TOPIC_0000002554424873"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p173073381243"><a name="p173073381243"></a><a name="p173073381243"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

**该接口废弃，并将在后续版本移除，请不要使用该接口。**

计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2d卷积层多用于图像识别，使用过滤器提取图像中的特征。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, typename U>
__aicore__ inline void Conv2D(const LocalTensor<T>& dst, const LocalTensor<U>& featureMap, const LocalTensor<U>& weight, Conv2dParams& conv2dParams, Conv2dTilling& tilling)
```

入参中的tiling结构需要通过如下切分方案计算接口来获取：

```
template <typename T>
__aicore__ inline Conv2dTilling GetConv2dTiling(Conv2dParams& conv2dParams)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table1515192015127"></a>
<table><thead align="left"><tr id="row373122010120"><th class="cellrowborder" valign="top" width="18.47%" id="mcps1.2.4.1.1"><p id="p10731420121214"><a name="p10731420121214"></a><a name="p10731420121214"></a><strong id="b0731420101210"><a name="b0731420101210"></a><a name="b0731420101210"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="13.5%" id="mcps1.2.4.1.2"><p id="p1273162013120"><a name="p1273162013120"></a><a name="p1273162013120"></a><strong id="b1731120111216"><a name="b1731120111216"></a><a name="b1731120111216"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="68.03%" id="mcps1.2.4.1.3"><p id="p15734202129"><a name="p15734202129"></a><a name="p15734202129"></a><strong id="b117313202120"><a name="b117313202120"></a><a name="b117313202120"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row5731820181218"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.4.1.1 "><p id="p137342013124"><a name="p137342013124"></a><a name="p137342013124"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p167312207122"><a name="p167312207122"></a><a name="p167312207122"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.2.4.1.3 "><p id="p87372017121"><a name="p87372017121"></a><a name="p87372017121"></a>目的操作数。</p>
<p id="p342943218619"><a name="p342943218619"></a><a name="p342943218619"></a>结果中有效张量格式为[Cout/16, Ho, Wo, 16]，大小为Cout * Ho * Wo，Ho与Wo可以根据其他数据计算得出。</p>
<p id="p74291321568"><a name="p74291321568"></a><a name="p74291321568"></a>Ho = floor((H + pad_top + pad_bottom - dilation_h * (Kh - 1) - 1) / stride_h + 1)</p>
<p id="p242913218618"><a name="p242913218618"></a><a name="p242913218618"></a>Wo = floor((W + pad_left + pad_right - dilation_w * (Kw - 1) - 1) / stride_w + 1)</p>
<p id="p18429103213613"><a name="p18429103213613"></a><a name="p18429103213613"></a>由于硬件要求Ho*Wo需为16倍数，在申请dst Tensor时，shape应向上16对齐，实际申请shape大小应为Cout * round_howo。</p>
<p id="p14291732964"><a name="p14291732964"></a><a name="p14291732964"></a>round_howo = ceil(Ho * Wo /16) * 16。</p>
</td>
</tr>
<tr id="row107316202129"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.4.1.1 "><p id="p127382041211"><a name="p127382041211"></a><a name="p127382041211"></a>featureMap</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p57382015128"><a name="p57382015128"></a><a name="p57382015128"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.2.4.1.3 "><p id="p1373102011213"><a name="p1373102011213"></a><a name="p1373102011213"></a>输入张量，Tensor的TPosition为A1。</p>
<p id="p7282186779"><a name="p7282186779"></a><a name="p7282186779"></a>输入张量“feature_map”的形状，格式是<span>[C1, H, W, C0]。</span></p>
<p id="p2282661175"><a name="p2282661175"></a><a name="p2282661175"></a>C1*C0为输入的channel数，要求如下：</p>
<a name="ul2028218612716"></a><a name="ul2028218612716"></a><ul id="ul2028218612716"><li>当feature_map的数据类型为half时，C0=16。</li><li>当feature_map的数据类型为int8_t时，C0=32。</li><li>C1取值范围：[1,4], 输入的channel的范围：[16，32，64，128]。</li></ul>
<p id="p4282146779"><a name="p4282146779"></a><a name="p4282146779"></a>H为高，取值范围：[1,40]。</p>
<p id="p8282365712"><a name="p8282365712"></a><a name="p8282365712"></a>W为宽，取值范围：[1,40]。</p>
</td>
</tr>
<tr id="row173132020125"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.4.1.1 "><p id="p9735207125"><a name="p9735207125"></a><a name="p9735207125"></a>weight</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p9731720101211"><a name="p9731720101211"></a><a name="p9731720101211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.2.4.1.3 "><p id="p1373162051217"><a name="p1373162051217"></a><a name="p1373162051217"></a>卷积核（权重）张量，Tensor的TPosition为B1。</p>
<p id="p184064113711"><a name="p184064113711"></a><a name="p184064113711"></a>卷积核张量“weight”的形状，格式是[C1, Kh, Kw, Cout, C0]。</p>
<p id="p194016411873"><a name="p194016411873"></a><a name="p194016411873"></a>C1*C0为输入的channel数，对于C0要求如下：</p>
<a name="ul13401041679"></a><a name="ul13401041679"></a><ul id="ul13401041679"><li>当feature_map的数据类型为half时，C0=16。</li><li>当feature_map的数据类型为int8_t时，C0=32。</li><li>C1取值范围：[1,4]。</li><li>kernel_shape输入的channel数需与fm_shape输入的channel数保持一致。</li></ul>
<p id="p1440154112711"><a name="p1440154112711"></a><a name="p1440154112711"></a>Cout为卷积核数目，取值范围：[16，32，64，128]， Cout必须为16的倍数。</p>
<p id="p15403411278"><a name="p15403411278"></a><a name="p15403411278"></a>Kh为卷积核高；值的范围：[1,5]。</p>
<p id="p54054118713"><a name="p54054118713"></a><a name="p54054118713"></a>Kw表示卷积核宽；值的范围：[1,5]。</p>
</td>
</tr>
<tr id="row1573320151213"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.4.1.1 "><p id="p4741720181210"><a name="p4741720181210"></a><a name="p4741720181210"></a>conv2dParams</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p147492011220"><a name="p147492011220"></a><a name="p147492011220"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.2.4.1.3 "><p id="p7747203120"><a name="p7747203120"></a><a name="p7747203120"></a>输入矩阵形状等状态参数，类型为Conv2dParams。结构体具体定义为：</p>
<a name="screen12333123510548"></a><a name="screen12333123510548"></a><pre class="screen" codetype="Cpp" id="screen12333123510548">struct Conv2dParams {
    uint32_t imgShape[CONV2D_IMG_SIZE];       // [H, W]
    uint32_t kernelShapeIn[CONV2D_KERNEL_SIZE]; // [Kh, Kw]
    uint32_t stride[CONV2D_STRIDE];          // [stride_h, stride_w]
    uint32_t cin;                            // cin = C0 * C1;
    uint32_t cout;
    uint32_t padList[CONV2D_PAD];       // [pad_left, pad_right, pad_top, pad_bottom]
    uint32_t dilation[CONV2D_DILATION]; // [dilation_h, dilation_w]
    uint32_t initY;
    uint32_t partialSum;
};</pre>
</td>
</tr>
<tr id="row874520141218"><td class="cellrowborder" valign="top" width="18.47%" headers="mcps1.2.4.1.1 "><p id="p1697485617123"><a name="p1697485617123"></a><a name="p1697485617123"></a>tilling</p>
</td>
<td class="cellrowborder" valign="top" width="13.5%" headers="mcps1.2.4.1.2 "><p id="p35488535120"><a name="p35488535120"></a><a name="p35488535120"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.03%" headers="mcps1.2.4.1.3 "><p id="p3529145318127"><a name="p3529145318127"></a><a name="p3529145318127"></a>分形控制参数，类型为Conv2dTilling。结构体具体定义为：</p>
<a name="screen8348103082311"></a><a name="screen8348103082311"></a><pre class="screen" codetype="Cpp" id="screen8348103082311">struct Conv2dTilling {
    const uint32_t blockSize = 16; // # M block size is always 16
    LoopMode loopMode = LoopMode::MODE_NM;

    uint32_t c0Size = 32;
    uint32_t dTypeSize = 1;

    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t hi = 0;
    uint32_t wi = 0;
    uint32_t ho = 0;
    uint32_t wo = 0;

    uint32_t height = 0;
    uint32_t width = 0;

    uint32_t howo = 0;

    uint32_t mNum = 0;
    uint32_t nNum = 0;
    uint32_t kNum = 0;

    uint32_t mBlockNum = 0;
    uint32_t kBlockNum = 0;
    uint32_t nBlockNum = 0;

    uint32_t roundM = 0;
    uint32_t roundN = 0;
    uint32_t roundK = 0;

    uint32_t mTileBlock = 0;
    uint32_t nTileBlock = 0;
    uint32_t kTileBlock = 0;

    uint32_t mIterNum = 0;
    uint32_t nIterNum = 0;
    uint32_t kIterNum = 0;

    uint32_t mTileNums = 0;

    bool mHasTail = false;
    bool nHasTail = false;
    bool kHasTail = false;

    uint32_t kTailBlock = 0;
    uint32_t mTailBlock = 0;
    uint32_t nTailBlock = 0;

    uint32_t mTailNums = 0;
};</pre>
</td>
</tr>
</tbody>
</table>

**表 2**  Conv2DParams结构体内参数说明：

<a name="table1893416475138"></a>
<table><thead align="left"><tr id="row397694741317"><th class="cellrowborder" valign="top" width="15.629999999999999%" id="mcps1.2.4.1.1"><p id="p1997654721313"><a name="p1997654721313"></a><a name="p1997654721313"></a><strong id="b1097694751315"><a name="b1097694751315"></a><a name="b1097694751315"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="14.2%" id="mcps1.2.4.1.2"><p id="p1897644791317"><a name="p1897644791317"></a><a name="p1897644791317"></a><strong id="b99761147131313"><a name="b99761147131313"></a><a name="b99761147131313"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="70.17%" id="mcps1.2.4.1.3"><p id="p1297616472134"><a name="p1297616472134"></a><a name="p1297616472134"></a><strong id="b12976204715134"><a name="b12976204715134"></a><a name="b12976204715134"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row10976447111312"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p6976144721314"><a name="p6976144721314"></a><a name="p6976144721314"></a>imgShape</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p5976047161310"><a name="p5976047161310"></a><a name="p5976047161310"></a>vector&lt;int&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><div class="p" id="p897664791317"><a name="p897664791317"></a><a name="p897664791317"></a>输入张量“feature_map”的形状，格式是[ H, W]。<a name="ul19340130112112"></a><a name="ul19340130112112"></a><ul id="ul19340130112112"><li>H为高，取值范围：[1,40]。</li><li>W为宽，取值范围：[1,40]。</li></ul>
</div>
</td>
</tr>
<tr id="row3976347191313"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p3976134710133"><a name="p3976134710133"></a><a name="p3976134710133"></a>kernelShape</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p597664712135"><a name="p597664712135"></a><a name="p597664712135"></a>vector&lt;int&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><p id="p12976204719135"><a name="p12976204719135"></a><a name="p12976204719135"></a>卷积核张量“weight”的形状，格式是[Kh, Kw]。</p>
<a name="ul6807512173415"></a><a name="ul6807512173415"></a><ul id="ul6807512173415"><li>Kh为高，取值范围：[1,5]。</li><li>Kw为宽，取值范围：[1,5]。</li></ul>
</td>
</tr>
<tr id="row9976154718139"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p1976247131311"><a name="p1976247131311"></a><a name="p1976247131311"></a>stride</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p119761147101311"><a name="p119761147101311"></a><a name="p119761147101311"></a>vector&lt;int&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><div class="p" id="p1797611477139"><a name="p1797611477139"></a><a name="p1797611477139"></a>卷积步长，格式是[stride_h, stride_w]。<a name="ul6783195819212"></a><a name="ul6783195819212"></a><ul id="ul6783195819212"><li>stride_h表示步长高， 值的范围：[1,4]。</li><li>stride_w表示步长宽， 值的范围：[1,4]。</li></ul>
</div>
</td>
</tr>
<tr id="row12976164710133"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p797644719134"><a name="p797644719134"></a><a name="p797644719134"></a>cin</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p109761479137"><a name="p109761479137"></a><a name="p109761479137"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><p id="p4977747121317"><a name="p4977747121317"></a><a name="p4977747121317"></a>分形排布参数，Cin = C1 * C0，Cin为输入的channel数，C1取值范围：[1,4]。</p>
<a name="ul96071551115711"></a><a name="ul96071551115711"></a><ul id="ul96071551115711"><li>当feature_map的数据类型为float时，C0=8。输入的channel的范围：[8，16，24，32]。</li><li>当feature_map的数据类型为half时，C0=16。输入的channel的范围：[16，32，48，64]。</li><li>当feature_map的数据类型为int8_t时，C0=32。输入的channel的范围：[32，64，96，128]。</li></ul>
</td>
</tr>
<tr id="row189771347131315"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p17977247181310"><a name="p17977247181310"></a><a name="p17977247181310"></a>cout</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p997712475137"><a name="p997712475137"></a><a name="p997712475137"></a>int</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><p id="p1971119142593"><a name="p1971119142593"></a><a name="p1971119142593"></a>Cout为卷积核数目，取值范围：[16，32，64，128]， Cout必须为16的倍数。</p>
</td>
</tr>
<tr id="row1897710473138"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p29771247131318"><a name="p29771247131318"></a><a name="p29771247131318"></a>padList</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p119772477139"><a name="p119772477139"></a><a name="p119772477139"></a>vector&lt;int&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><div class="p" id="p03953523599"><a name="p03953523599"></a><a name="p03953523599"></a>padding行数/列数，格式是[pad_left, pad_right, pad_top, pad_bottom]。<a name="ul4214157182413"></a><a name="ul4214157182413"></a><ul id="ul4214157182413"><li>pad_left为feature_map左侧pad列数，范围[0,4]。pad_right为feature_map右侧pad列数，范围[0,4]。</li><li>pad_top为feature_map顶部pad行数，范围[0,4]。</li><li>pad_bottom为feature_map底部pad行数，范围[0,4]。</li></ul>
</div>
</td>
</tr>
<tr id="row1597724761312"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p1197784791315"><a name="p1197784791315"></a><a name="p1197784791315"></a>dilation</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p4977114761310"><a name="p4977114761310"></a><a name="p4977114761310"></a>vector&lt;int&gt;</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><div class="p" id="p1648920514017"><a name="p1648920514017"></a><a name="p1648920514017"></a>空洞卷积参数，格式[dilation_h, dilation_w]。<a name="ul20221143313252"></a><a name="ul20221143313252"></a><ul id="ul20221143313252"><li>dilation_h为空洞高，范围：[1,4]。</li><li>dilation_w为空洞宽，范围：[1,4]。</li></ul>
</div>
<p id="p3489205506"><a name="p3489205506"></a><a name="p3489205506"></a>膨胀后卷积核宽为dilation_w * (Kw - 1) + 1，高为dilation_h * (Kh - 1) + 1。</p>
</td>
</tr>
<tr id="row097714711314"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p159779477137"><a name="p159779477137"></a><a name="p159779477137"></a>initY</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p16977164731318"><a name="p16977164731318"></a><a name="p16977164731318"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><p id="p139771247141311"><a name="p139771247141311"></a><a name="p139771247141311"></a>表示dst是否需要初始化。</p>
<a name="ul1482311102619"></a><a name="ul1482311102619"></a><ul id="ul1482311102619"><li>取值0：不使用bias，L0C需要初始化，dst初始矩阵保存有之前结果，新计算结果会累加前一次conv2d计算结果。</li><li>取值1：不使用bias，L0C不需要初始化，dst初始矩阵中数据无意义，计算结果直接覆盖dst中的数据。</li></ul>
</td>
</tr>
<tr id="row29771847161312"><td class="cellrowborder" valign="top" width="15.629999999999999%" headers="mcps1.2.4.1.1 "><p id="p1597724715135"><a name="p1597724715135"></a><a name="p1597724715135"></a>partialSum</p>
</td>
<td class="cellrowborder" valign="top" width="14.2%" headers="mcps1.2.4.1.2 "><p id="p106701032312"><a name="p106701032312"></a><a name="p106701032312"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="70.17%" headers="mcps1.2.4.1.3 "><div class="p" id="p18202154220519"><a name="p18202154220519"></a><a name="p18202154220519"></a>当dst参数所在的TPosition为CO2时，通过该参数控制计算结果是否搬出。<a name="ul89871568272"></a><a name="ul89871568272"></a><ul id="ul89871568272"><li>取值0：搬出计算结果</li><li>取值1：不搬出计算结果，可以进行后续计算</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

**表 3**  Conv2dTilling结构体内参数说明

<a name="table873439203818"></a>
<table><thead align="left"><tr id="row178072943820"><th class="cellrowborder" valign="top" width="20.61%" id="mcps1.2.4.1.1"><p id="p1280711910381"><a name="p1280711910381"></a><a name="p1280711910381"></a><strong id="b7808696387"><a name="b7808696387"></a><a name="b7808696387"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.2"><p id="p1480816918386"><a name="p1480816918386"></a><a name="p1480816918386"></a><strong id="b980879153814"><a name="b980879153814"></a><a name="b980879153814"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="64.4%" id="mcps1.2.4.1.3"><p id="p168081296385"><a name="p168081296385"></a><a name="p168081296385"></a><strong id="b480819912387"><a name="b480819912387"></a><a name="b480819912387"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row128083918386"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p158083943817"><a name="p158083943817"></a><a name="p158083943817"></a>blockSize</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p128081195388"><a name="p128081195388"></a><a name="p128081195388"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1680811983820"><a name="p1680811983820"></a><a name="p1680811983820"></a>固定值，恒为16，一个维度内存放的元素个数。</p>
</td>
</tr>
<tr id="row108081495383"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1080810923814"><a name="p1080810923814"></a><a name="p1080810923814"></a>loopMode</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1280899143818"><a name="p1280899143818"></a><a name="p1280899143818"></a>LoopMode</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p880829103813"><a name="p880829103813"></a><a name="p880829103813"></a>遍历模式，结构体具体定义为：</p>
<a name="screen1670614014719"></a><a name="screen1670614014719"></a><pre class="screen" codetype="Cpp" id="screen1670614014719">enum class LoopMode {
    MODE_NM = 0,
    MODE_MN = 1,
    MODE_KM = 2,
    MODE_KN = 3
};</pre>
</td>
</tr>
<tr id="row11808796386"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1796910312429"><a name="p1796910312429"></a><a name="p1796910312429"></a>c0Size</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p2096833134217"><a name="p2096833134217"></a><a name="p2096833134217"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p109671431174217"><a name="p109671431174217"></a><a name="p109671431174217"></a>一个block的字节长度，范围[16或者32]。</p>
</td>
</tr>
<tr id="row280810910383"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1996717311428"><a name="p1996717311428"></a><a name="p1996717311428"></a>dtypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p8966123114219"><a name="p8966123114219"></a><a name="p8966123114219"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p096653110425"><a name="p096653110425"></a><a name="p096653110425"></a>传入的数据类型的字节长度，范围[1, 2]。</p>
</td>
</tr>
<tr id="row181751352154318"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p96171285112"><a name="p96171285112"></a><a name="p96171285112"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1239911585115"><a name="p1239911585115"></a><a name="p1239911585115"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p817511521436"><a name="p817511521436"></a><a name="p817511521436"></a>卷积步长-高，范围:[1,4]。</p>
</td>
</tr>
<tr id="row17723111144420"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1649551875112"><a name="p1649551875112"></a><a name="p1649551875112"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p144147157516"><a name="p144147157516"></a><a name="p144147157516"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1372351144415"><a name="p1372351144415"></a><a name="p1372351144415"></a>卷积步长-宽，范围:[1,4]。</p>
</td>
</tr>
<tr id="row19871135914319"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1758242035115"><a name="p1758242035115"></a><a name="p1758242035115"></a>dilationH</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1042811515119"><a name="p1042811515119"></a><a name="p1042811515119"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p158711359114311"><a name="p158711359114311"></a><a name="p158711359114311"></a>空洞卷积参数-高，范围：[1,4]。</p>
</td>
</tr>
<tr id="row17585202215115"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p619122595117"><a name="p619122595117"></a><a name="p619122595117"></a>dilationW</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p16131373524"><a name="p16131373524"></a><a name="p16131373524"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p14586322125112"><a name="p14586322125112"></a><a name="p14586322125112"></a>空洞卷积参数-宽，范围：[1,4]。</p>
</td>
</tr>
<tr id="row528327135120"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1170530125115"><a name="p1170530125115"></a><a name="p1170530125115"></a>hi</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p152877125219"><a name="p152877125219"></a><a name="p152877125219"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p19289278511"><a name="p19289278511"></a><a name="p19289278511"></a>feature_map形状-高，范围：[1,40]。</p>
</td>
</tr>
<tr id="row55051832145119"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1875913535110"><a name="p1875913535110"></a><a name="p1875913535110"></a>wi</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p232187155211"><a name="p232187155211"></a><a name="p232187155211"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p042133214553"><a name="p042133214553"></a><a name="p042133214553"></a>feature_map形状-宽，范围：[1,40]。</p>
</td>
</tr>
<tr id="row2470239165114"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p99457426516"><a name="p99457426516"></a><a name="p99457426516"></a>ho</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p53657125220"><a name="p53657125220"></a><a name="p53657125220"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1642963255512"><a name="p1642963255512"></a><a name="p1642963255512"></a>feature_map形状-高，范围：[1,40]。</p>
</td>
</tr>
<tr id="row1744204515519"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1191554716514"><a name="p1191554716514"></a><a name="p1191554716514"></a>wo</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1740774526"><a name="p1740774526"></a><a name="p1740774526"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1131516338552"><a name="p1131516338552"></a><a name="p1131516338552"></a>feature_map形状-宽，范围：[1,40]。</p>
</td>
</tr>
<tr id="row721816505517"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p5541155375110"><a name="p5541155375110"></a><a name="p5541155375110"></a>height</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p11459713523"><a name="p11459713523"></a><a name="p11459713523"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p20219135010519"><a name="p20219135010519"></a><a name="p20219135010519"></a>weight形状-高，[1,5]。</p>
</td>
</tr>
<tr id="row746895517511"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p105982575516"><a name="p105982575516"></a><a name="p105982575516"></a>width</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p9498716525"><a name="p9498716525"></a><a name="p9498716525"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p43395201559"><a name="p43395201559"></a><a name="p43395201559"></a>weight形状-宽，[1,5]。</p>
</td>
</tr>
<tr id="row33717596510"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1873211212528"><a name="p1873211212528"></a><a name="p1873211212528"></a>howo</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p65377105215"><a name="p65377105215"></a><a name="p65377105215"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p14625816367"><a name="p14625816367"></a><a name="p14625816367"></a>feature_map形状大小，为ho * wo。</p>
</td>
</tr>
<tr id="row1080915963814"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p10808179143811"><a name="p10808179143811"></a><a name="p10808179143811"></a>mNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1680815983812"><a name="p1680815983812"></a><a name="p1680815983812"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p13808209153810"><a name="p13808209153810"></a><a name="p13808209153810"></a>M轴等效数据长度参数值，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row78091491388"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p17808394383"><a name="p17808394383"></a><a name="p17808394383"></a>nNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p128087917386"><a name="p128087917386"></a><a name="p128087917386"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p7808090384"><a name="p7808090384"></a><a name="p7808090384"></a>N轴等效数据长度参数值，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row1880918993812"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1480911963810"><a name="p1480911963810"></a><a name="p1480911963810"></a>kNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p128092933813"><a name="p128092933813"></a><a name="p128092933813"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p880949103811"><a name="p880949103811"></a><a name="p880949103811"></a>K轴等效数据长度参数值，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row88091293380"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p880918920380"><a name="p880918920380"></a><a name="p880918920380"></a>roundM</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p108096993819"><a name="p108096993819"></a><a name="p108096993819"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p680912915384"><a name="p680912915384"></a><a name="p680912915384"></a>M轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row98091496385"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p58091918382"><a name="p58091918382"></a><a name="p58091918382"></a>roundN</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p38093918389"><a name="p38093918389"></a><a name="p38093918389"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1180919193810"><a name="p1180919193810"></a><a name="p1180919193810"></a>N轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row1880912916387"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1780917912385"><a name="p1780917912385"></a><a name="p1780917912385"></a>roundK</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p6809997389"><a name="p6809997389"></a><a name="p6809997389"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p108098911384"><a name="p108098911384"></a><a name="p108098911384"></a>K轴等效数据长度参数值且以c0Size为倍数向上取整，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row580959153816"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p88094912387"><a name="p88094912387"></a><a name="p88094912387"></a>mBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p58098915388"><a name="p58098915388"></a><a name="p58098915388"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p12809189203814"><a name="p12809189203814"></a><a name="p12809189203814"></a>M轴Block个数，mBlockNum = mNum / blockSize，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row0809189103820"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p38093933819"><a name="p38093933819"></a><a name="p38093933819"></a>nBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p4809189183811"><a name="p4809189183811"></a><a name="p4809189183811"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p680999123812"><a name="p680999123812"></a><a name="p680999123812"></a>N轴Block个数，nBlockNum = nNum / blockSize，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row1980919933812"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p168091298383"><a name="p168091298383"></a><a name="p168091298383"></a>kBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1180914914386"><a name="p1180914914386"></a><a name="p1180914914386"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p380999143819"><a name="p380999143819"></a><a name="p380999143819"></a>K轴Block个数，kBlockNum = kNum / blockSize，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row178106911383"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p5810592381"><a name="p5810592381"></a><a name="p5810592381"></a>mIterNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1881019912388"><a name="p1881019912388"></a><a name="p1881019912388"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p2029613115483"><a name="p2029613115483"></a><a name="p2029613115483"></a>遍历M轴维度数量，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row381029113820"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p38101694388"><a name="p38101694388"></a><a name="p38101694388"></a>nIterNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p8810169133812"><a name="p8810169133812"></a><a name="p8810169133812"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1681069193812"><a name="p1681069193812"></a><a name="p1681069193812"></a>遍历N轴维度数量，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row6810999388"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p48106915383"><a name="p48106915383"></a><a name="p48106915383"></a>kIterNum</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1810291383"><a name="p1810291383"></a><a name="p1810291383"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p181039183818"><a name="p181039183818"></a><a name="p181039183818"></a>遍历K轴维度数量，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row1381011943817"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p781014993814"><a name="p781014993814"></a><a name="p781014993814"></a>mTileBlock</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p108102916384"><a name="p108102916384"></a><a name="p108102916384"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p7810109103813"><a name="p7810109103813"></a><a name="p7810109103813"></a>M轴切分块个数，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row7810189163816"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p19810496381"><a name="p19810496381"></a><a name="p19810496381"></a>nTileBlock</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p178109918383"><a name="p178109918383"></a><a name="p178109918383"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p118108910388"><a name="p118108910388"></a><a name="p118108910388"></a>N轴切分块个数，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row981029123813"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p281010916381"><a name="p281010916381"></a><a name="p281010916381"></a>kTileBlock</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p10810109113814"><a name="p10810109113814"></a><a name="p10810109113814"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p1181017911384"><a name="p1181017911384"></a><a name="p1181017911384"></a>K轴切分块个数，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row178101498381"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p381099133817"><a name="p381099133817"></a><a name="p381099133817"></a>kTailBlock</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p581010912380"><a name="p581010912380"></a><a name="p581010912380"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p168101391384"><a name="p168101391384"></a><a name="p168101391384"></a>K轴尾块个数，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row981018953819"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1381012915386"><a name="p1381012915386"></a><a name="p1381012915386"></a>mTailBlock</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p3810189103813"><a name="p3810189103813"></a><a name="p3810189103813"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p8810159103815"><a name="p8810159103815"></a><a name="p8810159103815"></a>M轴尾块个数，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row5810159173813"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p581010916382"><a name="p581010916382"></a><a name="p581010916382"></a>nTailBlock</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p198101092384"><a name="p198101092384"></a><a name="p198101092384"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p481019113817"><a name="p481019113817"></a><a name="p481019113817"></a>N轴尾块个数，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row1281099123818"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p1181120916388"><a name="p1181120916388"></a><a name="p1181120916388"></a>kHasTail</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p168116973812"><a name="p168116973812"></a><a name="p168116973812"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p281129183814"><a name="p281129183814"></a><a name="p281129183814"></a>K轴是否存在尾块。</p>
</td>
</tr>
<tr id="row1281117918388"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p28117943819"><a name="p28117943819"></a><a name="p28117943819"></a>mHasTail</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p2811892386"><a name="p2811892386"></a><a name="p2811892386"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p208113918385"><a name="p208113918385"></a><a name="p208113918385"></a>M轴是否存在尾块。</p>
</td>
</tr>
<tr id="row581117943815"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p138111915388"><a name="p138111915388"></a><a name="p138111915388"></a>nHasTail</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p8811995387"><a name="p8811995387"></a><a name="p8811995387"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p178112983815"><a name="p178112983815"></a><a name="p178112983815"></a>N轴是否存在尾块。</p>
</td>
</tr>
<tr id="row681119918381"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p531826165718"><a name="p531826165718"></a><a name="p531826165718"></a>mTileNums</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p1580815567561"><a name="p1580815567561"></a><a name="p1580815567561"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p13821173911576"><a name="p13821173911576"></a><a name="p13821173911576"></a>M轴切分块个数的长度，范围：[1,4096]。</p>
</td>
</tr>
<tr id="row88116919385"><td class="cellrowborder" valign="top" width="20.61%" headers="mcps1.2.4.1.1 "><p id="p45609812575"><a name="p45609812575"></a><a name="p45609812575"></a>mTailNums</p>
</td>
<td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.2 "><p id="p5806135685619"><a name="p5806135685619"></a><a name="p5806135685619"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="64.4%" headers="mcps1.2.4.1.3 "><p id="p7150185945713"><a name="p7150185945713"></a><a name="p7150185945713"></a>M轴尾块个数的长度，范围：[1,4096]。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  imgShape、kernelShape和dst的数据类型组合

<a name="table7518203512413"></a>
<table><thead align="left"><tr id="row13531103532418"><th class="cellrowborder" valign="top" width="31.630000000000003%" id="mcps1.2.4.1.1"><p id="p05318355240"><a name="p05318355240"></a><a name="p05318355240"></a>feature_map.dtype</p>
</th>
<th class="cellrowborder" valign="top" width="33.629999999999995%" id="mcps1.2.4.1.2"><p id="p653183517246"><a name="p653183517246"></a><a name="p653183517246"></a>weight.dtype</p>
</th>
<th class="cellrowborder" valign="top" width="34.74%" id="mcps1.2.4.1.3"><p id="p4531143542414"><a name="p4531143542414"></a><a name="p4531143542414"></a>dst.dtype</p>
</th>
</tr>
</thead>
<tbody><tr id="row5531173512414"><td class="cellrowborder" valign="top" width="31.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p135316351243"><a name="p135316351243"></a><a name="p135316351243"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="33.629999999999995%" headers="mcps1.2.4.1.2 "><p id="p1153117354241"><a name="p1153117354241"></a><a name="p1153117354241"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="34.74%" headers="mcps1.2.4.1.3 "><p id="p11531103542411"><a name="p11531103542411"></a><a name="p11531103542411"></a>int32_t</p>
</td>
</tr>
<tr id="row6531193572411"><td class="cellrowborder" valign="top" width="31.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p185311535102416"><a name="p185311535102416"></a><a name="p185311535102416"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="33.629999999999995%" headers="mcps1.2.4.1.2 "><p id="p7531173522416"><a name="p7531173522416"></a><a name="p7531173522416"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="34.74%" headers="mcps1.2.4.1.3 "><p id="p18531103512242"><a name="p18531103512242"></a><a name="p18531103512242"></a>float</p>
</td>
</tr>
<tr id="row15119192914617"><td class="cellrowborder" valign="top" width="31.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p1087772917613"><a name="p1087772917613"></a><a name="p1087772917613"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="33.629999999999995%" headers="mcps1.2.4.1.2 "><p id="p138774291168"><a name="p138774291168"></a><a name="p138774291168"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="34.74%" headers="mcps1.2.4.1.3 "><p id="p98774298619"><a name="p98774298619"></a><a name="p98774298619"></a>half</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   该接口当前不支持W=Kw并且H\>Kh的场景，其将产生不可预期的结果。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

