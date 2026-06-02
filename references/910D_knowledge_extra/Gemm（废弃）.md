# Gemm（废弃）<a name="ZH-CN_TOPIC_0000002523304324"></a>

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

根据输入的切分规则，将给定的两个输入张量做矩阵乘，输出至结果张量。将A和B两个输入矩阵乘法在一起，得到一个输出矩阵C。

## 函数原型<a name="section620mcpsimp"></a>

-   功能接口：

    ```
    template <typename T, typename U, typename S>
    __aicore__ inline void Gemm(const LocalTensor<T>& dst, const LocalTensor<U>& src0, const LocalTensor<S>& src1, const uint32_t m, const uint32_t k, const uint32_t n, GemmTiling tilling, bool partialsum = true, int32_t initValue = 0)
    ```

-   切分方案计算接口：

    ```
    template <typename T>
    __aicore__ inline GemmTiling GetGemmTiling(uint32_t m, uint32_t k, uint32_t n)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  接口参数说明

<a name="table284254116544"></a>
<table><thead align="left"><tr id="row10842204115543"><th class="cellrowborder" valign="top" width="18.5018501850185%" id="mcps1.2.4.1.1"><p id="p565185175414"><a name="p565185175414"></a><a name="p565185175414"></a><strong id="b16565112547"><a name="b16565112547"></a><a name="b16565112547"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="13.67136713671367%" id="mcps1.2.4.1.2"><p id="p7651751165417"><a name="p7651751165417"></a><a name="p7651751165417"></a><strong id="b1365351135416"><a name="b1365351135416"></a><a name="b1365351135416"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="67.82678267826783%" id="mcps1.2.4.1.3"><p id="p36519513546"><a name="p36519513546"></a><a name="p36519513546"></a><strong id="b16545185411"><a name="b16545185411"></a><a name="b16545185411"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1784224115547"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p670781885515"><a name="p670781885515"></a><a name="p670781885515"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p13707161835517"><a name="p13707161835517"></a><a name="p13707161835517"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p14707151813554"><a name="p14707151813554"></a><a name="p14707151813554"></a>目的操作数。</p>
</td>
</tr>
<tr id="row1484210419547"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p97071518105511"><a name="p97071518105511"></a><a name="p97071518105511"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p870771885511"><a name="p870771885511"></a><a name="p870771885511"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p1470731835517"><a name="p1470731835517"></a><a name="p1470731835517"></a>源操作数，TPosition为A1。</p>
</td>
</tr>
<tr id="row16259173555"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p16707121817554"><a name="p16707121817554"></a><a name="p16707121817554"></a>src1</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p1707718195512"><a name="p1707718195512"></a><a name="p1707718195512"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p1770818189556"><a name="p1770818189556"></a><a name="p1770818189556"></a>源操作数，TPosition为B1。</p>
</td>
</tr>
<tr id="row14842841145419"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p1170841855513"><a name="p1170841855513"></a><a name="p1170841855513"></a>m</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p8708101815553"><a name="p8708101815553"></a><a name="p8708101815553"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p18653233161017"><a name="p18653233161017"></a><a name="p18653233161017"></a>左矩阵Src0Local有效Height，范围：[1, 4096]。</p>
<p id="p6653933131011"><a name="p6653933131011"></a><a name="p6653933131011"></a>注意：m可以不是16的倍数。</p>
</td>
</tr>
<tr id="row23311816165517"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p107081018115518"><a name="p107081018115518"></a><a name="p107081018115518"></a>k</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p7708818195510"><a name="p7708818195510"></a><a name="p7708818195510"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><div class="p" id="p138408408105"><a name="p138408408105"></a><a name="p138408408105"></a>左矩阵Src0Local有效Width、右矩阵Src1Local有效Height。<a name="ul643893015319"></a><a name="ul643893015319"></a><ul id="ul643893015319"><li>当输入张量Src0Local的数据类型为float时，范围：[1, 8192]</li><li>当输入张量Src0Local的数据类型为half时，范围：[1, 16384]</li><li>当输入张量Src0Local的数据类型为int8_t时，范围：[1, 32768]</li></ul>
</div>
<p id="p9840104011105"><a name="p9840104011105"></a><a name="p9840104011105"></a>注意：k可以不是16的倍数。</p>
</td>
</tr>
<tr id="row11842164195417"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p19708018125520"><a name="p19708018125520"></a><a name="p19708018125520"></a>n</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p1570815184559"><a name="p1570815184559"></a><a name="p1570815184559"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p895316152119"><a name="p895316152119"></a><a name="p895316152119"></a>右矩阵Src1Local有效Width，范围：[1, 4096]。</p>
<p id="p59539152113"><a name="p59539152113"></a><a name="p59539152113"></a>注意：n可以不是16的倍数。</p>
</td>
</tr>
<tr id="row18843104116541"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p197081718205514"><a name="p197081718205514"></a><a name="p197081718205514"></a>tilling</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p1570871813554"><a name="p1570871813554"></a><a name="p1570871813554"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p47082187557"><a name="p47082187557"></a><a name="p47082187557"></a>切分规则，类型为GemmTiling，结构体具体定义为：</p>
<a name="screen8348103082311"></a><a name="screen8348103082311"></a><pre class="screen" codetype="Cpp" id="screen8348103082311">struct GemmTiling {
    const uint32_t blockSize = 16;
    LoopMode loopMode = LoopMode::MODE_NM;
    uint32_t mNum = 0;
    uint32_t nNum = 0;
    uint32_t kNum = 0;
    uint32_t roundM = 0;
    uint32_t roundN = 0;
    uint32_t roundK = 0;
    uint32_t c0Size = 32;
    uint32_t dtypeSize = 1;
    uint32_t mBlockNum = 0;
    uint32_t nBlockNum = 0;
    uint32_t kBlockNum = 0;
    uint32_t mIterNum = 0;
    uint32_t nIterNum = 0;
    uint32_t kIterNum = 0;
    uint32_t mTileBlock = 0;
    uint32_t nTileBlock = 0;
    uint32_t kTileBlock = 0;
    uint32_t kTailBlock = 0;
    uint32_t mTailBlock = 0;
    uint32_t nTailBlock = 0;
    bool kHasTail = false;
    bool mHasTail = false;
    bool nHasTail = false;
    bool kHasTailEle = false;
    uint32_t kTailEle = 0;
};</pre>
<p id="p12287014111614"><a name="p12287014111614"></a><a name="p12287014111614"></a>参数说明请参考<a href="#table946018393169">表3</a>。</p>
</td>
</tr>
<tr id="row18843124135420"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p1708101885513"><a name="p1708101885513"></a><a name="p1708101885513"></a>partialsum</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p17085183558"><a name="p17085183558"></a><a name="p17085183558"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><div class="p" id="p18202154220519"><a name="p18202154220519"></a><a name="p18202154220519"></a>当dst参数所在的TPosition为CO2时，通过该参数控制计算结果是否搬出。<a name="ul785116545312"></a><a name="ul785116545312"></a><ul id="ul785116545312"><li>取值0：搬出计算结果</li><li>取值1：不搬出计算结果，可以进行后续计算</li></ul>
</div>
</td>
</tr>
<tr id="row7877131475511"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p16807192215559"><a name="p16807192215559"></a><a name="p16807192215559"></a>initValue</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p48071622155510"><a name="p48071622155510"></a><a name="p48071622155510"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><div class="p" id="p139771247141311"><a name="p139771247141311"></a><a name="p139771247141311"></a>表示dst是否需要初始化。<a name="ul1052612249324"></a><a name="ul1052612249324"></a><ul id="ul1052612249324"><li>取值0: dst需要初始化，dst初始矩阵保存有之前结果，新计算结果会累加前一次conv2d计算结果。</li><li>取值1: dst不需要初始化，dst初始矩阵中数据无意义，计算结果直接覆盖dst中的数据。</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

**表 2**  feature\_map、weight和dst的数据类型组合

<a name="table7518203512413"></a>
<table><thead align="left"><tr id="row13531103532418"><th class="cellrowborder" valign="top" width="31.630000000000003%" id="mcps1.2.4.1.1"><p id="p05318355240"><a name="p05318355240"></a><a name="p05318355240"></a>src0.dtype</p>
</th>
<th class="cellrowborder" valign="top" width="33.629999999999995%" id="mcps1.2.4.1.2"><p id="p653183517246"><a name="p653183517246"></a><a name="p653183517246"></a>src1.dtype</p>
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
<tr id="row1758205917253"><td class="cellrowborder" valign="top" width="31.630000000000003%" headers="mcps1.2.4.1.1 "><p id="p659719062613"><a name="p659719062613"></a><a name="p659719062613"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="33.629999999999995%" headers="mcps1.2.4.1.2 "><p id="p1059750112616"><a name="p1059750112616"></a><a name="p1059750112616"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="34.74%" headers="mcps1.2.4.1.3 "><p id="p85979015268"><a name="p85979015268"></a><a name="p85979015268"></a>half</p>
</td>
</tr>
</tbody>
</table>

**表 3**  GemmTiling结构内参数说明

<a name="table946018393169"></a>
<table><thead align="left"><tr id="row4460203971617"><th class="cellrowborder" valign="top" width="20.382038203820382%" id="mcps1.2.4.1.1"><p id="p3675135251615"><a name="p3675135251615"></a><a name="p3675135251615"></a><strong id="b136752527166"><a name="b136752527166"></a><a name="b136752527166"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="19.451945194519453%" id="mcps1.2.4.1.2"><p id="p126750529168"><a name="p126750529168"></a><a name="p126750529168"></a><strong id="b1767525220161"><a name="b1767525220161"></a><a name="b1767525220161"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="60.16601660166017%" id="mcps1.2.4.1.3"><p id="p167512524162"><a name="p167512524162"></a><a name="p167512524162"></a><strong id="b2675125210166"><a name="b2675125210166"></a><a name="b2675125210166"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row946193911613"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p4675115220161"><a name="p4675115220161"></a><a name="p4675115220161"></a>blockSize</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p136751152151613"><a name="p136751152151613"></a><a name="p136751152151613"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p1680811983820"><a name="p1680811983820"></a><a name="p1680811983820"></a>固定值，恒为16，一个维度内存放的元素个数。</p>
</td>
</tr>
<tr id="row9461173991618"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1267515521162"><a name="p1267515521162"></a><a name="p1267515521162"></a>loopMode</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p12675195211614"><a name="p12675195211614"></a><a name="p12675195211614"></a>LoopMode</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p880829103813"><a name="p880829103813"></a><a name="p880829103813"></a>遍历模式，结构体具体定义为：</p>
<a name="screen1670614014719"></a><a name="screen1670614014719"></a><pre class="screen" codetype="Cpp" id="screen1670614014719">enum class LoopMode {
    MODE_NM = 0,
    MODE_MN = 1,
    MODE_KM = 2,
    MODE_KN = 3
};</pre>
</td>
</tr>
<tr id="row646123917168"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1467515281618"><a name="p1467515281618"></a><a name="p1467515281618"></a>mNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p36751752191612"><a name="p36751752191612"></a><a name="p36751752191612"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p13808209153810"><a name="p13808209153810"></a><a name="p13808209153810"></a>M轴等效数据长度参数值，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row114611139191611"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p15675752181616"><a name="p15675752181616"></a><a name="p15675752181616"></a>nNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p14676252171618"><a name="p14676252171618"></a><a name="p14676252171618"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p7808090384"><a name="p7808090384"></a><a name="p7808090384"></a>N轴等效数据长度参数值，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row24611393167"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p2067695214167"><a name="p2067695214167"></a><a name="p2067695214167"></a>kNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p156761052191615"><a name="p156761052191615"></a><a name="p156761052191615"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><div class="p" id="p880949103811"><a name="p880949103811"></a><a name="p880949103811"></a>K轴等效数据长度参数值。<a name="ul12581856193213"></a><a name="ul12581856193213"></a><ul id="ul12581856193213"><li>当输入张量Src0Local的数据类型为float时，范围：[1, 8192]</li><li>当输入张量Src0Local的数据类型为half时，范围：[1, 16384]</li><li>当输入张量Src0Local的数据类型为int8_t时，范围：[1, 32768]</li></ul>
</div>
</td>
</tr>
<tr id="row1246133915165"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p46766527163"><a name="p46766527163"></a><a name="p46766527163"></a>roundM</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p5676205217164"><a name="p5676205217164"></a><a name="p5676205217164"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p680912915384"><a name="p680912915384"></a><a name="p680912915384"></a>M轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1, 4096]</p>
</td>
</tr>
<tr id="row546193913161"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1167675261611"><a name="p1167675261611"></a><a name="p1167675261611"></a>roundN</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p967613529164"><a name="p967613529164"></a><a name="p967613529164"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p1180919193810"><a name="p1180919193810"></a><a name="p1180919193810"></a>N轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1, 4096]</p>
</td>
</tr>
<tr id="row194621399163"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p18676145211161"><a name="p18676145211161"></a><a name="p18676145211161"></a>roundK</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p46761352181610"><a name="p46761352181610"></a><a name="p46761352181610"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><div class="p" id="p108098911384"><a name="p108098911384"></a><a name="p108098911384"></a>K轴等效数据长度参数值且以c0Size为倍数向上取整。<a name="ul45801622173311"></a><a name="ul45801622173311"></a><ul id="ul45801622173311"><li>当输入张量Src0Local的数据类型为float时，范围：[1, 8192]</li><li>当输入张量Src0Local的数据类型为half时，范围：[1, 16384]</li><li>当输入张量Src0Local的数据类型为int8_t时，范围：[1, 32768]</li></ul>
</div>
</td>
</tr>
<tr id="row1946243910169"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p8676135219169"><a name="p8676135219169"></a><a name="p8676135219169"></a>c0Size</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p967605219168"><a name="p967605219168"></a><a name="p967605219168"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p109671431174217"><a name="p109671431174217"></a><a name="p109671431174217"></a>一个block的字节长度，范围：[16或者32]。</p>
</td>
</tr>
<tr id="row114628390163"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p4676552121611"><a name="p4676552121611"></a><a name="p4676552121611"></a>dtypeSize</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p967655213164"><a name="p967655213164"></a><a name="p967655213164"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p096653110425"><a name="p096653110425"></a><a name="p096653110425"></a>传入的数据类型的字节长度，范围：[1, 2]。</p>
</td>
</tr>
<tr id="row174621739131619"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1367717524167"><a name="p1367717524167"></a><a name="p1367717524167"></a>mBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p12677175217167"><a name="p12677175217167"></a><a name="p12677175217167"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p12809189203814"><a name="p12809189203814"></a><a name="p12809189203814"></a>M轴Block个数，mBlockNum = mNum / blockSize。</p>
</td>
</tr>
<tr id="row246273919168"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p176771052121611"><a name="p176771052121611"></a><a name="p176771052121611"></a>nBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p6677185216164"><a name="p6677185216164"></a><a name="p6677185216164"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p680999123812"><a name="p680999123812"></a><a name="p680999123812"></a>N轴Block个数，nBlockNum = nNum / blockSize。</p>
</td>
</tr>
<tr id="row18462839141611"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p196771952111620"><a name="p196771952111620"></a><a name="p196771952111620"></a>kBlockNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p11677195214168"><a name="p11677195214168"></a><a name="p11677195214168"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p380999143819"><a name="p380999143819"></a><a name="p380999143819"></a>K轴Block个数，kBlockNum = kNum / blockSize。</p>
</td>
</tr>
<tr id="row12462153916161"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1567719522161"><a name="p1567719522161"></a><a name="p1567719522161"></a>mIterNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p76771752181619"><a name="p76771752181619"></a><a name="p76771752181619"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p6677352191616"><a name="p6677352191616"></a><a name="p6677352191616"></a>遍历维度数量，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row646283912168"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p867735241616"><a name="p867735241616"></a><a name="p867735241616"></a>nIterNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p96770527168"><a name="p96770527168"></a><a name="p96770527168"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p56771252131610"><a name="p56771252131610"></a><a name="p56771252131610"></a>遍历维度数量，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row144621139151618"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p6677155251615"><a name="p6677155251615"></a><a name="p6677155251615"></a>kIterNum</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p6678952101613"><a name="p6678952101613"></a><a name="p6678952101613"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p4678125219161"><a name="p4678125219161"></a><a name="p4678125219161"></a>遍历维度数量，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row7463639181613"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1767812527164"><a name="p1767812527164"></a><a name="p1767812527164"></a>mTileBlock</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p1467805218166"><a name="p1467805218166"></a><a name="p1467805218166"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p767875241615"><a name="p767875241615"></a><a name="p767875241615"></a>M轴切分块个数，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row13463193991618"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p15678125251617"><a name="p15678125251617"></a><a name="p15678125251617"></a>nTileBlock</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p186786522166"><a name="p186786522166"></a><a name="p186786522166"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p5678115217168"><a name="p5678115217168"></a><a name="p5678115217168"></a>N轴切分块个数，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row12463153911167"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p18678115210163"><a name="p18678115210163"></a><a name="p18678115210163"></a>kTileBlock</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p146781352161613"><a name="p146781352161613"></a><a name="p146781352161613"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p1867845241611"><a name="p1867845241611"></a><a name="p1867845241611"></a>K轴切分块个数，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row246316397167"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p13678252131616"><a name="p13678252131616"></a><a name="p13678252131616"></a>kTailBlock</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p14678145211615"><a name="p14678145211615"></a><a name="p14678145211615"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p5678052191612"><a name="p5678052191612"></a><a name="p5678052191612"></a>K轴尾块个数，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row18463639171613"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p667835261614"><a name="p667835261614"></a><a name="p667835261614"></a>mTailBlock</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p19678155231618"><a name="p19678155231618"></a><a name="p19678155231618"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p16781452111610"><a name="p16781452111610"></a><a name="p16781452111610"></a>M轴尾块个数，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row1246323991610"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1667835211164"><a name="p1667835211164"></a><a name="p1667835211164"></a>nTailBlock</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p767914523167"><a name="p767914523167"></a><a name="p767914523167"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p967995281613"><a name="p967995281613"></a><a name="p967995281613"></a>N轴尾块个数，范围：[1, 4096]。</p>
</td>
</tr>
<tr id="row84634396168"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p136791952111616"><a name="p136791952111616"></a><a name="p136791952111616"></a>kHasTail</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p176791252131617"><a name="p176791252131617"></a><a name="p176791252131617"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p11679125217164"><a name="p11679125217164"></a><a name="p11679125217164"></a>K轴是否存在尾块。</p>
</td>
</tr>
<tr id="row2463839111616"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p267911521162"><a name="p267911521162"></a><a name="p267911521162"></a>mHasTail</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p2679105219168"><a name="p2679105219168"></a><a name="p2679105219168"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p19679155211163"><a name="p19679155211163"></a><a name="p19679155211163"></a>M轴是否存在尾块。</p>
</td>
</tr>
<tr id="row3463113961619"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p10679195215167"><a name="p10679195215167"></a><a name="p10679195215167"></a>nHasTail</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p11679205216162"><a name="p11679205216162"></a><a name="p11679205216162"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p15679165261620"><a name="p15679165261620"></a><a name="p15679165261620"></a>N轴是否存在尾块。</p>
</td>
</tr>
<tr id="row146453918167"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p1067925221615"><a name="p1067925221615"></a><a name="p1067925221615"></a>kHasTailEle</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p967955218166"><a name="p967955218166"></a><a name="p967955218166"></a>bool</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p1679145291610"><a name="p1679145291610"></a><a name="p1679145291610"></a>是否存在尾块元素。</p>
</td>
</tr>
<tr id="row9464103918163"><td class="cellrowborder" valign="top" width="20.382038203820382%" headers="mcps1.2.4.1.1 "><p id="p13679205216165"><a name="p13679205216165"></a><a name="p13679205216165"></a>kTailEle</p>
</td>
<td class="cellrowborder" valign="top" width="19.451945194519453%" headers="mcps1.2.4.1.2 "><p id="p967975211614"><a name="p967975211614"></a><a name="p967975211614"></a>uint32_t</p>
</td>
<td class="cellrowborder" valign="top" width="60.16601660166017%" headers="mcps1.2.4.1.3 "><p id="p156791352151617"><a name="p156791352151617"></a><a name="p156791352151617"></a>K轴尾块元素，范围：[1, 4096]。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   参数m，k，n可以不是16对齐，但因硬件原因，操作数dst，Src0Local和Src1Local的shape需满足对齐要求，即m方向，n方向要求向上16对齐，k方向根据操作数数据类型按16或32向上对齐。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

