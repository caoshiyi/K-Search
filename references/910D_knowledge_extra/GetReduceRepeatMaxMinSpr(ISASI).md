# GetReduceRepeatMaxMinSpr\(ISASI\)<a name="ZH-CN_TOPIC_0000002523303584"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="50.71000000000001%" id="mcps1.1.4.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="25.770000000000003%" id="mcps1.1.4.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持（仅获取最值的原型）</p>
</th>
<th class="cellrowborder" valign="top" width="23.520000000000003%" id="mcps1.1.4.1.3"><p id="p950233175911"><a name="p950233175911"></a><a name="p950233175911"></a>是否支持（获取最值和索引的原型）</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="50.71000000000001%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="25.770000000000003%" headers="mcps1.1.4.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>x</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="23.520000000000003%" headers="mcps1.1.4.1.3 "><p id="p19714340116"><a name="p19714340116"></a><a name="p19714340116"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

获取[ReduceMax](ReduceMax.md)、[ReduceMin](ReduceMin.md)连续场景下的最大/最小值以及相应的索引值。

## 函数原型<a name="section620mcpsimp"></a>

-   获取[ReduceMax](ReduceMax.md)、[ReduceMin](ReduceMin.md)连续场景下的最大值与最小值，以及相应的索引值。

    ```
    template <typename T>
    __aicore__ inline void GetReduceRepeatMaxMinSpr(T &maxMinValue, T &maxMinIndex)
    ```

-   获取[ReduceMax](ReduceMax.md)、[ReduceMin](ReduceMin.md)连续场景下的最大值与最小值。

    ```
    template <typename T>
    __aicore__ inline void GetReduceRepeatMaxMinSpr(T &maxMinValue)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="18.58%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.42%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="18.58%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.42%" headers="mcps1.2.3.1.2 "><p id="p5507191632414"><a name="p5507191632414"></a><a name="p5507191632414"></a>ReduceMax/ReduceMin指令的数据类型，支持half/float。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>maxMinValue</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p178192610120"><a name="p178192610120"></a><a name="p178192610120"></a>ReduceMax/ReduceMin指令的最大值/最小值。</p>
</td>
</tr>
<tr id="row1441613274210"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p14168210429"><a name="p14168210429"></a><a name="p14168210429"></a>maxMinIndex</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p54161122420"><a name="p54161122420"></a><a name="p54161122420"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11819258217"><a name="p11819258217"></a><a name="p11819258217"></a>ReduceMax/ReduceMin指令的最值对应的索引值。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   针对Ascend 950PR/Ascend 950DT，由于ReduceMax/ReduceMin的内部实现原因，直接调用GetReduceRepeatMaxMinSpr接口无法获取到准确的索引值，验证时需要使用[WholeReduceMax](WholeReduceMax.md)/[WholeReduceMin](WholeReduceMin.md)接口来获取准确的索引值。同时，GetReduceRepeatMaxMinSpr必须紧跟着[WholeReduceMax](WholeReduceMax.md)/[WholeReduceMin](WholeReduceMin.md)接口进行调用。
-   索引maxMinIndex数据\`是按照ReduceMax/ReduceMin的数据类型进行存储的，比如ReduceMax/ReduceMin使用half类型时，maxMinIndex是按照half类型进行存储的，如果按照half格式进行读取，maxMinIndex的值是不对的，因此maxMinIndex的读取需要使用reinterpret\_cast方法转换到整数类型，若输入数据类型是half，需要使用reinterpret\_cast<uint16\_t\*\>，若输入是float，需要使用reinterpret\_cast<uint32\_t\*\>。

## 调用示例<a name="section837496171220"></a>

1.  以ReduceMax指令为例，首先执行ReduceMax指令。

    ```
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<float> work;
    AscendC::LocalTensor<float> dst;
    int32_t mask = 64;
    AscendC::ReduceMax(dst, src, work, mask, 1, 8, true); // 连续场景，srcRepStride = 8，且calIndex = true
    ```

2.  获取上述ReduceMax指令的最值与索引值。

    针对Ascend 950PR/Ascend 950DT，需要先执行WholeReduceMax指令，随后立即调用GetReduceRepeatMaxMinSpr指令。

    ```
    AscendC::LocalTensor<float> src;
    AscendC::LocalTensor<float> dst;
    int32_t mask = 64;
    float val = 0;   // 最大值
    float idx = 0;   // 最大值的索引值，与ReduceMax的结果相同
    AscendC::WholeReduceMax(dst, src, mask, 1, 1, 1, 8);
    AscendC::GetReduceRepeatMaxMinSpr<float>(val, idx); // 保证和WholeReduceMax的调动次序，而且要配对调用
    ```

