# Conv3D模板参数<a name="ZH-CN_TOPIC_0000002523344768"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section4299173755913"></a>

创建Conv3D对象时需要传入：

-   Input、Weight、Output和Bias（可选）的参数类型信息， 类型信息通过[ConvType](Conv3D使用说明.md#table19081115275)来定义，包括：内存逻辑位置、数据格式、数据类型。
-   Conv3dParam信息（可选），用于使能不同场景的性能优化模板。**当前暂不支持使用。**

## 函数原型<a name="section079519516019"></a>

```
template <class INPUT_TYPE, class WEIGHT_TYPE, class OUTPUT_TYPE, class BIAS_TYPE = biasType, class CONV_CFG = Conv3dParam>
using Conv3D = Conv3dIntfExt<Config<ConvApi::ConvDataType<INPUT_TYPE, WEIGHT_TYPE, OUTPUT_TYPE, BIAS_TYPE, CONV_CFG>>, Impl, Intf>
```

## 参数说明<a name="section423183813019"></a>

**表 1**  模板参数说明

<a name="table17247917193819"></a>
<table><thead align="left"><tr id="row826411177387"><th class="cellrowborder" valign="top" width="21.43%" id="mcps1.2.4.1.1"><p id="p1626431733817"><a name="p1626431733817"></a><a name="p1626431733817"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.4.1.2"><p id="p526411177386"><a name="p526411177386"></a><a name="p526411177386"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="59.18%" id="mcps1.2.4.1.3"><p id="p122641917173815"><a name="p122641917173815"></a><a name="p122641917173815"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1526491719388"><td class="cellrowborder" valign="top" width="21.43%" headers="mcps1.2.4.1.1 "><p id="p17264191712384"><a name="p17264191712384"></a><a name="p17264191712384"></a>INPUT_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.4.1.2 "><p id="p112644172389"><a name="p112644172389"></a><a name="p112644172389"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.18%" headers="mcps1.2.4.1.3 "><p id="p5264151723816"><a name="p5264151723816"></a><a name="p5264151723816"></a>ConvType类型模板参数，指定Input的参数类型信息。</p>
</td>
</tr>
<tr id="row925415432126"><td class="cellrowborder" valign="top" width="21.43%" headers="mcps1.2.4.1.1 "><p id="p0255144371216"><a name="p0255144371216"></a><a name="p0255144371216"></a>WEIGHT_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.4.1.2 "><p id="p1925544314126"><a name="p1925544314126"></a><a name="p1925544314126"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.18%" headers="mcps1.2.4.1.3 "><p id="p1625544311127"><a name="p1625544311127"></a><a name="p1625544311127"></a>ConvType类型模板参数，指定Weight的参数类型信息。</p>
</td>
</tr>
<tr id="row116212131311"><td class="cellrowborder" valign="top" width="21.43%" headers="mcps1.2.4.1.1 "><p id="p1862921101319"><a name="p1862921101319"></a><a name="p1862921101319"></a>OUTPUT_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.4.1.2 "><p id="p11621421151315"><a name="p11621421151315"></a><a name="p11621421151315"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.18%" headers="mcps1.2.4.1.3 "><p id="p172284541514"><a name="p172284541514"></a><a name="p172284541514"></a>ConvType类型模板参数，指定Output的参数类型信息。</p>
</td>
</tr>
<tr id="row868991181411"><td class="cellrowborder" valign="top" width="21.43%" headers="mcps1.2.4.1.1 "><p id="p1468916111419"><a name="p1468916111419"></a><a name="p1468916111419"></a>BIAS_TYPE</p>
</td>
<td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.4.1.2 "><p id="p868941121418"><a name="p868941121418"></a><a name="p868941121418"></a>可选输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.18%" headers="mcps1.2.4.1.3 "><p id="p580413631610"><a name="p580413631610"></a><a name="p580413631610"></a>ConvType类型模板参数，指定Bias的参数类型信息。</p>
</td>
</tr>
<tr id="row9874182141411"><td class="cellrowborder" valign="top" width="21.43%" headers="mcps1.2.4.1.1 "><p id="p3875112116149"><a name="p3875112116149"></a><a name="p3875112116149"></a>CONV_CFG</p>
</td>
<td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.4.1.2 "><p id="p98751421121411"><a name="p98751421121411"></a><a name="p98751421121411"></a>可选输入</p>
</td>
<td class="cellrowborder" valign="top" width="59.18%" headers="mcps1.2.4.1.3 "><p id="p148751821131416"><a name="p148751821131416"></a><a name="p148751821131416"></a>ConvParam类型模板参数，用于使能不同场景的性能优化模板，<strong id="b1547928121812"><a name="b1547928121812"></a><a name="b1547928121812"></a>当前版本只支持基础模板，不使能性能优化。</strong></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section11828111017"></a>

无

## 约束说明<a name="section29522578115"></a>

无

## 调用示例<a name="section1242919111927"></a>

```
#include "lib/conv/conv3d/conv3d_api.h"

using inputType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::NDC1HWC0, bfloat16_t>;
using weightType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::FRACTAL_Z_3D, bfloat16_t>;
using outputType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::NDC1HWC0, bfloat16_t>;
using biasType = ConvApi::ConvType<AscendC::TPosition::GM, ConvFormat::ND, float>; // 可选参数，如果不带Bias场景，可以不传
struct ConvCustom : public ConvApi::ConvParam {
    __aicore__ inline ConvCustom(){};
}; // 可选参数，当前版本只支持基础模板，不使能性能优化，可以不传

Conv3dApi::Conv3D<inputType, weightType, outputType, biasType, ConvCustom> conv3dApi;
```

