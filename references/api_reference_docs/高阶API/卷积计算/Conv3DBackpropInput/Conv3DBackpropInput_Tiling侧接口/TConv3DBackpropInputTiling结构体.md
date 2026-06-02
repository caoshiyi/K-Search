# TConv3DBackpropInputTiling结构体

**页面ID:** atlasascendc_api_07_0932  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0932.html

---

TConv3DBackpropInputTiling结构体包含Conv3DBackpropInput算子规格信息及Tiling切分算法的相关参数，被传递给Conv3DBackpropInput Kernel侧，用于数据切分、数据搬运和计算等。TConv3DBackpropInputTiling结构体的参数说明见下表。

用户通过调用GetTiling接口获取TConv3DBackpropInputTiling结构体，具体流程请参考Conv3DBackpropInput Tiling使用说明。当前暂不支持用户自定义配置TConv3DBackpropInputTiling结构体中的参数。

**表1 **TConv3DBackpropInputTiling结构说明

| 参数名称 | 说明 |
| --- | --- |
| batch | 输入GradOutput的N，等于卷积正向输入Input的N。 |
| cin | 输出GradInput的Channel，等于卷积正向输入Input的Channel。 |
| cout | 输入GradOutput的Channel。 |
| cout1 | 输入GradOutput的C1，等于cout/c0。 |
| cin1 | 输出GradInput的C1，等于卷积正向输入Input的C1，等于cin/c0。 |
| cout1G | 预留参数，用户无需感知。 |
| cin1G | 预留参数，用户无需感知。 |
| c0 | 当前输入数据类型下C0的大小。该参数目前只支持取值为16。 |
| c0Bits | 任意一个数除以c0等价的右移位数，例如c0=8则c0Bits=3，c0=16则c0Bits=4。 |
| dout | 输入GradOutput的Depth大小，单位元素。 |
| ho | 输入GradOutput的Height大小，单位元素。 |
| wo | 输入GradOutput的Width大小，单位元素。 |
| di | 输出GradInput的Depth大小，等于卷积正向输入Input的Depth大小，单位元素。 |
| hi | 输出GradInput的Height大小，等于卷积正向输入Input的Height大小，单位元素。 |
| wi | 输出GradInput的Width大小，等于卷积正向输入Input的Width大小，单位元素。 |
| dk | 输入Weight的Depth大小，单位元素。 |
| hk | 输入Weight的Height大小，单位元素。 |
| wk | 输入Weight的Width大小，单位元素。 |
| group | 预留参数，用户无需感知。 |
| strideD | 卷积反向计算中Stride的Depth大小，单位元素。 |
| strideH | 卷积反向计算中StrideHeight大小，单位元素。 |
| strideW | 卷积反向计算中StrideWidth大小，单位元素。 |
| padFront | 卷积反向计算中输出矩阵GradInput Padding的Depth维度的前方向，单位元素。 |
| padBack | 卷积反向计算中输出矩阵GradInput Padding的Depth维度的后方向，单位元素。 |
| padUp | 卷积反向计算中输出矩阵GradInput Padding的Height维度的上方向，单位元素。 |
| padDown | 卷积反向计算中输出矩阵GradInput Padding的Height维度的下方向，单位元素。 |
| padLeft | 卷积反向计算中输出矩阵GradInput Padding的Width维度的左方向，单位元素。 |
| padRight | 卷积反向计算中输出矩阵GradInput Padding的Width维度的右方向，单位元素。 |
| backpropPadTail | 预留参数，用户无需感知。 |
| backpropPadUp | 卷积反向计算中输入矩阵GradOutput Padding的Height维度的上方向，单位元素。 |
| backpropPadDown | 卷积反向计算中输入矩阵GradOutput Padding的Height维度的下方向，单位元素。 |
| backpropPadLeft | 卷积反向计算中输入矩阵GradOutput Padding的Width维度的左方向，单位元素。 |
| backpropPadRight | 卷积反向计算中输入矩阵GradOutput Padding的Width维度的右方向，单位元素。 |
| dilationD | 卷积反向计算中Dilation的Depth大小，单位元素。 |
| dilationH | 卷积反向计算中Dilation的Height大小，单位元素。 |
| dilationW | 卷积反向计算中Dilation的Width大小，单位元素。 |
| al0Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| bl0Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| cl0Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| al1Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| bl1Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| singleCoreGroup | 预留参数，用户无需感知。 |
| singleCoreCout | 单核M方向上计算cout数据量的大小。 |
| singleCoreCout1 | 单核上cout1的大小。 |
| singleCoreCin1 | 单核上cin1的大小。 |
| singleCoreDin | 单核上Din的大小。 |
| singleCoreHo | 单核K方向上计算ho数据量的大小。 |
| baseM | L0上M方向大小。 |
| baseK | L0上K方向大小。 |
| baseN | L0上N方向大小。 |
| baseD | 预留参数，用户无需感知。 |
| baseBatch | 预留参数，用户无需感知。 |
| baseGroup | 预留参数，用户无需感知。 |
| stepM | 特征矩阵在L1中缓存的buffer M方向上baseM的倍数。 |
| stepN | 权重矩阵在L1中缓存的buffer N方向上baseN的倍数。 |
| stepKa | 特征矩阵在L1中缓存的buffer K方向上baseK的倍数。 |
| stepKb | 权重矩阵在L1中缓存的buffer K方向上baseK的倍数。 |
| stepBatch | 预留参数，用户无需感知。 |
| stepGroup | 预留参数，用户无需感知。 |
| iterateOrder | 预留参数，用户无需感知。 |
| hf32Flag | 预留参数，用户无需感知。 |
| initOutputFlag | 预留参数，用户无需感知。 |
| reserved | 预留参数，用户无需感知。 |
| singleCoreBatch | 预留参数，用户无需感知。 |
| singleCoreM | 单核M方向上需要计算的数据量大小。 |
| singleCoreCin | 单核N方向上计算cin数据量的大小。 |
