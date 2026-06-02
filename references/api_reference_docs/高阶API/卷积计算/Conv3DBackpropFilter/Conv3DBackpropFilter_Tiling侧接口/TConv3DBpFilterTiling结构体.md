# TConv3DBpFilterTiling结构体

**页面ID:** atlasascendc_api_07_0907  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0907.html

---

TConv3DBpFilterTiling结构体包含Conv3dBackpropFilter算子规格信息及Tiling切分算法的相关参数，被传递给Conv3dBackpropFilter Kernel侧，用于数据切分、数据搬运和计算等。TConv3DBpFilterTiling结构体的参数说明见表1。

用户通过调用GetTiling接口获取TConv3DBpFilterTiling结构体，具体流程请参考使用说明。当前暂不支持用户自定义配置TConv3DBpFilterTiling结构体中的参数。

**表1 **TConv3DBpFilterTiling结构说明

| 参数名称 | 说明 |
| --- | --- |
| batch | 输入GradOutput的Batch，单位元素。 |
| cin | 输入Input的Channel，单位元素。 |
| cout | 输入GradOutput的Channel，单位元素。 |
| cin1G | 预留参数，用户无需感知。 |
| cout1G | 预留参数，用户无需感知。 |
| dout | 输入GradOutput的Depth，单位元素。 |
| ho | 输入GradOutput的Height，单位元素。 |
| wo | 输入GradOutput的Width，单位元素。 |
| di | 输入Input的Depth，单位元素。 |
| hi | 输入Input的Height，单位元素。 |
| wi | 输入Input的Width，单位元素。 |
| dk | 输出Weight的Depth，单位元素。 |
| hk | 输出Weight的Height，单位元素。 |
| wk | 输出Weight的Width，单位元素。 |
| group | 预留参数，用户无需感知。 |
| strideD | 卷积反向计算中Stride的Depth，单位元素。 |
| strideH | 卷积反向计算中Stride的Height，单位元素。 |
| strideW | 卷积反向计算中Stride的Width，单位元素。 |
| padFront | 卷积反向计算中Padding的Depth维度的前方向，单位元素。 |
| padBack | 卷积反向计算中Padding的Depth维度的后方向，单位元素。 |
| padUp | 卷积反向计算中Padding的Height维度的上方向，单位元素。 |
| padDown | 卷积反向计算中Padding的Height维度的下方向，单位元素。 |
| padLeft | 卷积反向计算中Padding的Width维度的左方向，单位元素。 |
| padRight | 卷积反向计算中Padding的Width维度的右方向，单位元素。 |
| dilationD | 卷积反向计算中Dilation的Depth，单位元素。 |
| dilationH | 卷积反向计算中Dilation的Height，单位元素。 |
| dilationW | 卷积反向计算中Dilation的Width，单位元素。 |
| channelSize | 当前输入数据类型下C0的大小。该参数目前只支持取值为16。 |
| al0Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| bl0Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| cl0Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| al1Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| bl1Pbuffer | 1表示不使能DoubleBuffer，2表示使能DoubleBuffer。 |
| baseM | L0上M方向大小，单位元素。 |
| baseK | L0上K方向大小，单位元素。 |
| baseN | L0上N方向大小，单位元素。 |
| m0 | L0上最小分形M方向大小。 |
| k0 | L0上最小分形K方向大小。 |
| n0 | L0上最小分形N方向大小。 |
| stepM | 矩阵在L1中缓存的buffer M方向上baseM的倍数。 |
| stepN | 矩阵在L1中缓存的buffer N方向上baseN的倍数。 |
| stepKa | 矩阵在L1中缓存的buffer K方向上baseK的倍数。 |
| stepKb | 矩阵在L1中缓存的buffer K方向上baseK的倍数。 |
| iterateOrder | 预留参数，用户无需感知。 |
| bl1Bound | L1中载入GradOutput矩阵的最大数据量。 |
| hf32Flag | 预留参数，用户无需感知。 |
| singleCoreDK | 预留参数，用户无需感知。 |
| singleCoreGroup | 预留参数，用户无需感知。 |
| singleCoreCout | 单核M方向上计算cout数据量的大小，单位元素。 |
| singleCoreHo | 单核K方向上计算ho数据量的大小，单位元素。 |
| singleCoreBatch | 单核上batch的大小，单位元素。 |
| singleCoreCin | 单核N方向上计算cin数据量的大小，单位元素。 |
| totalL1Size | L1 size大小，单位元素。 |
| singleCoreM | 单核上M的大小，单位元素。 |
| singleCoreN | 单核上N的大小，单位元素。 |
| singleCoreK | 单核上K的大小，单位元素。 |
