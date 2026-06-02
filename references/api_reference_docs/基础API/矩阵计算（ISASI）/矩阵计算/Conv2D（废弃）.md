# Conv2D（废弃）

**页面ID:** atlasascendc_api_07_0262  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0262.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

**该接口废弃，并将在后续版本移除，请不要使用该接口。**

计算给定输入张量和权重张量的2-D卷积，输出结果张量。Conv2d卷积层多用于图像识别，使用过滤器提取图像中的特征。

#### 函数原型

```
template <typename T, typename U>
__aicore__ inline void Conv2D(const LocalTensor<T>& dst, const LocalTensor<U>& featureMap, const LocalTensor<U>& weight, Conv2dParams& conv2dParams, Conv2dTilling& tilling)
```

入参中的tiling结构需要通过如下切分方案计算接口来获取：

```
template <typename T>
__aicore__ inline Conv2dTilling GetConv2dTiling(Conv2dParams& conv2dParams)
```

#### 参数说明

**表1 **接口参数说明

| **参数名称** | **类型** | **说明** |
| --- | --- | --- |
| dst | 输出 | 目的操作数。 Atlas 训练系列产品，支持的TPosition为：CO1，CO2 Atlas 推理系列产品AI Core，支持的TPosition为：CO1，CO2 结果中有效张量格式为[Cout/16, Ho, Wo, 16]，大小为Cout * Ho * Wo，Ho与Wo可以根据其他数据计算得出。 Ho = floor((H + pad_top + pad_bottom - dilation_h * (Kh - 1) - 1) / stride_h + 1) Wo = floor((W + pad_left + pad_right - dilation_w * (Kw - 1) - 1) / stride_w + 1) 由于硬件要求Ho*Wo需为16倍数，在申请dst Tensor时，shape应向上16对齐，实际申请shape大小应为Cout * round_howo。 round_howo = ceil(Ho * Wo /16) * 16。 |
| featureMap | 输入 | 输入张量，Tensor的TPosition为A1。 输入张量“feature_map”的形状，格式是[C1, H, W, C0]。 C1*C0为输入的channel数，要求如下： - 当feature_map的数据类型为half时，C0=16。- 当feature_map的数据类型为int8_t时，C0=32。- C1取值范围：[1,4], 输入的channel的范围：[16，32，64，128]。 H为高，取值范围：[1,40]。 W为宽，取值范围：[1,40]。 |
| weight | 输入 | 卷积核（权重）张量，Tensor的TPosition为B1。 卷积核张量“weight”的形状，格式是[C1, Kh, Kw, Cout, C0]。 C1*C0为输入的channel数，对于C0要求如下： - 当feature_map的数据类型为half时，C0=16。- 当feature_map的数据类型为int8_t时，C0=32。- C1取值范围：[1,4]。- kernel_shape输入的channel数需与fm_shape输入的channel数保持一致。 Cout为卷积核数目，取值范围：[16，32，64，128]， Cout必须为16的倍数。 Kh为卷积核高；值的范围：[1,5]。 Kw表示卷积核宽；值的范围：[1,5]。 |
| 输入矩阵形状等状态参数，类型为Conv2dParams。结构体具体定义为： ``` struct Conv2dParams {     uint32_t imgShape[CONV2D_IMG_SIZE];       // [H, W]     uint32_t kernelShapeIn[CONV2D_KERNEL_SIZE]; // [Kh, Kw]     uint32_t stride[CONV2D_STRIDE];          // [stride_h, stride_w]     uint32_t cin;                            // cin = C0 * C1;     uint32_t cout;     uint32_t padList[CONV2D_PAD];       // [pad_left, pad_right, pad_top, pad_bottom]     uint32_t dilation[CONV2D_DILATION]; // [dilation_h, dilation_w]     uint32_t initY;     uint32_t partialSum; }; ``` |  |  |
| 分形控制参数，类型为Conv2dTilling。结构体具体定义为： ``` struct Conv2dTilling {     const uint32_t blockSize = 16; // # M block size is always 16     LoopMode loopMode = LoopMode::MODE_NM;      uint32_t c0Size = 32;     uint32_t dTypeSize = 1;      uint32_t strideH = 0;     uint32_t strideW = 0;     uint32_t dilationH = 0;     uint32_t dilationW = 0;     uint32_t hi = 0;     uint32_t wi = 0;     uint32_t ho = 0;     uint32_t wo = 0;      uint32_t height = 0;     uint32_t width = 0;      uint32_t howo = 0;      uint32_t mNum = 0;     uint32_t nNum = 0;     uint32_t kNum = 0;      uint32_t mBlockNum = 0;     uint32_t kBlockNum = 0;     uint32_t nBlockNum = 0;      uint32_t roundM = 0;     uint32_t roundN = 0;     uint32_t roundK = 0;      uint32_t mTileBlock = 0;     uint32_t nTileBlock = 0;     uint32_t kTileBlock = 0;      uint32_t mIterNum = 0;     uint32_t nIterNum = 0;     uint32_t kIterNum = 0;      uint32_t mTileNums = 0;      bool mHasTail = false;     bool nHasTail = false;     bool kHasTail = false;      uint32_t kTailBlock = 0;     uint32_t mTailBlock = 0;     uint32_t nTailBlock = 0;      uint32_t mTailNums = 0; }; ``` |  |  |

**表2 **Conv2DParams结构体内参数说明：

| **参数名称** | **类型** | **说明** |
| --- | --- | --- |
| imgShape | vector<int> | 输入张量“feature_map”的形状，格式是[ H, W]。- H为高，取值范围：[1,40]。- W为宽，取值范围：[1,40]。 |
| kernelShape | vector<int> | 卷积核张量“weight”的形状，格式是[Kh, Kw]。 - Kh为高，取值范围：[1,5]。- Kw为宽，取值范围：[1,5]。 |
| stride | vector<int> | 卷积步长，格式是[stride_h, stride_w]。- stride_h表示步长高， 值的范围：[1,4]。- stride_w表示步长宽， 值的范围：[1,4]。 |
| cin | int | 分形排布参数，Cin = C1 * C0，Cin为输入的channel数，C1取值范围：[1,4]。 - 当feature_map的数据类型为float时，C0=8。输入的channel的范围：[8，16，24，32]。- 当feature_map的数据类型为half时，C0=16。输入的channel的范围：[16，32，48，64]。- 当feature_map的数据类型为int8_t时，C0=32。输入的channel的范围：[32，64，96，128]。 |
| cout | int | Cout为卷积核数目，取值范围：[16，32，64，128]， Cout必须为16的倍数。 |
| padList | vector<int> | padding行数/列数，格式是[pad_left, pad_right, pad_top, pad_bottom]。- pad_left为feature_map左侧pad列数，范围[0,4]。pad_right为feature_map右侧pad列数，范围[0,4]。- pad_top为feature_map顶部pad行数，范围[0,4]。- pad_bottom为feature_map底部pad行数，范围[0,4]。 |
| dilation | vector<int> | 空洞卷积参数，格式[dilation_h, dilation_w]。- dilation_h为空洞高，范围：[1,4]。- dilation_w为空洞宽，范围：[1,4]。  膨胀后卷积核宽为dilation_w * (Kw - 1) + 1，高为dilation_h * (Kh - 1) + 1。 |
| initY | uint32_t | 表示dst是否需要初始化。 - 取值0：不使用bias，L0C需要初始化，dst初始矩阵保存有之前结果，新计算结果会累加前一次conv2d计算结果。- 取值1：不使用bias，L0C不需要初始化，dst初始矩阵中数据无意义，计算结果直接覆盖dst中的数据。 |
| partialSum | uint32_t | 当dst参数所在的TPosition为CO2时，通过该参数控制计算结果是否搬出。- 取值0：搬出计算结果- 取值1：不搬出计算结果，可以进行后续计算 |

**表3 **Conv2dTilling结构体内参数说明

| **参数名称** | **类型** | **说明** |
| --- | --- | --- |
| blockSize | uint32_t | 固定值，恒为16，一个维度内存放的元素个数。 |
| 遍历模式，结构体具体定义为： ``` enum class LoopMode {     MODE_NM = 0,     MODE_MN = 1,     MODE_KM = 2,     MODE_KN = 3 }; ``` |  |  |
| c0Size | uint32_t | 一个block的字节长度，范围[16或者32]。 |
| dtypeSize | uint32_t | 传入的数据类型的字节长度，范围[1, 2]。 |
| strideH | uint32_t | 卷积步长-高，范围:[1,4]。 |
| strideW | uint32_t | 卷积步长-宽，范围:[1,4]。 |
| dilationH | uint32_t | 空洞卷积参数-高，范围：[1,4]。 |
| dilationW | uint32_t | 空洞卷积参数-宽，范围：[1,4]。 |
| hi | uint32_t | feature_map形状-高，范围：[1,40]。 |
| wi | uint32_t | feature_map形状-宽，范围：[1,40]。 |
| ho | uint32_t | feature_map形状-高，范围：[1,40]。 |
| wo | uint32_t | feature_map形状-宽，范围：[1,40]。 |
| height | uint32_t | weight形状-高，[1,5]。 |
| width | uint32_t | weight形状-宽，[1,5]。 |
| howo | uint32_t | feature_map形状大小，为ho * wo。 |
| mNum | uint32_t | M轴等效数据长度参数值，范围：[1,4096]。 |
| nNum | uint32_t | N轴等效数据长度参数值，范围：[1,4096]。 |
| kNum | uint32_t | K轴等效数据长度参数值，范围：[1,4096]。 |
| roundM | uint32_t | M轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1,4096]。 |
| roundN | uint32_t | N轴等效数据长度参数值且以blockSize为倍数向上取整，范围：[1,4096]。 |
| roundK | uint32_t | K轴等效数据长度参数值且以c0Size为倍数向上取整，范围：[1,4096]。 |
| mBlockNum | uint32_t | M轴Block个数，mBlockNum = mNum / blockSize，范围：[1,4096]。 |
| nBlockNum | uint32_t | N轴Block个数，nBlockNum = nNum / blockSize，范围：[1,4096]。 |
| kBlockNum | uint32_t | K轴Block个数，kBlockNum = kNum / blockSize，范围：[1,4096]。 |
| mIterNum | uint32_t | 遍历M轴维度数量，范围：[1,4096]。 |
| nIterNum | uint32_t | 遍历N轴维度数量，范围：[1,4096]。 |
| kIterNum | uint32_t | 遍历K轴维度数量，范围：[1,4096]。 |
| mTileBlock | uint32_t | M轴切分块个数，范围：[1,4096]。 |
| nTileBlock | uint32_t | N轴切分块个数，范围：[1,4096]。 |
| kTileBlock | uint32_t | K轴切分块个数，范围：[1,4096]。 |
| kTailBlock | uint32_t | K轴尾块个数，范围：[1,4096]。 |
| mTailBlock | uint32_t | M轴尾块个数，范围：[1,4096]。 |
| nTailBlock | uint32_t | N轴尾块个数，范围：[1,4096]。 |
| kHasTail | bool | K轴是否存在尾块。 |
| mHasTail | bool | M轴是否存在尾块。 |
| nHasTail | bool | N轴是否存在尾块。 |
| mTileNums | uint32_t | M轴切分块个数的长度，范围：[1,4096]。 |
| mTailNums | uint32_t | M轴尾块个数的长度，范围：[1,4096]。 |

**表4 **imgShape、kernelShape和dst的数据类型组合

| feature_map.dtype | weight.dtype | dst.dtype |
| --- | --- | --- |
| int8_t | int8_t | int32_t |
| half | half | float |
| half | half | half |

#### 约束说明

- 该接口当前不支持W=Kw并且H>Kh的场景，其将产生不可预期的结果。
