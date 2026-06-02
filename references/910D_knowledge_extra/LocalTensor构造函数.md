# LocalTensor构造函数<a name="ZH-CN_TOPIC_0000002523343536"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="23.1%" id="mcps1.1.4.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持（Pipe框架）</p>
</th>
<th class="cellrowborder" align="center" valign="top" width="23.26%" id="mcps1.1.4.1.3"><p id="p510433120182"><a name="p510433120182"></a><a name="p510433120182"></a>是否支持（静态Tensor编程）</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="23.1%" headers="mcps1.1.4.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
<td class="cellrowborder" align="center" valign="top" width="23.26%" headers="mcps1.1.4.1.3 "><p id="p1110513131811"><a name="p1110513131811"></a><a name="p1110513131811"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

LocalTensor构造函数。

## 函数原型<a name="section620mcpsimp"></a>

-   适用于[Pipe编程框架](编程范式.md)，通常情况下开发者不直接调用，该函数不会对LocaTensor成员变量赋初值，均为随机值。

    ```
    __aicore__ inline LocalTensor<T>() {}
    ```

-   适用于[静态Tensor编程](静态Tensor编程.md)，根据指定的逻辑位置/地址/长度，返回Tensor对象。

    ```
    __aicore__ inline LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize)
    
    __aicore__ inline LocalTensor<T>(uint32_t addr)
    
    
    
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="16.28%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.72%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="16.28%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.72%" headers="mcps1.2.3.1.2 "><a name="ul171781281310"></a><a name="ul171781281310"></a><ul id="ul171781281310"><li>适用于Pipe编程框架的原型，支持基础数据类型以及<a href="TensorTrait.md">TensorTrait</a>类型。</li><li>适用于静态Tensor编程的原型，支持的数据类型如下：<pre class="screen" id="screen148501140172610"><a name="screen148501140172610"></a><a name="screen148501140172610"></a>// 仅支持基础数据类型
__aicore__ inline LocalTensor&lt;T&gt;(TPosition pos, uint32_t addr, uint32_t tileSize)
// 仅支持TensorTrait类型
__aicore__ inline LocalTensor&lt;T&gt;(uint32_t addr)</pre>
</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.53%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.4%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.07000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.53%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>pos</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.07000000000001%" headers="mcps1.2.4.1.3 "><p id="p17825114223"><a name="p17825114223"></a><a name="p17825114223"></a>LocalTensor所在的逻辑位置。</p>
</td>
</tr>
<tr id="row74161018112218"><td class="cellrowborder" valign="top" width="16.53%" headers="mcps1.2.4.1.1 "><p id="p1041641811229"><a name="p1041641811229"></a><a name="p1041641811229"></a>addr</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p241611832219"><a name="p241611832219"></a><a name="p241611832219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.07000000000001%" headers="mcps1.2.4.1.3 "><p id="p241615183223"><a name="p241615183223"></a><a name="p241615183223"></a>LocalTensor的起始地址，其范围为[0, 对应物理内存最大值)。起始地址需要保证32字节对齐。</p>
</td>
</tr>
<tr id="row583211182317"><td class="cellrowborder" valign="top" width="16.53%" headers="mcps1.2.4.1.1 "><p id="p1083210116237"><a name="p1083210116237"></a><a name="p1083210116237"></a>tileSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.4.1.2 "><p id="p183281112316"><a name="p183281112316"></a><a name="p183281112316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.07000000000001%" headers="mcps1.2.4.1.3 "><p id="p20832711172314"><a name="p20832711172314"></a><a name="p20832711172314"></a>LocalTensor的元素个数，addr和tileSize（转换成所占字节数）之和不应超出对应物理内存的范围。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section17531157161314"></a>

本节提供了LocalTensor构造函数的使用示例和其所有成员函数的调用示例。

```
// srcLen = 256, num = 100, M=50
// 示例1
for (int32_t i = 0; i < srcLen; ++i) {
    inputLocal.SetValue(i, num); // 对inputLocal中第i个位置进行赋值为num
}
// 示例1结果如下：
// 数据(inputLocal): [100 100 100  ... 100]

// 示例2
for (int32_t i = 0; i < srcLen; ++i) {
    auto element = inputLocal.GetValue(i); // 获取inputLocal中第i个位置的数值
}
// 示例2结果如下：
// element 为100

// 示例3
for (int32_t i = 0; i < srcLen; ++i) {
    inputLocal(i) = num; // 对inputLocal中第i个位置进行赋值为num
}
// 示例3结果如下：
// 数据(inputLocal): [100 100 100  ... 100]

// 示例4
for (int32_t i = 0; i < srcLen; ++i) {
    auto element = inputLocal(i); // 获取inputLocal中第i个位置的数值
}
// 示例4结果如下：
// element 为100

// 示例5
auto size = inputLocal.GetSize(); // 获取inputLocal的长度，size大小为inputLocal有多少个元素
// 示例5结果如下：
// size大小为srcLen，256。

// 示例6
// operator[]使用方法, inputLocal[16]为从起始地址开始偏移量为16的新tensor
AscendC::Add(outputLocal[16], inputLocal[16], inputLocal2[16], M);
// 示例6结果如下：
// 输入数据(inputLocal): [100 100 100 ... 100]
// 输入数据(inputLocal2): [1 2 3 ... 66]
// 输出数据(outputLocal): [... 117 118 119 ... 166]

// 示例7
AscendC::TTagType tag = 10;
inputLocal.SetUserTag(tag); // 对LocalTensor设置tag信息。

// 示例8
AscendC::LocalTensor<half> tensor1 = que1.DeQue<half>();
AscendC::TTagType tag1 = tensor1.GetUserTag();
AscendC::LocalTensor<half> tensor2 = que2.DeQue<half>();
AscendC::TTagType tag2 = tensor2.GetUserTag();
AscendC::LocalTensor<half> tensor3 = que3.AllocTensor<half>();
/* 使用Tag控制条件语句执行*/
if ((tag1 <= 10) && (tag2 >= 9)) {
    AscendC::Add(tensor3, tensor1, tensor2, TILE_LENGTH); // 当tag1小于等于10，tag2大于等于9的时候，才能进行相加操作。
}
// 示例9
// input_local为int32_t 类型，包含16个元素(64字节)
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对inputLocal中第i个位置进行赋值为i
}

// 调用ReinterpretCast将input_local重解释为int16_t类型
AscendC::LocalTensor<int16_t> interpreTensor = inputLocal.ReinterpretCast<int16_t>();
// 示例9结果如下，二者数据完全一致，在物理内存上也是同一地址，仅根据不同类型进行了重解释
// inputLocal:0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// interpreTensor:0 0 1 0 2 0 3 0 4 0 5 0 6 0 7 0 8 0 9 0 10 0 11 0 12 0 13 0 14 0 15 0

// 示例10
// 调用GetPhyAddr()返回LocalTensor地址，CPU上返回的是指针类型(T*)，NPU上返回的是物理存储的地址(uint64_t)
#ifdef ASCEND_CPU_DEBUG
float *inputLocalCpuPtr = inputLocal.GetPhyAddr();
uint64_t realAddr = (uint64_t)inputLocalCpuPtr - (uint64_t)(GetTPipePtr()->GetBaseAddr(static_cast<int8_t>(AscendC::TPosition::VECCALC)));
#else
uint64_t realAddr = inputLocal.GetPhyAddr();
#endif

// 示例11
AscendC::TPosition srcPos = (AscendC::TPosition)inputLocal.GetPosition();
if (srcPos == AscendC::TPosition::VECCALC) {
    // 处理逻辑1
} else if (srcPos == AscendC::TPosition::A1) {
    // 处理逻辑2
} else {
    // 处理逻辑3
}

// 示例12
// 获取localTensor的长度(单位为Byte)，数据类型为int32_t，所以是16*sizeof(int32_t)
uint32_t len = inputLocal.GetLength();
// inputLocal:0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// len: 64

// 示例13 设置Tensor的ShapeInfo信息
AscendC::LocalTensor<float> maxUb = softmaxMaxBuf.template Get<float>();
uint32_t shapeArray[] = {16, 1024};
maxUb.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray, AscendC::DataFormat::ND));

// 示例14 获取Tensor的ShapeInfo信息
AscendC::ShapeInfo maxShapeInfo = maxUb.GetShapeInfo();
uint32_t orgShape0 = maxShapeInfo.originalShape[0];
uint32_t orgShape1 = maxShapeInfo.originalShape[1];
uint32_t orgShape2 = maxShapeInfo.originalShape[2];
uint32_t orgShape3 = maxShapeInfo.originalShape[3];
uint32_t shape2 = maxShapeInfo.shape[2];

// 示例15 SetAddrWithOffset，用于快速获取定义一个Tensor，同时指定新Tensor相对于旧Tensor首地址的偏移
// 需要注意，偏移的长度为旧Tensor的元素个数
AscendC::LocalTensor<float> tmpBuffer1 = tempBmm2Queue.AllocTensor<float>();
AscendC::LocalTensor<half> tmpHalfBuffer;
tmpHalfBuffer.SetAddrWithOffset(tmpBuffer1, calcSize * 2);

// 示例16 SetBufferLen 如下示例将申请的Tensor长度修改为1024(单位为字节)
AscendC::LocalTensor<float> tmpBuffer2 = tempBmm2Queue.AllocTensor<float>();
tmpBuffer2.SetBufferLen(1024);

// 示例17 SetSize 如下示例将申请的Tensor长度修改为256(单位为元素)
AscendC::LocalTensor<float> tmpBuffer3 = tempBmm2Queue.AllocTensor<float>();
tmpBuffer3.SetSize(256);

#ifdef ASCEND_CPU_DEBUG
// 示例18 只限于CPU调试，将LocalTensor数据Dump到文件中，用于精度调试，文件保存在执行目录
AscendC::LocalTensor<float> tmpTensor = softmaxMaxBuf.template Get<float>();
tmpTensor.ToFile("tmpTensor.bin");

// 示例19 只限于CPU调试，在调试窗口中打印LocalTensor数据用于精度调试，每一行打印一个datablock(32Bytes)的数据
AscendC::LocalTensor<int32_t> inputLocal = softmaxMaxBuf.template Get<int32_t>();
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对input_local中第i个位置进行赋值为i
}
inputLocal.Print();
// 0000: 0 1 2 3 4 5 6 7 8
// 0008: 9 10 11 12 13 14 15
#endif

// 示例20 在静态Tensor编程场景使用，根据传入的逻辑位置VECIN、起始地址128、元素个数32、数据类型float，构造出Tensor对象
uint32_t addr = 128;
uint32_t tileSize = 32;
AscendC::LocalTensor<float> tensor1 = AscendC::LocalTensor<float>(AscendC::TPosition::VECIN, addr, tileSize);
// 根据传入的TensorTrait信息、起始地址128构造出Tensor对象
// 其逻辑位置为VECIN，数据类型为float，Tensor元素个数为16*16*16
template <uint32_t v>
using UIntImm = Std::integral_constant<uint32_t, v>;
...
auto shape = AscendC::MakeShape(UIntImm<16>{}, UIntImm<16>{}, UIntImm<16>{});
auto stride = AscendC::MakeStride(UIntImm<0>{}, UIntImm<0>{}, UIntImm<0>{});
auto layoutMake = AscendC::MakeLayout(shape, stride);
auto tensorTraitMake = AscendC::MakeTensorTrait<float, AscendC::TPosition::VECIN>(layoutMake);
uint32_t addr = 128;
auto tensor1 = AscendC::LocalTensor<decltype(tensorTraitMake)>(addr);
```

