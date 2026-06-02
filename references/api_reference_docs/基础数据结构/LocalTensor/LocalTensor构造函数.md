# LocalTensor构造函数

**页面ID:** atlasascendc_api_07_00101  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00101.html

---

#### 产品支持情况

| 产品 | 是否支持（Pipe框架） | 是否支持（静态Tensor编程） |
| --- | --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ | √ |
| Atlas 200I/500 A2 推理产品 | √ | x |
| Atlas 推理系列产品            AI Core | √ | √ |
| Atlas 推理系列产品            Vector Core | √ | √ |
| Atlas 训练系列产品 | √ | x |

#### 功能说明

LocalTensor构造函数。

#### 函数原型

- 适用于Pipe编程框架，通常情况下开发者不直接调用，该函数不会对LocaTensor成员变量赋初值，均为随机值。

```
__aicore__ inline LocalTensor<T>() {}
```

- 适用于静态Tensor编程，根据指定的逻辑位置/地址/长度，返回Tensor对象。

```
__aicore__ inline LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize)
__aicore__ inline LocalTensor<T>(uint32_t addr)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | - 适用于Pipe编程框架的原型，支持基础数据类型以及TensorTrait类型。           - 适用于静态Tensor编程的原型，支持的数据类型如下：            ``` // 仅支持基础数据类型 __aicore__ inline LocalTensor<T>(TPosition pos, uint32_t addr, uint32_t tileSize) // 仅支持TensorTrait类型 __aicore__ inline LocalTensor<T>(uint32_t addr) ``` |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pos | 输入 | LocalTensor所在的逻辑位置。 |
| addr | 输入 | LocalTensor的起始地址，其范围为[0, 对应物理内存最大值)。起始地址需要保证32字节对齐。 |
| tileSize | 输入 | LocalTensor的元素个数，addr和tileSize（转换成所占字节数）之和不应超出对应物理内存的范围。 |

#### 约束说明

无

#### 调用示例

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
