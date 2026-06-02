# VF循环优化<a name="ZH-CN_TOPIC_0000002523129100"></a>

Ascend 950PR/Ascend 950DT对应的架构中，Vector Function（VF）是实现高性能向量计算的核心载体。VF函数中可以包含最多四层嵌套循环，每层循环中还可以包含多个串行循环，同时支持非循环的向量操作和标量操作。VF循环对控制结构的支持有限，仅支持for循环和条件判断，不支持switch、do-while和 while-do等其他控制结构。VF循环通过尽可能优化为硬件级向量循环（Hardware Loop），从而实现性能优化。

当VF函数中的循环满足[Hardware Loop编码规范](#section11326136133217)时，会被编译器优化为Hardware Loop，提升整体的编码性能，否则它的循环逻辑会由迭代变量和条件判断语句构成Software Loop，无法使能VF循环优化。

在遵循Hardware Loop编码规范，确保循环可被优化为Hardware Loop基础上，可以通过成员变量访问、指令分布优化和地址管理优化等方面进一步提升性能。

## Hardware Loop编码规范<a name="section11326136133217"></a>

为了能让编译器识别并生成Hardware Loop，对应的Loop代码必须符合硬件设计的要求。具体规范如下：

-   迭代变量类型

    VF内所有Loop的迭代变量必须是uint16\_t 类型。

-   起始值与步长

    循环起始值从0开始。

    每次迭代的步长必须是递增1。

-   循环內不允许跳转指令，比如条件判断跳转, 如if/else，三元运算符?。

    VF内if/else在Loop内会阻碍Hardware Loop的生成，编译器虽然会尽可能的做if/else消除优化，但是不做完全性保证。

    if控制流可使用if constexpr或者for\(1\)替换。if constexpr在编译时已经完成，无运行时开销，但是需要传入的参数是编译时常量，不能依赖运行时变量；for\(1\)可以触发编译器的循环优化，在当前硬件条件下，Vector侧执行Loop的性能是远远优于条件分支跳转的。

    本示例展示了尾块处理场景，当尾块大小不为0时，hasTail为1，通过for\(1\)来代替if\(hasTail\)的判断，提高了Loop的性能。

    ```
    //【反例】使用if语句
    uint16_t tailK = srcK % floatRepSize;
    uint16_t hasTail = 0;
    // 通过!!tailK ，对是否产生尾块进行判断，如果srcK % floatRepSize余数为0，则hasTail对应bool值为0（false）,否则为1（true）
    hasTail = !!tailK;
    if(tailK > 0){
      // 尾块处理内容
    }
    ```

    ```
    //【正例】使用for(1)来替代if判断语句
    uint16_t tailK = srcK % floatRepSize;
    uint16_t hasTail = 0;
    hasTail = !!tailK;
    for(uint16_t i = 0 ; i < hasTail ; i++){
      // 尾块处理内容
    }
    ```

-   一旦执行，循环计数/边界不允许被更改。
-   若要利用外层循环的计数作为循环边界，将外层循环计数器移动到另一个寄存器中，然后将其设置为循环边界。

下文示例展示了编译器识别并处理为Hardware Loop和Software Loop的两种场景：

```
//【正例】被编译器优化为Hardware Loop
// 嵌套循环
for (uint16_t i = 0; i < LoopBound; i++) {            
    for (uint16_t j = 0; j < LoopBound; j++) {   
        for (uint16_t k = 0; k < LoopBound; k++) {   
            for (uint16_t m = 0; m < LoopBound; m++) {  
              // ...
            }
         }
     }
}
```

```
//【反例】无法被编译器优化，构成Software Loop
for (uint16_t i = 0; i < LoopBound; i++) {     // Software Loop，循环内包含了if判断
    if(){
    }
}
 
for (uint16_t i = 2; i < LoopBound*3; i+=2){  // Software Loop，循环起始值不为0，循环步长不为1
...
}
```

## 循环内成员变量访问优化<a name="section168682445382"></a>

在一个VF内，不推荐直接访问类的成员变量。直接访问类的成员变量相当于从栈上把内容搬运到Tensor寄存器中，之后通过地址访问Tensor上的内容，该操作将导致VF融合失效，建议通过局部变量临时传参解决，示例如下：

```
//【反例】直接读取成员变量
__aicore__ inline void SoftMaxGenericNDImpVF(__ubuf__ float* dstAddr, __ubuf__ float* sumAddr, __ubuf__ float* maxAddr,
    __ubuf__ float* srcAddr, __ubuf__ float* workAddr, const LastAxisShapeND originalSrcShape, const SoftMaxTiling tiling)
{
    for (uint16_t i = 0; i < (uint16_t)tiling.srcM; i++) {
        AscendC::ReduceMax(maxAddr + i * FLOAT_NUM_PER_BLK, srcAddr + i * tiling.srcK, workAddr, (uint16_t)originalSrcShape.k);
    }
}
```

```
//【正例】通过局部变量传递VF内访问的成员变量
__aicore__ inline void SoftMaxGenericNDImpVF(__ubuf__ float* dstAddr, __ubuf__ float* sumAddr, __ubuf__ float* maxAddr,
    __ubuf__ float* srcAddr, __ubuf__ float* workAddr, const LastAxisShapeND originalSrcShape, const SoftMaxTiling tiling)
{
    uint16_t srcK = tiling.srcK;
    uint16_t srcM = tiling.srcM;
    uint16_t reduceK = FLOAT_NUM_PER_BLK;
    uint16_t originK = (uint16_t)originalSrcShape.k;
    for (uint16_t i = 0; i < (uint16_t)srcM; i++) {
        AscendC::ReduceMax(maxAddr + i * reduceK, srcAddr + i * srcK, workAddr, originK);
    }
}
```

## 循环内指令分布优化<a name="section214134513910"></a>

减少Loop循环内非索引相关语句。for循环中存在与索引无关的语句可以提出for循环外来减少指令数。

```
//【反例】Duplicate语句放在for循环中，每次循环都会执行一次
template<typename T>
__simd_vf__ inline void DuplicateVF(__ubuf__ T* dstAddr, T scalarValue, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();  
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::Duplicate(dstReg, scalarValue);
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

```
//【正例】Duplicate放在for循环外，仅执行一次，有效减少指令数
template<typename T>
__simd_vf__ inline void DuplicateVF(__ubuf__ T* dstAddr, T scalarValue, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();  
    AscendC::MicroAPI::Duplicate(dstReg, scalarValue);
    for (uint16_t i = 0; i < repeatTimes; i++) {
        AscendC::MicroAPI::StoreAlign(dstAddr + i * oneRepeatSize, dstReg, mask);
    }
}
```

## 循环内地址管理优化<a name="section2717961412"></a>

在VF循环中，当使用搬运指令时，需要计算搬入和搬出的地址偏移量，由此会引入较多的标量计算开销。在Ascend 950PR/Ascend 950DT中，通过引入地址寄存器，可有效优化地址偏移量的计算。当满足如下的地址寄存器生成模式时，编译器有机会生成地址寄存器，从而消除相关的Scalar计算消耗，提升整体性能。

其中，地址寄存器最多支持4层循环寻址，如下图所示。

<!-- img2text -->
```
┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐   ┌─────────┐   ┌─────────┐
│         │   │         │   │         │        │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘        └─────────┘   └─────────┘   └─────────┘

┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐   ┌─────────┐   ┌─────────┐
│         │   │         │   │         │        │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘        └─────────┘   └─────────┘   └─────────┘


┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐   ┌─────────┐   ┌─────────┐
│         │   │         │   │         │        │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘        └─────────┘   └─────────┘   └─────────┘

┌─────────┐   ┌─────────┐   ┌─────────┐        ┌─────────┐   ┌─────────┐   ┌─────────┐
│         │   │         │   │         │        │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘        └─────────┘   └─────────┘   └─────────┘
```

说明:
- 图片中仅可辨识为 4 行 × 6 列的小矩形块阵列，中间分为左右两组，每组各 3 列。
- 未能从图片中清晰识别到任何文字标注、箭头、参数名或连接关系。
- 可辨识的颜色分组仅用于区分区域：左上两行为一组，右上两行为一组，左下两行为一组，右下两行为一组。

如下是一个满足地址寄存器生成模式的代码示例，源操作数地址按照上图的方式进行最多四维的寻址：

```
for(uint16_t i = 0;i < extent1; i++){
    for(uint16_t j = 0;j < extent2; j++){
        for(uint16_t k = 0;k < extent3; k++){
            for(uint16_t m = 0;m < extent4; m++){
               AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + i * const1 + j * const2 + k * const3 + m * const4);
            }
        }
    }
}
```

编译器会将以上模式优化为使用AddrReg进行地址管理，优化为以下模式：

```
AscendC::MicroAPI::AddrReg aReg;
for(uint16_t i = 0;i < extent1; i++){
    for(uint16_t j = 0;j < extent2; j++){
        for(uint16_t k = 0;k < extent3; k++){
            for(uint16_t m = 0;m < extent4; m++){
                aReg = AscendC::MicroAPI::CreateAddrReg(i, const1, j, const2, k, const3, m, const4);
                AscendC::MicroAPI::LoadAlign(srcReg, srcAddr, aReg);
            }
        }
    }
}
```

编译器通过识别LoadAlign、StoreAlign等指令进行模式匹配，当代码结构满足特定优化模式时，编译器将有机会进行高效优化，从而获得最佳的性能收益。相反，若直接使用AddrReg存储偏移量或其它底层接口，可能会限制编译器的全局优化能力。

特别地，当四层循环构成连续访问场景时，地址管理可被简化为一维模式，从而实现更高效的搬运优化，进一步提升数据访问的局部性和执行效率。

```
// 使能向量地址生成指令优化
__simd_vf__ inline void ComputeModeVF(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<T> dstReg;
    AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<T>();
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        AscendC::MicroAPI::LoadAlign(dstReg, srcAddr+ i * oneRepeatSize);
        AscendC::MicroAPI::StoreAlign(dstAddr, dstReg, i * oneRepeatSize, mask);
    }
}
```

