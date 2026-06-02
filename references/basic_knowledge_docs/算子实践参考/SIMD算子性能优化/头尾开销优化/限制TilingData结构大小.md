# 限制TilingData结构大小

**页面ID:** atlas_ascendc_best_practices_10_0018  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0018.html

---

【优先级】中

【描述】TilingData结构是Tiling切分信息的载体，当Host侧按照Tiling切分策略计算完Tiling后，算子会以入参的方式将Tiling切分信息从Host侧传递到Device侧，此时Tiling信息存放在GM上。调用GET_TILING_DATA宏后，会将Tiling信息从GM拷贝到AI处理器的栈空间上，期间会有拷贝开销，由于GM访问效率较低，同时考虑到栈空间限制，需要限制TilingData结构大小。拷贝耗时为us级别，在小shape的场景下，进行此类优化收益会更加明显。

限制TilingData结构大小，可以从以下方面考虑：

- 减少不必要的TilingData结构变量；
- 根据Tiling的数据范围选择合适的变量类型；
- 合理排布TilingData结构；
- TilingData整体结构要求8字节补齐。

【反例】

- 如下的示例中存在TilingData结构变量冗余的情况：BlockDim信息已经通过SetBlockDim接口进行设置，可以在Kernel侧调用GetBlockNum接口获取，无需通过TilingData结构传递。
- 此外，变量的数据类型也不合理：formerNum和tailNum分别为计算整块数据的核数和计算尾块数据的核数，不会超过BLOCK_DIM的值，使用uint8_t类型即可；formerLength等变量根据其计算逻辑，不会超出uint32_t的范围，使用uint32_t类型即可。

```
// Tiling结构体定义
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
  TILING_DATA_FIELD_DEF(uint64_t, blockDim);
  TILING_DATA_FIELD_DEF(uint64_t, formerNum);
  TILING_DATA_FIELD_DEF(uint64_t, tailNum);
  TILING_DATA_FIELD_DEF(uint64_t, formerLength);
  TILING_DATA_FIELD_DEF(uint64_t, tailLength);
  TILING_DATA_FIELD_DEF(uint64_t, alignNum);
END_TILING_DATA_DEF;
```

```
// Host侧Tiling函数计算Tiling结构信息
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_HALF;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingDataUnalign tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    // BlockDim信息已经通过SetBlockDim接口进行设置
    context->SetBlockDim(BLOCK_DIM);
    uint32_t totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    // formerNum、tailNum保证不超过0-BLOCK_DIM数据范围
    uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM;
    uint32_t tailNum = BLOCK_DIM - formerNum;
    // formerLength等变量根据其计算逻辑，不会超出uint32_t的范围   
    uint32_t formerLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t tailLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM;
    ...
}
```

【正例】

Tiling变量无冗余，变量数据类型最小化。

```
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
  TILING_DATA_FIELD_DEF(uint8_t, formerNum);
  TILING_DATA_FIELD_DEF(uint8_t, tailNum); 
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);
END_TILING_DATA_DEF;
```

【反例】

如下的示例中TilingData结构不合理：由于AI处理器访存需要8字节对齐，在用户定义TilingData结构后，Ascend C工程框架会按照8字节对齐的方式对字节进行补齐，并保证整体TilingData结构满足8字节对齐要求。如下TilingData结构formerNum和tailNum变量都会补充3个字节，整体TilingData结构会因为8字节对齐再补充4个字节，该TilingData结构共计补充**10**个字节。

```
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
  TILING_DATA_FIELD_DEF(uint8_t, formerNum); // 需补充3个字节，使得formerLength变量访问无误
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint8_t, tailNum); // 需补充3个字节，使得tailLength变量访问无误
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);// 需补充4个字节，使得下个TilingData结构访问无误
END_TILING_DATA_DEF;
```

【正例】

如下的示例中，对Tiling参数的排布进行了调整，字节排布合理，只需要补充**2**个字节，达到了减小TilingData结构的目的。

```
BEGIN_TILING_DATA_DEF(TilingDataUnalign)
  TILING_DATA_FIELD_DEF(uint8_t, formerNum);
  TILING_DATA_FIELD_DEF(uint8_t, tailNum); // 需补充2个字节，使得formerLength变量访问无误
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);
END_TILING_DATA_DEF;
```
