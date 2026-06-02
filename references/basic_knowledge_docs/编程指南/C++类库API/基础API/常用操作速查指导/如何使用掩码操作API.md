# 如何使用掩码操作API

**页面ID:** atlas_ascendc_10_0024  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0024.html

---

Mask用于控制矢量计算中参与计算的元素个数，支持以下工作模式及配置方式：

**表1 **Mask工作模式

| 工作模式 | 说明 |
| --- | --- |
| Normal模式 | 默认模式，支持单次迭代内的Mask能力，需要开发者配置迭代次数，额外进行尾块的计算。         **Normal模式下，Mask用来控制单次迭代内参与计算的元素个数。**         通过调用SetMaskNorm设置Normal模式。 |
| Counter模式 | 简化模式，直接传入计算数据量，自动推断迭代次数，不需要开发者去感知迭代次数、处理非对齐尾块的操作；但是不具备单次迭代内的Mask能力。         **Counter模式下，Mask表示整个矢量计算参与计算的元素个数。**         通过调用SetMaskCount设置Counter模式。 |

**表2 **Mask配置方式

| 配置方式 | 说明 |
| --- | --- |
| 接口传参（默认） | 通过矢量计算API的入参直接传递Mask值。矢量计算API的模板参数isSetMask（仅部分API支持）用于控制接口传参还是外部API配置，默认值为true，表示接口传参。Mask对应于高维切分计算API中的mask/mask[]参数或者tensor前n个数据计算API中的calCount参数。 |
| 外部API配置 | 调用SetVectorMask接口设置Mask值，矢量计算API的模板参数isSetMask设置为false，接口入参中的Mask参数（对应于高维切分计算API中的mask/mask[]参数或者tensor前n个数据计算API中的calCount参数）不生效。适用于Mask参数相同，多次重复使用的场景，无需在矢量计算API内部反复设置，会有一定的性能优势。 |

Mask操作的使用方式如下：

**表3 **Mask操作的使用方式

| 配置方式 | 工作模式 | 前n个数据计算API | 高维切分计算API |
| --- | --- | --- | --- |
| 接口传参 | Normal模式 | 不涉及。 | isSetMask模板参数设置为true，通过接口入参传入Mask，根据使用场景配置dataBlockStride、repeatStride、repeatTime参数。 |
| Counter模式 | isSetMask模板参数设置为true，通过接口入参传入Mask。 | - isSetMask模板参数设置为true，通过接口入参传入Mask。          - 根据使用场景配置dataBlockStride、repeatStride参数。repeatTime传入固定值即可，建议统一设置为1，该值不生效。 |  |
| 外部API配置 | Normal模式 | 不涉及。 | 调用SetVectorMask设置Mask，之后调用高维切分计算API。                     - isSetMask模板参数设置为false，接口入参中的mask值设置为占位符MASK_PLACEHOLDER，用于占位，无实际含义。           - 根据使用场景配置repeatTime、dataBlockStride、repeatStride参数。 |
| Counter模式 | 调用SetVectorMask设置Mask，之后调用前n个数据计算API，isSetMask模板参数设置为false；接口入参中的calCount建议设置成1。 | 调用SetVectorMask设置Mask，之后调用高维切分计算API。                     - isSetMask模板参数设置为false；接口入参中的mask值设置为MASK_PLACEHOLDER，用于占位，无实际含义。           - 根据使用场景配置dataBlockStride、repeatStride参数。repeatTime传入固定值即可，建议统一设置为1，该值不生效。 |  |

典型场景的使用示例如下：

- 场景1：Normal模式 + 外部API配置 + 高维切分计算API

```
AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;

// 1、设置Normal模式
AscendC::SetMaskNorm();
// 2、设置Mask
AscendC::SetVectorMask<half, AscendC::MaskMode::NORMAL>(0xffffffffffffffff, 0xffffffffffffffff);  // 逐bit模式
// SetVectorMask<half, MaskMode::NORMAL>(128);  // 连续模式

// 3、多次调用矢量计算API, isSetMask模板参数设置为false，接口入参中的mask值设置为占位符MASK_PLACEHOLDER，用于占位，无实际含义
// 根据使用场景配置repeatTime、dataBlockStride、repeatStride参数
// dstBlkStride, src0BlkStride, src1BlkStride = 1, 单次迭代内数据连续读取和写入
// dstRepStride, src0RepStride, src1RepStride = 8, 相邻迭代间数据连续读取和写入
AscendC::Add<half, false>(dstLocal, src0Local, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 2, 2, 2, 8, 8, 8 });
AscendC::Sub<half, false>(src0Local, dstLocal, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 2, 2, 2, 8, 8, 8 });
AscendC::Mul<half, false>(src1Local, dstLocal, src0Local, AscendC::MASK_PLACEHOLDER, 1, { 2, 2, 2, 8, 8, 8 });
// 4、恢复Mask值为默认值
AscendC::ResetMask();
```

- 场景2：Counter模式 + 外部API配置 + 高维切分计算API

```
AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;
int32_t len = 128;  // 参与计算的元素个数
// 1、设置Counter模式
AscendC::SetMaskCount();
// 2、设置Mask
AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(len);
// 3、多次调用矢量计算API, isSetMask模板参数设置为false；接口入参中的mask值设置为MASK_PLACEHOLDER，用于占位，无实际含义
// 根据使用场景正确配置dataBlockStride、repeatStride参数。repeatTime传入固定值即可，建议统一设置为1，该值不生效
AscendC::Add<half, false>(dstLocal, src0Local, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
AscendC::Sub<half, false>(src0Local, dstLocal, src1Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
AscendC::Mul<half, false>(src1Local, dstLocal, src0Local, AscendC::MASK_PLACEHOLDER, 1, { 1, 1, 1, 8, 8, 8 });
// 4、恢复工作模式
AscendC::SetMaskNorm();
// 5、恢复Mask值为默认值
AscendC::ResetMask();
```

- 场景3：Counter模式 + 外部API配置 + 前n个数据计算接口配合使用

```
AscendC::LocalTensor<half> dstLocal;
AscendC::LocalTensor<half> src0Local;
half num = 2; 
// 1、设置Mask
AscendC::SetVectorMask<half, AscendC::MaskMode::COUNTER>(128); // 参与计算的元素个数为128
// 2、调用前n个数据计算API，isSetMask模板参数设置为false；接口入参中的calCount建议设置成1。
AscendC::Adds<half, false>(dstLocal, src0Local, num, 1);
AscendC::Muls<half, false>(dstLocal, src0Local, num, 1);
// 3、恢复工作模式
AscendC::SetMaskNorm();
// 4、恢复Mask值为默认值
AscendC::ResetMask();
```

> **注意:** 

- 前n个数据计算API接口内部会设置工作模式为Counter模式，所以如果前n个数据计算API配合Counter模式使用时，无需手动调用SetMaskCount设置Counter模式。
- 所有手动使用Counter模式的场景，使用完毕后，需要调用SetMaskNorm恢复工作模式。
- 调用SetVectorMask设置Mask，使用完毕后，需要调用ResetMask恢复Mask值为默认值。
- 使用高维切分计算API配套Counter模式使用时，比前n个数据计算API增加了可间隔的计算，支持dataBlockStride、repeatStride参数。
