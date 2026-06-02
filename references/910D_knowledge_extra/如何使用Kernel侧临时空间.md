# 如何使用Kernel侧临时空间<a name="ZH-CN_TOPIC_0000002554351457"></a>

Kernel侧接口的内部实现一般涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。除矩阵计算、HCCL通信类、卷积计算等，对于多数高阶API中临时空间的处理，开发者可以通过Kernel侧接口的入参sharedTmpBuffer传入提前申请的临时空间、通过接口框架申请临时空间两种方式。

-   通过sharedTmpBuffer入参传入，Kernel侧接口使用该传入的Tensor作为临时空间。该方式下，开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需在Kernel侧申请临时空间，但是需要预留临时空间的大小，即在分配内存空间时，应在可用空间大小中减去需预留的临时空间大小。

无论开发者采用上述哪种方式，在申请Tensor空间或预留临时空间时，都需要提前获取Kernel侧接口所需的临时空间大小BufferSize，为此相应类别API中提供了GetxxxMaxMinTmpSize接口，用于获取所需预留空间的大小范围，其中xxx为对应的Kernel侧接口。开发者在Host侧调用GetxxxMaxMinTmpSize接口，获取预留/申请的最大和最小临时空间大小，基于此范围选择合适的空间大小作为Tiling参数传递到Kernel侧使用。

-   为保证功能正确，预留/申请的临时空间大小不能小于最小临时空间大小；
-   在最小临时空间-最大临时空间范围内，随着临时空间增大，Kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间预留/申请。

以Asin接口为例：

```
// 算子输入的数据类型T为half，isReuseSource传入默认值false
auto shape_input = context->GetInputTensor(0)->GetOriginShape();    
std::vector<int64_t> srcDims = {shape_input.GetDim(0), shape_input.GetDim(1)};
uint32_t srcSize = 1;
for (auto dim : srcDims) {
    srcSize *= dim;
}
uint32_t typeSize = 2;
ge::Shape shape(srcDims);
uint32_t minValue = 0;
uint32_t maxValue = 0;
AscendC::GetAsinMaxMinTmpSize(shape, typeSize, false, maxValue, minValue);

auto platformInfo = context->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
uint64_t tailSize = 0; // UB剩余空间大小
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tailSize); // 本样例中使用完整的ub空间，实际情况下tailSize需要减掉用户已使用的UB空间
auto tmpSize = tailSize >= maxValue ? maxValue : tailSize;

AsinCustomTilingData tiling;
tiling.set_tmpBufferSize(tmpSize); // 将临时空间大小设置为Tiling参数
```

另外，多数高阶API中提供了GetxxxTmpBufferFactorSize接口，该接口用于获取maxLiveNodeCnt和extraBuf，maxLiveNodeCnt表示临时空间是单次计算数据量所占空间的多少倍；extraBuf表示Kernel侧接口所需的临时空间大小。在固定空间大小的情况下，通过maxLiveNodeCnt和extraBuf可以推算算子单次最大计算元素数量。

推算示例如下：

-   算子实现需要调用Mean接口，开发者为其预留currBuff大小的空间（即总可用空间），利用GetMeanTmpBufferFactorSize接口得到maxLiveNodeCnt、extraBuf输出值，可推导算子单次最大计算元素数量为：

    currentShapeSize = \(currBuff - extraBuf\) / maxLiveNodeCnt / typeSize

-   算子实现需要调用两个Kernel侧API KernelIntf1、KernelIntf2，利用两个GetXxxTmpBufferFactorSize（其中Xxx为需要调用的两个高阶API）接口的两组输出值\(maxLiveNodeCnt、extraBuf\)以及当前现有的临时空间currBuff，推导单次最大计算元素数量currentShapeSize为：

    currentShapeSize1 = \(currBuff - extraBuf1\) / maxLiveNodeCnt1 / typeSize

    currentShapeSize2 = \(currBuff - extraBuf2\) / maxLiveNodeCnt2 / typeSize

    currentShapeSize = min\(currentShapeSize1 , currentShapeSize2\)

注意上文中的currBuff表示接口计算可用的空间，需要去除用户输入输出等空间。

以算子中需要同时调用Asin、Acos接口为例：

```
// 算子输入的数据类型T为half
auto shape_input = context->GetInputTensor(0)->GetOriginShape();
std::vector<int64_t> srcDims = { shape_input.GetDim(0), shape_input.GetDim(1) };
uint32_t srcSize = 1;
uint32_t srcCurSize = 1;
for (auto dim : srcDims) {
    srcSize *= dim;
}
uint32_t typeSize = 2;

auto platformInfo = context->GetPlatformInfo();
auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
uint64_t tailSize = 0; // UB剩余空间大小
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tailSize);

uint32_t asinMaxLiveNodeCount = 0;
uint32_t asinExtraBuf = 0;

uint32_t acosMaxLiveNodeCount = 0;
uint32_t acosExtraBuf = 0;

AscendC::GetAsinTmpBufferFactorSize(typeSize, asinMaxLiveNodeCount, asinExtraBuf);
AscendC::GetAcosTmpBufferFactorSize(typeSize, acosMaxLiveNodeCount, acosExtraBuf);
// tmp的大小需要减去UB上调用api接口输入和输出占用的大小
// 该示例中包括Asin接口的输入输出，以及Acos的输入输出，其中Asin接口的输出作为Acos的输入，因此一共需要3份src的空间大小
auto tmpSize = tailSize - srcSize * typeSize * 3;
assert(tmpSize >= asinExtraBuf);
assert(tmpSize >= acosExtraBuf);
// 计算Asin算子单次最大计算元素数量
if (asinMaxLiveNodeCount != 0) {
    srcAsinCurSize = (tmpSize - asinExtraBuf) / asinMaxLiveNodeCount / typeSize;
} else {
    srcAsinCurSize = srcSize;
}
// 计算Acos算子单次最大计算元素数量
if (acosMaxLiveNodeCount != 0) {
    srcAcosCurSize = (tmpSize - acosExtraBuf) / acosMaxLiveNodeCount / typeSize; 
} else {
    srcAcosCurSize = srcSize;
}
srcCurSize = std::min(srcAsinCurSize, srcAcosCurSize);

AsinCustomTilingData tiling;
tiling.set_srcCurSize(srcCurSize); // 将单次最大计算元素数量设置为Tiling参数
```

