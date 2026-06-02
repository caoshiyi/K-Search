# SoftMax Tiling使用说明<a name="ZH-CN_TOPIC_0000002523304144"></a>

Ascend C提供一组SoftMax Tiling API，方便用户获取SoftMax kernel计算时所需的Tiling参数。阅读本节之前，请先参考[Tiling实现](Host侧Tiling实现.md)了解Tiling实现基本流程。

获取Tiling参数主要分为如下两步：

1.  获取SoftMax接口计算所需最小和最大临时空间大小，注意该步骤不是必须的，只是作为一个参考，供合理分配计算空间。
2.  获取输入SoftMax kernel侧接口所需tiling参数，需要传入输入shape、剩余的可供softmax接口计算的空间大小和计算的数据类型大小。

    SoftMax Tiling结构体的定义如下，开发者无需关注该tiling结构的具体信息，只需要传递到kernel侧，传入SoftMax高阶API接口，直接进行使用即可。

    ```
    struct SoftMaxTiling {
        uint32_t srcM = 0;
        uint32_t srcK = 0;
        uint32_t srcSize = 0;
        uint32_t outMaxM = 0;
        uint32_t outMaxK = 0;
        uint32_t outMaxSize = 0;
        uint32_t splitM = 0;
        uint32_t splitK = 0;
        uint32_t splitSize = 0;
        uint32_t reduceM = 0;
        uint32_t reduceK = 0;
        uint32_t reduceSize = 0;
        uint32_t rangeM = 0;
        uint32_t tailM = 0;
        uint32_t tailSplitSize = 0;
        uint32_t tailReduceSize = 0;
    };
    ```

对于SoftMax/SimpleSoftMax请参考[SoftMax/SimpleSoftMax Tiling](SoftMax-SimpleSoftMax-Tiling.md)；

对于SoftmaxFlash请参考[SoftmaxFlash Tiling接口](SoftmaxFlash-Tiling接口.md)；

对于SoftmaxGrad请参考[SoftmaxGrad Tiling接口](SoftmaxGrad-Tiling接口.md)；

对于SoftmaxFlashV2请参考[SoftmaxFlashV2 Tiling接口](SoftmaxFlashV2-Tiling接口.md)；

判断SoftMaxTiling是否为基本块Tiling请参考[IsBasicBlockInSoftMax](IsBasicBlockInSoftMax.md)。

## 调用示例<a name="section94691236101419"></a>

如下样例介绍了使用SoftMax高阶API时host侧获取Tiling参数的流程以及该参数如何在kernel侧使用。样例中输入Tensor的shape大小为\[320,64\]，输入的数据类型为half。

1.  将SoftMaxTiling结构体参数增加至TilingData结构体，作为TilingData结构体的一个字段。

    ```
    BEGIN_TILING_DATA_DEF(TilingData)               // 注册一个tiling的类，以tiling的名字作为入参
      TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 添加tiling字段，总计算数据量
      TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 添加tiling字段，每个核上总计算数据分块个数
      ...                                           // 添加其他tiling字段
      TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData); // 将SoftMaxTiling结构体参数增加至TilingData结构体
    END_TILING_DATA_DEF;
    ```

2.  Tiling实现函数中，首先调用**GetSoftMaxMaxTmpSize/GetSoftMaxMinTmpSize**接口获取SoftMax接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小；然后根据输入shape、剩余的可供计算的空间大小等信息获取SoftMax kernel侧接口所需tiling参数。

    ```
    namespace optiling {
    const uint32_t NUM_BLOCKS = 8;
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        TilingData tiling;
        uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
        context->SetBlockDim(NUM_BLOCKS);
        tiling.set_totalLength(totalLength);
        tiling.set_tileNum(TILE_NUM);
        // 设置其他Tiling参数
        ... 
        std::vector<int64_t> shapeVec = {320,64};
        ge::Shape srcShape(shapeVec);
        // 本样例中仅作为样例说明，通过GetSoftMaxMinTmpSize获取最小值并传入，来保证功能正确，开发者可以根据需要传入合适的空间大小
        const uint32_t localWorkSpaceSize = AscendC::GetSoftMaxMinTmpSize(srcShape, sizeof(half), false);
        // 获取SoftMax Tiling参数
        AscendC::SoftMaxTilingFunc(srcShape, sizeof(half), localWorkSpaceSize, tiling.softmaxTilingData); 
         ... // 其他逻辑
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->SetTilingKey(1);
        return ge::GRAPH_SUCCESS;
    }
    } // namespace optiling
    ```

3.  对应的kernel侧通过在核函数中调用GET\_TILING\_DATA获取TilingData，继而将TilingData中的SoftMax Tiling信息传入SoftMax接口参与计算。完整的kernel侧样例请参考[调用示例](SoftMax.md#section94691236101419)。

    ```
    extern "C" __global__ __aicore__ void func_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        KernelFunc op;
        op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum,tilingData.SoftMaxTiling);
        if (TILING_KEY_IS(1)) {
            op.Process();
        }
    }
    ```

