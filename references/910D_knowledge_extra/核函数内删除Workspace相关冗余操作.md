# 核函数内删除Workspace相关冗余操作<a name="ZH-CN_TOPIC_0000002554409035"></a>

【优先级】中

【描述】在Ascend C算子工程中，编写核函数时传入的参数workspace已经直接赋值为用户Workspace，因此无需再通过SetSysWorkspace和GetUserWorkspace来设置和获取Workspace。减少这些冗余判断后，编译器可以在不使用该参数的情况下进一步优化未用到的workspace变量。

【反例】

fast\_gelu函数的参数workspace等价于用户workspace，且不为空，仍然对workspace进行判空，并且设置SetSysWorkspace和GetUserWorkspace来获取用户Workspace。

```
template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 反例，冗余判断
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (dType == static_cast<uint64_t>(TPL_FP16)) {
        ElementwiseSch<schMode, FastGeluDag::FastGeluNeedCast<half>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_BF16)) {
        ElementwiseSch<schMode, FastGeluDag::FastGeluNeedCast<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_FP32)) {
        ElementwiseSch<schMode, FastGeluDag::FastGeluNoCast<float>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    }
}
```

【正例】

fast\_gelu函数中删除对workspace参数进行空指针判断，也无需设置SetSysWorkspace和通过GetUserWorkspace来获取Workspace。

```
template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (dType == static_cast<uint64_t>(TPL_FP16)) {
        ElementwiseSch<schMode, FastGeluDag::FastGeluNeedCast<half>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_BF16)) {
        ElementwiseSch<schMode, FastGeluDag::FastGeluNeedCast<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_FP32)) {
        ElementwiseSch<schMode, FastGeluDag::FastGeluNoCast<float>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    }
}
```

