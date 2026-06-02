# SetCcTilingV2

**页面ID:** atlasascendc_api_07_10222  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10222.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

用于设置HCCL客户端中某个通信算法配置的TilingData地址。

#### 函数原型

```
__aicore__ inline int32_t SetCcTilingV2(uint64_t offset)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| offset | 输入 | 通信算法配置Mc2CcTiling参数地址相对于Mc2InitTiling起始地址的偏移。Mc2CcTiling在Host侧计算得出，具体请参考表2 Mc2CcTiling参数说明，由框架传递到Kernel函数中使用。 |

#### 返回值说明

- HCCL_SUCCESS，表示成功。
- HCCL_FAILED，表示失败。

#### 约束说明

- 若调用本接口，必须保证InitV2在本接口前被调用。
- Tiling参数相同的同一种通信算法在调用Prepare接口前，只需要调用一次本接口，请参考调用示例：类型不同、Tiling参数不同的通信。
- 对于同一种通信算法，如果Tiling参数不同，重复调用本接口会覆盖之前的Tiling参数地址，因此需要在调用Prepare接口后再调用本接口设置新的Tiling参数。请参考调用示例：类型相同、Tiling参数不同的通信。
- 若调用本接口，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考使用标准C++语法定义Tiling结构体。

#### 调用示例

- 用户自定义TilingData结构体：

```
class UserCustomTilingData {
    AscendC::tiling::Mc2InitTiling initTiling;
    AscendC::tiling::Mc2CcTiling allGatherTiling;
    AscendC::tiling::Mc2CcTiling allReduceTiling1;
    AscendC::tiling::Mc2CcTiling allReduceTiling2;
    CustomTiling param;
};
```

- 类型不同、Tiling参数不同的通信

```
extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    REGISTER_TILING_DEFAULT(UserCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();
    hccl.InitV2(contextGM, &tilingData);

    // 在下发任务之前，通过SetCcTilingV2设置对应的tiling
    if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allGatherTiling)) != HCCL_SUCCESS ||
        hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling1)) != HCCL_SUCCESS) {
        return;
    }
    const auto agHandleId = hccl.AllGather<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16);
    hccl.Wait(agHandleId);

    const auto arHandleId = hccl.AllReduce<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
    hccl.Wait(arHandleId);

    hccl.Finalize();
}
```

- 类型相同、Tiling参数不同的通信

```
extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    REGISTER_TILING_DEFAULT(UserCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData, tilingData, tilingGM);

    Hccl hccl;
    GM_ADDR contextGM = AscendC::GetHcclContext<0>();
    hccl.InitV2(contextGM, &tilingData);

    // 在下发通信任务之前，通过SetCcTilingV2设置对应的Tiling参数地址
    if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling1)) != HCCL_SUCCESS) {
        return;
    }
    const auto arHandleId1 = hccl.AllReduce<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
    hccl.Wait(arHandleId1);

    // 第二次AllReduce的Tiling参数与第一次不同，在第一次Prepare之后再调用SetCcTilingV2
    if (hccl.SetCcTilingV2(offsetof(UserCustomTilingData, allReduceTiling2)) != HCCL_SUCCESS) {
        return;
    }
    const auto arHandleId2 = hccl.AllReduce<true>(sendBuf, recvBuf, dataCount, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);
    hccl.Wait(arHandleId2);

    hccl.Finalize();
}
```
