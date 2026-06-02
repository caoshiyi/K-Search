# REGISTER_NONE_TILING

**页面ID:** atlasascendc_api_07_00164  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00164.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

在Kernel侧使用标准C++语法自定义的TilingData结构体时，若用户不确定需要注册哪些结构体，可使用该接口告知框架侧需使用未注册的标准C++语法来定义TilingData，并配套GET_TILING_DATA_WITH_STRUCT，GET_TILING_DATA_MEMBER，GET_TILING_DATA_PTR_WITH_STRUCT来获取对应的TilingData。

#### 函数原型

```
REGISTER_NONE_TILING
```

#### 参数说明

无

#### 约束说明

- 暂不支持Kernel直调工程。
- 使用GET_TILING_DATA需提供默认注册的TilingData结构体，但本接口不注册TilingData结构体，故不支持与5.11.1-GET_TILING_DATA组合使用。
- 不支持和REGISTER_TILING_DEFAULT或REGISTER_TILING_FOR_TILINGKEY混用，即不支持注册TilingData结构体的场景与非注册场景混合使用。

#### 调用示例

```
# Tiling模板库提供方，无法预知用户实例化何种TilingData结构体
template <class BrcDag>
struct BroadcastBaseTilingData {
    int32_t scheMode;
    int32_t shapeLen;
    int32_t ubSplitAxis;
    int32_t ubFormer;
    int32_t ubTail;
    int64_t ubOuter;
    int64_t blockFormer;
    int64_t blockTail;
    int64_t dimProductBeforeUbInner;
    int64_t elemNum;
    int64_t blockNum;
    int64_t outputDims[BROADCAST_MAX_DIMS_NUM];
    int64_t outputStrides[BROADCAST_MAX_DIMS_NUM];
    int64_t inputDims[BrcDag::InputSize][2]; // 整块 + 尾块
    int64_t inputBrcDims[BrcDag::CopyBrcSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputVecBrcDims[BrcDag::VecBrcSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputStrides[BrcDag::InputSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputBrcStrides[BrcDag::CopyBrcSize][BROADCAST_MAX_DIMS_NUM];
    int64_t inputVecBrcStrides[BrcDag::VecBrcSize];
    char scalarData[BROADCAST_MAX_SCALAR_BYTES];
};

template <uint64_t schMode, class BrcDag> class BroadcastSch {
public:
    __aicore__ inline explicit BroadcastSch(GM_ADDR& tmpTiling)
        : tiling(tmpTiling)
    {}
    template <class... Args>
    __aicore__ inline void Process(Args... args)
    {
        REGISTER_NONE_TILING; // 告知框架侧使用未注册的TilingData结构体
        if constexpr (schMode == 1) {
            GET_TILING_DATA_WITH_STRUCT(BroadcastBaseTilingData<BrcDag>, tilingData, tiling);
            GET_TILING_DATA_MEMBER(BroadcastBaseTilingData<BrcDag>, blockNum, blockNumVar, tiling);
            TPipe pipe;
            BroadcastNddmaSch<BrcDag, false> sch(&tilingData); // 获取Schedule
            sch.Init(&pipe, args...);
            sch.Process();
        }   else if constexpr (schMode == 202) {
            GET_TILING_DATA_PTR_WITH_STRUCT(BroadcastOneDimTilingDataAdvance, tilingDataPtr, tiling);
            BroadcastOneDimAdvanceSch<BrcDag> sch(tilingDataPtr); // 获取Schedule
            sch.Init(args...);
            sch.Process();
        }
    }
public:
    GM_ADDR tiling;
};
```

```
#用户通过传入schMode, OpDag模板参数来实例化模板库
using namespace AscendC;
template <uint64_t schMode>
__global__ __aicore__ void mul(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {
        // int8
        using OpDag = MulDag::MulInt8Op::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    } else if constexpr (std::is_same<DTYPE_X1, uint8_t>::value) {
        // uint8
        using OpDag = MulDag::MulUint8Op::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(x1, x2, y);
    }
}
```
