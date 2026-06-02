# REGISTER\_NONE\_TILING<a name="ZH-CN_TOPIC_0000002523304312"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000001526206862_section212607105720"></a>

在Kernel侧使用标准C++语法自定义的TilingData结构体时，若用户不确定需要注册哪些结构体，可使用该接口告知框架侧需使用未注册的标准C++语法来定义TilingData，并配套[GET\_TILING\_DATA\_WITH\_STRUCT](GET_TILING_DATA_WITH_STRUCT.md)，[GET\_TILING\_DATA\_MEMBER](GET_TILING_DATA_MEMBER.md)，[GET\_TILING\_DATA\_PTR\_WITH\_STRUCT](GET_TILING_DATA_PTR_WITH_STRUCT.md)来获取对应的TilingData。

## 函数原型<a name="zh-cn_topic_0000001526206862_section1630753514297"></a>

```
REGISTER_NONE_TILING
```

## 参数说明<a name="zh-cn_topic_0000001526206862_section129451113125413"></a>

无

## 约束说明<a name="zh-cn_topic_0000001526206862_section65498832"></a>

-   暂不支持Kernel直调工程。
-   使用[GET\_TILING\_DATA](GET_TILING_DATA.md)需提供默认注册的TilingData结构体，但本接口不注册TilingData结构体，故不支持与[5.11.1-GET\_TILING\_DATA](GET_TILING_DATA.md)组合使用。
-   不支持和[REGISTER\_TILING\_DEFAULT](REGISTER_TILING_DEFAULT.md)或[REGISTER\_TILING\_FOR\_TILINGKEY](REGISTER_TILING_FOR_TILINGKEY.md)混用，即不支持注册TilingData结构体的场景与非注册场景混合使用。

## 调用示例<a name="zh-cn_topic_0000001526206862_section97001499599"></a>

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

