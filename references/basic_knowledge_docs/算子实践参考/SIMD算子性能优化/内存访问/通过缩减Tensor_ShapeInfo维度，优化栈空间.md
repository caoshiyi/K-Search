# 通过缩减Tensor ShapeInfo维度，优化栈空间

**页面ID:** atlas_ascendc_best_practices_10_0019  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0019.html

---

【优先级】中

【描述】GlobalTensor和LocalTensor中通过ShapeInfo类型的成员变量来保存shape信息，SetShapeInfo/GetShapeInfo可以设置或者获取ShapeInfo，在算子实现内部用于shape信息保存和传递。默认情况下支持的最大维度为8。在不使用上述ShapeInfo功能的情况下，不需要这些信息，可以通过K_MAX_SHAPE_DIM宏将其设置为0。经实测减小K_MAX_SHAPE_DIM值，可缩减栈空间，减少scalar指令和cache miss几率，提升算子性能。

```
...
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 8
#endif
...
struct ShapeInfo {
public:
    ...
    uint32_t shape[K_MAX_SHAPE_DIM];
    uint32_t originalShape[K_MAX_SHAPE_DIM];
};

template <typename T> class GlobalTensor {
....
private:
    ShapeInfo shapeInfo_;
}
template <typename T> class LocalTensor {
....
private:
    ShapeInfo shapeInfo_;
}
...
```

【反例】

算子无需使用ShapeInfo，但未对ShapeInfo大小进行限制（使用默认值8），导致浪费K_MAX_SHAPE_DIM * sizeof(uint32_t) * 2 * 4字节的栈空间。2表示有shape和originalShape两个数组，4表示该样例中使用了GlobalTensor和LocalTensor共4个Tensor。

```
...
#include "kernel_operator.h" ...
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    ...
    GlobalTensor<T> dataIn;
    GlobalTensor<T> dataOut;
    LocalTensor<T> vecIn;
    LocalTensor<T> vecOut;
    ...
}
...
```

【正例】

算子无需使用ShapeInfo，设置#define K_MAX_SHAPE_DIM 0，有效缩减了K_MAX_SHAPE_DIM * sizeof(uint32_t) * 2 * 4大小的栈空间。

```
#define K_MAX_SHAPE_DIM 0
...
#include "kernel_operator.h" //需注意定义K_MAX_SHAPE_DIM宏的位置须在包含Ascend C相关头文件之前
...
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    ...
    GlobalTensor<T> dataIn;
    GlobalTensor<T> dataOut;
    LocalTensor<T> vecIn;
    LocalTensor<T> vecOut;
    ...
}
...
```
