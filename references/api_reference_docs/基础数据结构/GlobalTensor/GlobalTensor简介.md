# GlobalTensor简介

**页面ID:** atlasascendc_api_07_00022  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00022.html

---

GlobalTensor用来存放Global Memory（外部存储）的全局数据。

GlobalTensor public成员函数如下。类型T支持基础数据类型以及TensorTrait类型，但需要遵循使用此GlobalTensor的指令的数据类型支持情况。

```
template <typename T> class GlobalTensor : public BaseGlobalTensor<T> {
public:
    // PrimT用于在T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型
    using PrimType = PrimT<T>;
    // 构造函数
    __aicore__ inline GlobalTensor<T>() {}
    // 初始化GlobalTensor
    __aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer, uint64_t bufferSize); 
    __aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer);
    // 获取全局数据的地址 
    __aicore__ inline const __gm__ PrimType* GetPhyAddr() const;
    __aicore__ inline __gm__ PrimType* GetPhyAddr(const uint64_t offset) const;
    // 获取GlobalTensor的相应偏移位置的值
    __aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint64_t offset) const; 
    // 获取某个索引位置的元素的引用
    __aicore__ inline __inout_pipe__(S) __gm__ PrimType& operator()(const uint64_t offset) const;
    // 设置GlobalTensor相应偏移位置的值
    __aicore__ inline void SetValue(const uint64_t offset, PrimType value);
    // 获取GlobalTensor的元素个数
    __aicore__ inline uint64_t GetSize() const;
    // 返回指定偏移量的GlobalTensor
    __aicore__ inline GlobalTensor operator const; 
    // 设置GlobalTensor的shape信息
    __aicore__ inline void SetShapeInfo(const ShapeInfo& shapeInfo);
    // 获取GlobalTensor的shape信息
    __aicore__ inline ShapeInfo GetShapeInfo() const;
    // 设置GlobalTensor写入L2 Cache的模式
    template<CacheRwMode rwMode = CacheRwMode::RW>
    __aicore__ inline void SetL2CacheHint(CacheMode mode);

    
    ...
};
```
