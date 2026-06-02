# LocalTensor简介

**页面ID:** atlasascendc_api_07_00100  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00100.html

---

LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置TPosition为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

#### 需要包含的头文件

```
#include "kernel_operator.h"
```

#### 原型定义

```
template <typename T> class LocalTensor : public BaseLocalTensor<T> {
public:
    // PrimT用于在T传入为TensorTrait类型时萃取TensorTrait中的LiteType基础数据类型
    using PrimType = PrimT<T>;
    __aicore__ inline LocalTensor<T>() {};
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    ~LocalTensor();
    explicit LocalTensor<T>(TBuffAddr& address);
    LocalTensor<T>(const LocalTensor<T>& other);
    LocalTensor<T> operator = (const LocalTensor<T>& other);
    PrimType* GetPhyAddr(const uint32_t offset) const;
    PrimType* GetPhyAddr() const;
    __inout_pipe__(S) PrimType GetValue(const uint32_t offset) const;
    __inout_pipe__(S) PrimType& operator()(const uint32_t offset) const;
    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
    template <typename T1> __inout_pipe__(S) void SetValue(const uint32_t index, const T1 value) const;
    LocalTensor operator const;
    template <typename T1> void SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset);
    inline void Print();
    inline void Print(uint32_t len);
    int32_t ToFile(const std::string& fileName) const;
#else
    __aicore__ inline uint64_t GetPhyAddr() const;
    __aicore__ inline uint64_t GetPhyAddr(const uint32_t offset) const;
    __aicore__ inline __inout_pipe__(S) PrimType GetValue(const uint32_t index) const;
    __aicore__ inline __inout_pipe__(S) __ubuf__ PrimType& operator()(const uint32_t offset) const;
    template <typename CAST_T> __aicore__ inline LocalTensor<CAST_T> ReinterpretCast() const;
    template <typename T1> __aicore__ inline __inout_pipe__(S)
        void SetValue(const uint32_t index, const T1 value) const;
    __aicore__ inline LocalTensor operator const;
    template <typename T1>
    [[deprecated("NOTICE: SetAddrWithOffset has been deprecated and will be removed in the next version. "
        "Please do not use it!")]]
    __aicore__ inline void SetAddrWithOffset(LocalTensor<T1> &src, uint32_t offset);
#endif
    __aicore__ inline LocalTensor<T>(AscendC::TPosition pos, uint32_t addr, uint32_t tieSize);
    __aicore__ inline int32_t GetPosition() const;
    __aicore__ inline void SetSize(const uint32_t size);
    __aicore__ inline uint32_t GetSize() const;
    [[deprecated("NOTICE: GetLength has been deprecated and will be removed in the next version. Please do not use "
                 "it!")]]
    __aicore__ inline uint32_t GetLength() const;
    __aicore__ inline void SetBufferLen(uint32_t dataLen);
    __aicore__ inline void SetUserTag(const TTagType tag);
    __aicore__ inline TTagType GetUserTag() const;
    ...
    __aicore__ inline void SetShapeInfo(const ShapeInfo& shapeInfo);
    __aicore__ inline ShapeInfo GetShapeInfo() const;
    ...
};
```

#### 模板参数

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| T | 类型T可以支持基础数据类型以及TensorTrait类型，但需要遵循使用此LocalTensor接口的数据类型支持情况。 特别地，针对根据指定的逻辑位置/地址/长度，返回Tensor对象的构造函数，类型T仅支持基础数据类型。 |
