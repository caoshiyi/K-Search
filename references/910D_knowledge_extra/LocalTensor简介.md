# LocalTensor简介<a name="ZH-CN_TOPIC_0000002523344074"></a>

LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置[TPosition](TPosition.md)为VECIN、VECOUT、VECCALC、A1、A2、B1、B2、CO1、CO2。

## 需要包含的头文件<a name="zh-cn_topic_0000002213064918_section78885814919"></a>

```
#include "kernel_operator.h"
```

## 原型定义<a name="section1380144323919"></a>

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
    LocalTensor operator[](const uint32_t offset) const;
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
    __aicore__ inline LocalTensor operator[](const uint32_t offset) const;
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

## 模板参数<a name="section116801320102618"></a>

**表 1**  模板参数说明

<a name="table13588175515344"></a>
<table><thead align="left"><tr id="row1160915519346"><th class="cellrowborder" valign="top" width="21.8%" id="mcps1.2.3.1.1"><p id="p9609105553412"><a name="p9609105553412"></a><a name="p9609105553412"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="78.2%" id="mcps1.2.3.1.2"><p id="p156091955143419"><a name="p156091955143419"></a><a name="p156091955143419"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1545073919457"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p1745103924512"><a name="p1745103924512"></a><a name="p1745103924512"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001441184464_p1742518920145"><a name="zh-cn_topic_0000001441184464_p1742518920145"></a><a name="zh-cn_topic_0000001441184464_p1742518920145"></a>类型T可以支持基础数据类型以及<a href="TensorTrait.md">TensorTrait</a>类型，但需要遵循使用此LocalTensor接口的数据类型支持情况。</p>
<p id="p2740141254519"><a name="p2740141254519"></a><a name="p2740141254519"></a>特别地，针对根据指定的逻辑位置/地址/长度，返回Tensor对象的构造函数，类型T仅支持基础数据类型。</p>
</td>
</tr>
</tbody>
</table>

