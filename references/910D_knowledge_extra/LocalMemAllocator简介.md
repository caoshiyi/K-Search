# LocalMemAllocator简介<a name="ZH-CN_TOPIC_0000002523344624"></a>

LocalMemAllocator是在使用[静态Tensor编程方式](静态Tensor编程.md)时用于内存管理的类，用户无需构建TPipe/TQue，而是直接创建LocalTensor对象（也可以直接通过[LocalTensor构造函数](LocalTensor构造函数.md#li1192912551322)进行构造）并开发算子，从而减少运行时的开销，实现更优的性能。

LocalMemAllocator仅支持在Ascend C静态Tensor编程方式中使用，不可以与TPipe等接口混用。

## 需要包含的头文件<a name="zh-cn_topic_0000002213064918_section78885814919"></a>

```
#include "kernel_operator.h"
```

## 原型定义<a name="section10580930144614"></a>

```
template<Hardware hard = Hardware::UB>
class LocalMemAllocator {
public:
    __aicore__ inline LocalMemAllocator();
    __aicore__ inline uint32_t GetCurAddr() const;
    template <class DataType, uint32_t tileSize> LocalTensor<DataType> __aicore__ inline Alloc();
    template <TPosition pos, class DataType, uint32_t tileSize> __aicore__ inline LocalTensor<DataType> Alloc();
    template <class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize);    
    template <TPosition pos, class DataType> LocalTensor<DataType> __aicore__ inline Alloc(uint32_t tileSize);
    
    template <class DataType> LocalTensor<DataType> __aicore__ inline Alloc();
    
    
   
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
<tbody><tr id="row260915573419"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p2060925573411"><a name="p2060925573411"></a><a name="p2060925573411"></a>hard</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p823866165711"><a name="p823866165711"></a><a name="p823866165711"></a><span id="ph46543142425"><a name="ph46543142425"></a><a name="ph46543142425"></a>用于表示数据的物理位置，Hardware枚举类型，定义如下，合法位置为：UB、L1、L0A、L0B、L0C、BIAS、FIXBUF。</span></p>
<a name="screen79541519214"></a><a name="screen79541519214"></a><pre class="screen" codetype="Cpp" id="screen79541519214">enum class Hardware : uint8_t { 
GM,     // Global Memory
UB,     // Unified Buffer
L1,     // L1 Buffer
L0A,    // L0A Buffer
L0B,    // L0B Buffer
L0C,    // L0C Buffer
BIAS,   // BiasTable Buffer
FIXBUF, // Fixpipe Buffer
MAX };</pre>
</td>
</tr>
</tbody>
</table>

