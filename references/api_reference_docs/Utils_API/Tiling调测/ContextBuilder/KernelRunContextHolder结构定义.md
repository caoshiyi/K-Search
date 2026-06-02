# KernelRunContextHolder结构定义

**页面ID:** atlasascendc_api_07_1010  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_1010.html

---

#### 功能说明

该结构体为ContextBuilder类最终的构造结果，可通过指定的接口获取内部算子信息或获取KernelContext类的对象。

#### 函数原型

```
struct KernelRunContextHolder {
    KernelRunContextHolder();
    ~KernelRunContextHolder();
    template<typename T>
    T *GetContext() const
    {
        return reinterpret_cast<T*>(context);
    }
    gert::ComputeNodeInfo *MutableComputeNodeInfo()
    {
        return reinterpret_cast<gert::ComputeNodeInfo *>(computeNodeExtendHolder.get());
    }
    std::unique_ptr<ValueHolderImpl> valueHolder;
    std::unique_ptr<uint8_t[]> computeNodeExtendHolder;
    KernelRunContext *context {nullptr};
};
```

#### 函数说明

**表1 **函数说明

| 函数名称 | 入参说明 | 含义 |
| --- | --- | --- |
| GetContext | 无 | 获取context成员变量转化为模板T的指针，T可选值为KernelContext以及它的子类如TilingContext |
| MutableComputeNodeInfo | 无 | 返回构造的gert::ComputeNodeInfo类指针 |

**表2 **变量说明

| 变量名称 | 变量含义 |
| --- | --- |
| valueHolder | 保证KernelRunContextHolder内部值不析构的智能指针 |
| computeNodeExtendHolder | 可转化成ComputeNodeInfo类的智能指针 |
| context | 指向KernelRunContext类的指针 |

#### 约束说明

无

#### 调用示例

```
auto holder = context_ascendc::ContextBuilder().Inputs().Outputs().BuildKernelRunContext();
if (holder != nullptr) {
    gert::KernelContext* tilingParseContext = holder->GetContext<gert::KernelContext>();
    gert::ComputeNodeInfo* info = holder->MutableComputeNodeInfo();
}
```
