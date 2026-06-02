# TPipe构造函数

**页面ID:** atlasascendc_api_07_0109  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0109.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

构造用来管理内存和同步的TPipe对象。

#### 函数原型

```
__aicore__ inline TPipe()
```

#### 约束说明

- 避免TPipe在对象内创建和初始化，TPipe在对象内创建时，可能会影响编译器对对象内常量的优化，引起scalar性能劣化，具体原理请参考避免TPipe在对象内创建和初始化。
- TPipe对象同一时刻全局只能存在一份，同时定义多个TPipe对象，会出现卡死等随机行为。如果需要使用多个TPipe时，请先调用Destroy接口释放前一个TPipe。

#### 调用示例

```
template <typename ComputeT> class KernelExample {
public:
    __aicore__ inline KernelExample() {}
    __aicore__ inline void Init(..., TPipe* pipeIn)
    {
        ...
        pipe = pipeIn;
        pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
        ...
    }
private:
    ...
    TPipe* pipe;
    ...
};
extern "C" __global__ __aicore__ void example_kernel(...) {
    ...
    TPipe pipe;
    KernelExample<float> op;
    op.Init(..., &pipe);
    ...
}
```
