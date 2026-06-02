# 纯搬运类算子VECIN和VECOUT建议复用<a name="ZH-CN_TOPIC_0000002554328995"></a>

【优先级】高

【描述】纯搬运类算子在执行时并不涉及实际vector计算，若存在冗余的vector指令，会导致算子整体执行时间变长。这种场景可以使用Ascend C针对纯搬运类算子提供的TQueBind接口，该接口可以将VECIN与VECOUT绑定，省略将数据从VECIN拷贝到VECOUT的步骤，从而避免vector的无谓消耗。

【反例】

此段代码为了保证数据搬入和数据搬出之间的流水同步，存在LocalTensor -\> LocalTensor的DataCopy指令。

```
template <typename ComputeT> class KernelExample {
 public:
     ...
     __aicore__ inline void Process(...)
     {
         for (int i = 0; i < iLen; ++i) {
             ... 
             auto iLocal = QueI.AllocTensor<ComputeT>();
             DataCopy(iLocal, inGm[i * 32], size);
             QueI.EnQue(iLocal);
             iLocal = QueI.DeQue<ComputeT>();
             for (int j = 0; j < jLen; ++j) { 
                 ...
                 auto oLocal = QueO.AllocTensor<ComputeT>();
                 DataCopy(oLocal, iLocal, size); // LocalTensor -> LocalTensor的DataCopy指令,以实现数据从VECIN到VECOUT的搬运
                 QueO.EnQue(oLocal);

                 auto oLocal = QueO.DeQue<ComputeT>();
                 DataCopyPad(outGm[j], oLocal, ...);
                 QueO.FreeTensor(oLocal);
             }
             QueI.FreeTensor(iLocal);
         }
     }

 private:
     ... 
     TQue<TPosition::VECIN, BUFFER_NUM> QueI;
     TQue<TPosition::VECOUT, BUFFER_NUM> QueO;
     ...
 };

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
     ...
     op.Process(...);
 }
```

【正例】

将LocalTensor -\> LocalTensor的DataCopy指令替换为TQueBind接口，减少将VECIN拷贝到VECOUT的步骤，从而避免了冗余拷贝。

```
template <typename ComputeT> class KernelExample {
 public:
     ...
     __aicore__ inline void Process(...)
     {
         for (int i = 0; i < iLen; ++i) {
             ... 
             auto bindLocal = queBind.AllocTensor<ComputeT>();
             DataCopy(bindLocal, inGm[i * 32], size);
             queBind.EnQue(bindLocal);
             bindLocal = queBind.DeQue<ComputeT>();
             for (int j = 0; j < jlen; ++j) {
                 ...
                 DataCopyPad(outGm[j], bindLocal, ...);
             }
             queBind.FreeTensor(bindLocal);
         }
     }

 private:
     ... 
     TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind; // 使用TQueBind替换原来的QueI，QueO
     ...
 };

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
     ...
     op.Process(...);
 }
```

【性能对比】

**图 1**  aiv\_vec\_time优化前后对比<a name="fig74881227195511"></a>  

<!-- img2text -->
```text
us
30 ┤
   │
24 ┤
   │
18 ┤  ▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲──▲  before_aiv_vec_time
   │
12 ┤
   │
 6 ┤
   │
 0 ┼──■──■──■──■──■──■──■──■──■──■──■──■──■──■──■──■──■──■──■──■  after_aiv_vec_time
    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20  次

图 1  aiv_vec_time优化前后对比

图例:
▲ before_aiv_vec_time
■ after_aiv_vec_time
```

如上图所示，将反例中DataCopy指令替换为TQueBind之后有明显优化。由于省略了数据从VECIN拷贝到VECOUT的步骤，aiv\_vec\_time几乎缩减为0。

