# 纯搬运类算子VECIN和VECOUT建议复用

**页面ID:** atlas_ascendc_best_practices_10_0027  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0027.html

---

【优先级】高

【描述】纯搬运类算子在执行时并不涉及实际vector计算，若存在冗余的vector指令，会导致算子整体执行时间变长。这种场景可以使用Ascend C针对纯搬运类算子提供的TQueBind接口，该接口可以将VECIN与VECOUT绑定，省略将数据从VECIN拷贝到VECOUT的步骤，从而避免vector的无谓消耗。

【反例】

此段代码为了保证数据搬入和数据搬出之间的流水同步，存在LocalTensor -> LocalTensor的DataCopy指令。

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

将LocalTensor -> LocalTensor的DataCopy指令替换为TQueBind接口，减少将VECIN拷贝到VECOUT的步骤，从而避免了冗余拷贝。

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

**图1 **aiv_vec_time优化前后对比

<!-- img2text -->
```
                          aiv_vec_time优化前后对比

      ─────────────────────────────────────────────────────────────
      ─────────────────────────────────────────────────────────────

      ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲  ▲
      ─────────────────────────────────────────────────────────────

      ─────────────────────────────────────────────────────────────

      ─────────────────────────────────────────────────────────────

      ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■

                    ▲                                   ■
```

说明:
- 图中为两条近似水平的性能对比曲线。
- 上方曲线使用三角形标记，整体位置较高，表示优化前。
- 下方曲线使用方形标记，整体位置较低，接近 0，表示优化后。
- 结合上下文含义：将 DataCopy 替换为 TQueBind 后，由于省略了数据从 VECIN 拷贝到 VECOUT 的步骤，aiv_vec_time 明显下降，几乎缩减为 0。

如上图所示，将反例中DataCopy指令替换为TQueBind之后有明显优化。由于省略了数据从VECIN拷贝到VECOUT的步骤，aiv_vec_time几乎缩减为0。
