# 避免TPipe在对象内创建和初始化<a name="ZH-CN_TOPIC_0000002523129076"></a>

【优先级】中

【编译器背景知识】创建类对象时，会分配内存空间，用于存储类中的相关成员变量或函数。当类中变量需要参与计算时，变量值从内存被加载到寄存器，计算完成后，变量从寄存器存储回内存。Scalar常量折叠和常量传播是编译器编译时的优化方式，优化前编译器会判断变量是否只初始化过一次或只赋值过一次，若满足此编译优化的前提条件，变量值将会尽量驻留在寄存器中，从而在后续使用变量时，将减少读取内存的操作，提升运行性能。

【描述】TPipe是用来管理全局内存和同步的框架，用户可以调用TPipe的接口，为TQue/TBuf进行内存分配。在编写Ascend C算子过程中，经常用一个类存放计算所需的相关变量，这里称类名为KernelExample。当TPipe对象在KernelExample类的实现中定义并初始化时，TPipe对象的内存空间在整个KernelExample对象的内存空间之中；需要注意的是，创建TPipe对象时，对象初始化会设置全局变量的TPipe指针，这导致KernelExample对象的内存有被外部污染的风险，此时编译器的编译优化将采取保守策略，不会对KernelExample对象中的Scalar变量进行常量折叠和常量传播。因此，在任何场景下，我们都建议将TPipe对象创建于KernelExample类外部，使得TPipe对象的内存空间独立于KernelExample类对象的内存空间，触发编译器对KernelExample类内Scalar的编译优化，减少算子Scalar指令耗时。

【反例】

代码中TPipe对象由KernelExample类内部创建并初始化，影响编译器Scalar折叠优化，在NPU侧导致Scalar不必要的增加。

```
template <typename ComputeT> class KernelExample {
 public:
     __aicore__ inline KernelExample() {}

     __aicore__ inline void Init(...)
     {
         ...
         pipe.InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
         ...
     }

 private:
     ...
     TPipe pipe;
     ...
 };

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
     ...
     KernelExample<float> op;
     op.Init(...);
     ...
 }
```

【正例】

改为由Kernel入口函数创建TPipe对象，在KernelExample类中保存TPipe指针使用。

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

 extern "C" __global__ __aicore__ void example_kernel(...)
 {
     ...
     TPipe pipe;
     KernelExample<float> op;
     op.Init(..., &pipe);
     ...
 }
```

【性能对比】

**图 1**  aiv\_scalar\_time优化前后对比<a name="fig31681942161513"></a>  
<!-- img2text -->
```
                 us
300 ─────────────────────────────────────────────────────────────
    │
280 ─┤  ▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲─▲
    │
260 ─┤
    │
240 ─┤  ■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■─■
    │
220 ─┤
    │
200 ─┴──────────────────────────────────────────────────────────→ 次
      1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20

      ▲ before_aiv_scalar_time
      ■ after_aiv_scalar_time
```

说明:
- 图标题：图 1  aiv_scalar_time优化前后对比
- 纵轴单位：us
- 横轴：1～20次
- before_aiv_scalar_time 整体约在 280～283 us 之间波动
- after_aiv_scalar_time 整体约在 236～237 us 之间波动

**图 2**  aiv\_scalar\_ratio优化前后对比<a name="fig105241506161"></a>  
<!-- img2text -->
```
                          %
40 ─────────────────────────────────────────────────────────────────────────
32 ─────────────────────────────────────────────────────────────────────────
24 ─────────────────────────────────────────────────────────────────────────
               ▲                 ▲            ▲                 ▲        ▲
           ▲  ─ ─ ▲   ▲     ▲     ▲  ▲  ▲     ─   ▲   ▲   ▲  ─   ▲     ─ ▲
20 ──■──▲────■───■───■──▲───▲───▲───▲──▲──■───▲───■──▲──■──▲───■───■──▲───
        ■  ■      ■   ■  ■      ■      ■  ■  ■  ■      ■  ■      ■  ■  ■
16 ───────■──────────■──────■──────■──────■────────■──────■──────■──────■──
 8 ─────────────────────────────────────────────────────────────────────────
 0 ─────────────────────────────────────────────────────────────────────────
    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20             次

                           ▲ before_aiv_scalar_ratio
                           ■ after_aiv_scalar_ratio
```

通过性能数据对比可以看出，Scalar优化效果显著，平均时间从281us减少到236us，下降17%；平均scalar\_time时延占比从21%下降到17%。因此在Scalar bound（达到上限）的场景下可以使用此优化措施。

