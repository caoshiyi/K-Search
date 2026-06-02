# TBuf简介<a name="ZH-CN_TOPIC_0000002554424485"></a>

使用Ascend C编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的[TPosition](TPosition.md)逻辑位置。

TBuf继承自TQueBind父类，继承关系如下：

<!-- img2text -->
```text
           父类
            │
            ↑
┌──────────────┐
│   TQueBind   │
└──────────────┘
            │
            ↓
┌──────────────┐
│     TBuf     │
└──────────────┘
           子类
```

TBuf占用的存储空间通过TPipe进行管理，您可以通过[InitBuffer](InitBuffer.md)接口为TBuf进行内存初始化操作，之后即可通过[Get](Get.md)获取指定长度的Tensor参与计算。

使用[InitBuffer](InitBuffer.md)为TBuf分配内存和为队列分配内存有以下差异：

-   为TBuf分配的内存空间只能参与计算，无法执行队列的入队出队操作。
-   调用一次内存初始化接口，TPipe只会为TBuf分配一块内存，为队列可以通过参数设置申请多块内存。如果要使用多个临时变量，需要定义多个TBuf数据结构，对每个TBuf数据结构分别调用[InitBuffer](InitBuffer.md)接口进行内存初始化。
-   TBuf获取的Tensor无需释放。

