# Scalar读写数据<a name="ZH-CN_TOPIC_0000002554351461"></a>

AI Core中Scalar计算单元负责各类型的标量数据运算和程序的流程控制。根据[硬件架构](硬件实现.md)设计，Scalar仅支持对Global Memory和Unified Buffer的读写操作，而不支持对L1 Buffer、L0A Buffer、L0B Buffer和L0C Buffer等其他类型存储的访问。下文分别介绍了Scalar读写Global Memory和Unified Buffer的方式和Scalar读写数据时的同步机制。

## Scalar读写Global Memory<a name="section7480536235"></a>

<!-- img2text -->
```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                      AI Core                                         │
│      ┌──────────────────────────────────────────────────────────────────────────┐      │
│      │                                  AIC1                                    │      │
│      │   ┌────────────────────────────────────────────────────────────────┐     │      │
│      │   │                              AIC 0                             │     │      │
│      │   │   ┌──────────────────────────────────────────────────────┐     │     │      │
│      │   │   │                        AIV 1                         │     │     │      │
│      │   │   │  ┌──────────────────────────────────────────────┐    │     │     │      │
│      │   │   │  │                    AIV 0                     │    │     │     │      │
│      │   │   │  │                                              │    │     │     │      │
│      │   │   │  │  ┌───────────────┐                           │    │     │     │      │
│      │   │   │  │  │    Scalar     │                           │    │     │     │      │
│      │   │   │  │  └───────────────┘                           │    │     │     │      │
│      │   │   │  │         ↑                                    │    │     │     │      │
│      │   │   │  │         │ Cacheable                          │    │     │     │      │
│      │   │   │  │         ↓                                    │    │     │     │      │
│      │   │   │  │  ┌────────────┬────────────┬────────────┬────┐│    │     │     │      │
│      │   │   │  │  │ Cache Line │ Cache Line │ Cache Line │    ││    │     │     │      │
│      │   │   │  │  │  ( 64B )   │  ( 64B )   │  ( 64B )   │    ││    │     │     │      │
│      │   │   │  │  └────────────┴────────────┴────────────┴────┘│    │     │     │      │
│      │   │   │  │                                  DataCache   │    │     │     │      │
│      │   │   │  └──────────────────────────────────────────────┘    │     │     │      │
│      │   │   └──────────────────────────────────────────────────────┘     │     │      │
│      │   └────────────────────────────────────────────────────────────────┘     │      │
│      └──────────────────────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                           ↑
                           │
                           ↓
┌────────────────┬────────────────┬────────────────┬────────────────┐
│   Cache Line   │   Cache Line   │   Cache Line   │   Cache Line   │  L2 Cache
└────────────────┴────────────────┴────────────────┴────────────────┘
        ⇅                                          ⇅
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                  GLobal Memory                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

如上图所示，Scalar读写GM数据时会经过DataCache，DataCache主要用于提高标量访存指令的执行效率，每一个AIC/AIV核内均有一个独立的DataCache。下面通过一个具体示例来讲解DataCache的具体工作机制。

globalTensor1是位于GM上的Tensor：

-   执行完GetValue\(0\)后，globalTensor1的前8个元素会进入DataCache，后续GetValue\(1\)\~GetValue\(7\)不需要再访问GM，而可以直接从DataCache的Cache Line中读取数据，提高了标量连续访问的效率。
-   执行完SetValue\(8, val\)后，globalTensor1的index为8\~15的元素会进入DataCache，SetValue只会修改DataCache中的Cache Line数据，同时将Cache Line的状态设置为Dirty，表明Cache Line中的数据与GM中的数据不一致。

```
AscendC::GlobalTensor<int64_t> globalTensor1;
globalTensor1.SetGlobalBuffer((__gm__ int64_t *)input);
// 从0~7共计8个uint64_t类型，DataCache的Cache Line长度为64字节
// 执行完GetValue(0)后，GetValue(1)~GetValue(7)可以直接从Cache Line中读取，不需要再访问GM
globalTensor1.GetValue(0);
globalTensor1.GetValue(1);
globalTensor1.GetValue(2);
globalTensor1.GetValue(3);
globalTensor1.GetValue(4);
globalTensor1.GetValue(5);
globalTensor1.GetValue(6);
globalTensor1.GetValue(7);

// 执行完SetValue(8)后，不会修改GM上的数据，只会修改DataCache中Cache Line数据
// 同时Cache Line的状态置为dirty，dirty表示DataCache中Cache Line数据与GM中的数据不一致
int64_t val = 32;
globalTensor1.SetValue(8, val);
globalTensor1.GetValue(8);
```

根据上文的工作机制（如下图所示），多核间访问globalTensor1会出现数据不一致的情况，如果其余核需要获取GM数据的变化，则需要开发者手动调用[DataCacheCleanAndInvalid](DataCacheCleanAndInvalid.md)来保证数据的一致性。

<!-- img2text -->
```text
                         ┌───────────────┐
                         │ globalTensor1 │
                         └───────────────┘
                                 │
                                 ↓
→ ┌────────────┬──────┬──────┬──────┬────┬────┬──────┬────────────┐
  │            │      │      │      │    │    │      │            │
  └────────────┴──────┴──────┴──────┴────┴────┴──────┴────────────┘
                 │      │             │    │
                 ↓      ↓             ↓    ↓
```

## Scalar读写Unified Buffer<a name="section8156161471119"></a>

Scalar读写Unified Buffer时，可以使用LocalTensor的SetValue和GetValue接口。示例如下：

```
for (int32_t i = 0; i < 16; ++i) {
    inputLocal.SetValue(i, i); // 对inputLocal中第i个位置进行赋值为i
}

for (int32_t i = 0; i < srcLen; ++i) {
    auto element = inputLocal.GetValue(i); // 获取inputLocal中第i个位置的数值
}
```

## Scalar读写数据时的同步<a name="section554364118119"></a>

Scalar读写Global MemoryUnified Buffer时属于PIPE\_S（Scalar流水）操作，当用户使用SetValue或者GetValue接口，且算子工程使能自动同步时，不需要手动插入同步事件。

如果用户关闭算子工程的自动同步功能时，则需要手动插入同步事件：

```
// GetValue为Scalar操作，与后续的Duplicate存在数据依赖
// 因此Vector流水需要等待Scalar操作结束
float inputVal = srcLocal.GetValue(0);
SetFlag<HardEvent::S_V>(eventID1);
WaitFlag<HardEvent::S_V>(eventID1);
AscendC::Duplicate(dstLocal, inputVal, srcDataSize); 

// SetValue为Scalar操作，与后续的数据搬运操作存在数据依赖
// 因此MTE3流水需要等待Scalar操作结束
srcLocal.SetValue(0, value);
SetFlag<HardEvent::S_MTE3>(eventID2);
WaitFlag<HardEvent::S_MTE3>(eventID2);
AscendC::DataCopy(dstGlobal, srcLocal, srcDataSize); 
```

