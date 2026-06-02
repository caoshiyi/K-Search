# 使能Iterate或IterateAll异步接口避免AIC/AIV同步依赖<a name="ZH-CN_TOPIC_0000002523289084"></a>

【优先级】高

【描述】在MIX场景，即AIC（AI Cube核）和AIV（AI Vector核）混合编程中，调用Matmul Iterate或者IterateAll时，AIV发送消息到AIC启动Matmul计算。若通过Iterate<true\>同步方式，如[图1 同步方式消息发送示意图](#fig99236286201)，每次调用都会触发一次消息发送，而通过Iterate<false\>异步方式，如[图2 异步方式消息发送示意图](#fig1511392207)，仅第一次需要发送消息，后续无需发送消息，从而减少Cube与Vector核间交互，减少核间通信开销。因此，MIX场景推荐使用Iterate<false\>或者IterateAll<false\>异步接口（注意：使用异步接口时需要设置Workspace）。

**图 1**  同步方式消息发送示意图<a name="fig99236286201"></a>  
<!-- img2text -->
```text
第1次调用Iterate()          第2次调用Iterate()          第3次调用Iterate()          第4次调用Iterate()

AIV
 ─────────────                ─────────────                ─────────────                ─────────────
              │                            │                            │                            │
              ↓                            ↓                            ↓                            ↓

AIC
        ─────────────                ─────────────                ─────────────                ─────────────
```

**图 2**  异步方式消息发送示意图<a name="fig1511392207"></a>  
<!-- img2text -->
```
第1次调用
Iterate<false>()

AIV
────────────────────

            ↘
             ↘
AIC           ▼
    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    └────────────────────────────────────────────────────────────┘
      第2次调用               第3次调用               第4次调用
      Iterate<false>()       Iterate<false>()       Iterate<false>()
```

【反例】

MIX场景使用Iterate接口的同步方式。

```
TQueBind<TPosition::CO2, TPosition::VECIN>  qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT>  qVecOut;
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
int16_t scalar = 2;

while(mm.template Iterate()){
    auto cInUB = qVecIn.AllocTensor<float>();
    mm.GetTensorC(cInUB);
    qVecIn.EnQue(cInUB);
    cInUB = qVecIn.DeQue<float>();
    auto cOutUB = qVecOut.AllocTensor<float>();
    Muls(cOutUB, cInUB, scalar, baseM*baseN);
    qVecIn.FreeTensor(cInUB);
    ...
}
```

【正例】

MIX场景使用Iterate接口的异步方式。

```
TQueBind<TPosition::CO2, TPosition::VECIN>  qVecIn;
TQueBind<TPosition::VECIN, TPosition::VECOUT>  qVecOut;
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
mm.SetWorkspace(workspace, size);//其中，workspace为临时空间的物理地址，size为singleCoreM*singleCoreN大小的矩阵C占用的内存大小：singleCoreM*singleCoreN*sizeof(float)
int16_t scalar = 2;

while(mm.template Iterate<false>()){
    auto cInUB = qVecIn.AllocTensor<float>();
    mm.GetTensorC(cInUB);
    qVecIn.EnQue(cInUB);
    cInUB = qVecIn.DeQue<float>();
    auto cOutUB = qVecOut.AllocTensor<float>();
    Muls(cOutUB, cInUB, scalar, baseM*baseN);
    qVecIn.FreeTensor(cInUB);
    ...
}
```

