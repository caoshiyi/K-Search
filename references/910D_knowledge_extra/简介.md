# 简介<a name="ZH-CN_TOPIC_0000002523343912"></a>

Mutex用于核内异步流水指令之间的同步处理，其功能类似于传统CPU中的锁机制。通过锁定指定流水再释放流水来完成流水间的同步依赖。每个锁有固定的一个MutexID，该ID可通过用户自定义（范围为0-27）或者通过[AllocMutexID/ReleaseMutexID](AllocMutexID-(ISASI).md)进行申请释放。当用户进行自定义配置时，接口性能达到最优，但此时无法与[TQue](TQue.md)等接口配合使用，若通过[AllocMutexID/ReleaseMutexID](AllocMutexID-(ISASI).md)申请释放，则可与TQue等框架接口混合使用。

-   Lock：根据MutexID获取Mutex，并阻塞当前流水指令队列，直到对应的MutexID被Unlock。
-   Unlock：直到前置当前流水指令退出后，根据MutexID释放对应Mutex。

## 原型定义<a name="section10580930144614"></a>

```
using MutexID = uint8_t;
class Mutex {
template <pipe_t pipe>
static __aicore__ inline void Lock(MutexID id);
template <pipe_t pipe>
static __aicore__ inline void Unlock(MutexID id);
};
```

