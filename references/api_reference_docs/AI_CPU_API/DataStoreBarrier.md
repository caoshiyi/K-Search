# DataStoreBarrier

**页面ID:** atlasascendc_api_07_00188  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00188.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品            AI Core | x |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

数据同步屏障指令，该指令会阻塞当前线程执行，确保所有先前的写内存操作完成并对其它硬件单元可见后，才继续执行后续指令。用于AI CPU与AI Core多核之间的数据一致性。

#### 函数原型

```
DataStoreBarrier(void)
```

#### 参数说明

无

#### 约束说明

- 该接口仅支持通过<<<...>>>调用，并在异构编译场景使用。

#### 调用示例

     在AI CPU算子Kernel侧实现代码中调用AscendC::DataStoreBarrier()，确保AI CPU算子对Tiling数据的修改写入内存，使得AI Core算子能够正确读取Tiling数据：

```
struct TilingInfo {
    uint64_t lock; // AI CPU/AI Core之间同步的锁
    int8_t type;
    int8_t mode;
    int8_t len;
};
struct KernelArgs {
    uint32_t *xDevice;
    uint32_t *yDevice;
    uint32_t *zDevice;
    TilingInfo *ti; // 与AI Core共享的参数，用于同步tiling选择
};

template<typename T, int8_t mode, int8_t len>
__aicore__ void hello_world_impl(GM_ADDR m)
{
    if constexpr (std::is_same_v<T, float>) {
       AscendC::printf("Hello World: float mode %u len %u.\n", mode, len);
    } else if constexpr (std::is_same_v<T, int>) {
       AscendC::printf("Hello World: int mode %u len %u.\n", mode, len);
    }
}

// AI Core算子总入口
// tilingInfo: 和AI CPU算子共同传递的参数，用于数据共享
template<typename T, int8_t mode, int8_t len>
__mix__(1,2) __global__ __aicore__ void hello_world(GM_ADDR m, GM_ADDR TilingPtr)
{
     __gm__ struct KernelInfo::TilingInfo *ti = (__gm__ struct KernelInfo::TilingInfo *)TilingPtr;
    AscendC::GlobalTensor<uint64_t> lock;
    lock.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(&ti->lock));
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            while (*reinterpret_cast<volatile __gm__ uint64_t*>(lock.GetPhyAddr(0)) == 0) {   // 下沉模式，AI Core等待AICPU tiling计算完成
                AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                    AscendC::DcciDst::CACHELINE_OUT>(lock);    //直接访问Global Memory，获取最新数据
            }
        }
    }
    // 上面是1个核等待AI CPU tiling计算完成，这里进行核间同步
    AscendC::SyncAll<false>();
    // 根据tiling参数值选择不同模板
    if (ti->type ==0 && ti->mode == 1 && ti->len == 2) {
        hello_world_impl<float, 1, 2>(m);
    } else if (ti->type == 1 && ti->mode == 2 && ti->len == 4) {
        hello_world_impl<int, 2, 4>(m);
    }
    // 执行完留一个核释放lock
    if ASCEND_IS_AIV {
        if (AscendC::GetBlockIdx() == 0) {
            lock.SetValue(0, 0);  // 刷新 lock
            AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                AscendC::DcciDst::CACHELINE_OUT>(lock);    //刷新Dcache，同步与GM之间的数据
        }
    }
}

extern "C" __global__ __aicpu__ uint32_t MyAicpuKernel(void *arg)
{
    KernelArgs* cfg = (KernelArgs*)arg;
    AscendC::printf("MyAicpuKernel inited!\n");
    cfg->ti->lock = 1;
    cfg->ti->type = 1;
    cfg->ti->mode = 2;
    cfg->ti->len = 4;
    AscendC::DataStoreBarrier(); // 对tilingInfo进行写同步
    AscendC::printf("MyAicpuKernel inited type %u mode %u len %u end!\n", cfg->ti->type, cfg->ti->mode, cfg->ti->len);
    return 0;
}
```
