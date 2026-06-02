# SetL2CacheHint

**页面ID:** atlasascendc_api_07_00033  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00033.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

设置GlobalTensor是否使能L2 Cache，默认使能L2 Cache。

#### 函数原型

```
template<CacheRwMode rwMode = CacheRwMode::RW>
__aicore__ inline void SetL2CacheHint(CacheMode mode);
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| 设置L2 Cache读写模式。 ``` enum CacheRwMode { READ = 1, WRITE = 2, RW = 3 }; ```  预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。 |  |

**表2 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 用户指定的L2 Cache模式。 ``` enum class CacheMode : uint8_t { CACHE_MODE_DISABLE = 0, // 不使能L2 Cache CACHE_MODE_NORMAL = 1,  // 使能L2 Cache }; ```  如果用户在写算子时，相比不使能L2 Cache，某GlobalTensor使能L2 Cache反而会导致实测性能下降，可以手动禁止该GlobalTensor使能L2 Cache。比如某算子仅会读一次某个GlobalTensor数据，数据进L2 Cache并不会对算子产生收益，反而会因为数据频繁的搬入L2 Cache造成性能损耗，可以考虑不使能该GlobalTensor L2 Cache能力。 如果不调用该接口，默认为CacheMode::CACHE_MODE_NORMAL，即GlobalTensor会使能L2 Cache。 |  |  |

#### 返回值说明

无。

#### 约束说明

该接口功能当前仅支持在自定义算子工程中使用，不支持Kernel直调工程。

#### 调用示例

```
void Init(__gm__ uint8_t *src_gm, __gm__ uint8_t *dst_gm)
{
    uint64_t dataSize = 256; //设置input_global的大小为256

    AscendC::GlobalTensor<int32_t> inputGlobal; // 类型为int32_t
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(src_gm), dataSize); // 设置源操作数在Global Memory上的起始地址为src_gm，所占外部存储的大小为256个int32_t
    inputGlobal.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE); // 设置GlobalTensor不会写入L2 Cache

    AscendC::LocalTensor<int32_t> inputLocal = inQueueX.AllocTensor<int32_t>();    
    AscendC::DataCopy(inputLocal, inputGlobal, dataSize); // 将Global Memory上的inputGlobal拷贝到Local Memory的inputLocal上
    ...
}
```
