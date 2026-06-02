# SetSelfDefineData

**页面ID:** atlasascendc_api_07_0665  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0665.html

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

使能模板参数MatmulCallBackFunc（自定义回调函数）时，设置需要的计算数据或在GM上存储的数据地址等信息，用于回调函数使用。复用同一个Matmul对象时，可以多次调用本接口重新设置对应数据信息。

#### 函数原型

```
__aicore__ inline void SetSelfDefineData(const uint64_t dataPtr)
```

```
__aicore__ inline void SetSelfDefineData(T dataPtr)
```

Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持SetSelfDefineData(T dataPtr)接口原型。

Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持SetSelfDefineData(T dataPtr)接口原型。

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dataPtr | 输入 | 设置的算子回调函数需要的计算数据或在GM上存储的数据地址等信息。其中，类型T支持用户自定义基础结构体。 |

#### 约束说明

- 若回调函数中需要使用dataPtr参数时，必须调用此接口；若回调函数不使用dataPtr参数，无需调用此接口。
- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 本接口必须在SetTensorA接口、SetTensorB接口之前调用。

#### 调用示例

```
//用户自定义回调函数
void DataCopyOut(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local, const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyA1(const LocalTensor<int8_t> &aMatrix, const __gm__ void *gm, int row, int col, int useM, int useK, const uint64_t tilingPtr, const uint64_t dataPtr);
void CopyB1(const LocalTensor<int8_t> &bMatrix, const __gm__ void *gm, int row, int col, int useK, int useN, const uint64_t tilingPtr, const uint64_t dataPtr);

typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> aType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> bType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> cType; 
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> biasType; 
AscendC::Matmul<aType, bType, cType, biasType, CFG_NORM, MatmulCallBackFunc<DataCopyOut, CopyA1, CopyB1>> mm;
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
GlobalTensor<SrcT> dataGM; // 保存有回调函数需使用的计算数据的GM
uint64_t dataGMPtr = reinterpret_cast<uint64_t>(dataGM.address_);
mm.SetSelfDefineData(dataGMPtr);
mm.SetTensorA(gmA);
mm.SetTensorB(gmB);
mm.IterateAll();
```
