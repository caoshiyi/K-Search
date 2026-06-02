# SetCcTiling（废弃）

**页面ID:** atlasascendc_api_07_10047  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10047.html

---

> **注意:** 

该接口废弃，并将在后续版本移除，请不要使用该接口。请使用SetCcTilingV2接口设置通信算法的Tiling地址。

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

用于设置HCCL客户端通信算法的Tiling地址。

#### 函数原型

```
__aicore__ inline int32_t SetCcTiling(__gm__ void *ccOpTilingData)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ccOpTilingData | 输入 | 通信算法的Mc2CcTiling参数的地址。Mc2CcTiling在Host侧计算得出，具体请参考表2 Mc2CcTiling参数说明，由框架传递到Kernel函数中使用，完整示例请参考8.13.1.2-调用示例。 |

#### 返回值说明

- HCCL_SUCCESS，表示成功。
- HCCL_FAILED，表示失败。

#### 约束说明

- 参数相同的同一种通信算法在调用Prepare接口前只需要调用一次本接口，否则需要多次调用本接口。
- 同一种通信算法只支持设置一个ccOpTilingData地址；对于同一种通信算法，重复调用本接口会覆盖该通信算法的ccOpTilingData地址。
- 若调用本接口，必须与传initTiling地址的Init接口配合使用，且Init接口在本接口前被调用。
- 若调用本接口，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考使用标准C++语法定义Tiling结构体。
