# Init（废弃）

**页面ID:** atlasascendc_api_07_0871  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0871.html

---

> **注意:** 

该接口废弃，并将在后续版本移除，请不要使用该接口。请使用InitV2接口进行初始化。

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

HCCL客户端初始化接口。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。

#### 函数原型

```
__aicore__ inline void Init(GM_ADDR context, __gm__ void *initTiling = nullptr)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| context | 输入 | 通信上下文，包含rankDim，rankID等相关信息。 |
| initTiling | 输入 | 可选参数，通信域初始化Mc2InitTiling的地址。Mc2InitTiling在Host侧计算得出，具体请参考表1 Mc2InitTiling参数说明，由框架传递到Kernel函数中使用，完整示例请参考8.13.1.2-调用示例。 |

#### 约束说明

- 若调用本接口时传入initTiling参数，则必须与SetCcTiling接口配合使用。
- 同一个程序中不能同时调用传入可选参数initTiling的接口和不传入可选参数initTiling的接口，推荐使用传入initTiling参数的调用方式。
- 若调用本接口时传入initTiling参数，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考使用标准C++语法定义Tiling结构体。
- 本接口不支持使用相同的context初始化多个HCCL对象。每个HCCL对象都应获取其自身的通信上下文。
