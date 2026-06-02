# InitV2

**页面ID:** atlasascendc_api_07_10221  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_10221.html

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

HCCL客户端初始化接口。该接口默认在所有核上工作，用户也可以在调用前通过GetBlockIdx指定其在某一个核上运行。

#### 函数原型

```
__aicore__ inline void InitV2(GM_ADDR context, const void *initTiling)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| context | 输入 | 通信上下文，包含rankDim，rankID等相关信息。通过框架提供的获取通信上下文的接口GetHcclContext获取context。 |
| initTiling | 输入 | 通信域初始化Mc2InitTiling的地址。Mc2InitTiling在Host侧计算得出，具体请参考表1 Mc2InitTiling参数说明，由框架传递到Kernel函数中使用。 |

#### 约束说明

- 本接口必须与SetCcTilingV2接口配合使用。
- 调用本接口时，必须使用标准C++语法定义TilingData结构体的开发方式，具体请参考使用标准C++语法定义Tiling结构体。
- 调用本接口传入的initTiling参数，不能使用Global Memory地址，建议通过GET_TILING_DATA_WITH_STRUCT接口获取TilingData的栈地址。
- 本接口不支持使用相同的context初始化多个HCCL对象。

#### 调用示例

用户自定义TilingData结构体：

```
class UserCustomTilingData {
    AscendC::tiling::Mc2InitTiling initTiling;
    AscendC::tiling::Mc2CcTiling tiling;
    CustomTiling param;
};
```

     在所有核上创建HCCL对象，并调用InitV2接口初始化：

```
extern "C" __global__ __aicore__ void userKernel(GM_ADDR aGM, GM_ADDR workspaceGM, GM_ADDR tilingGM) {
    REGISTER_TILING_DEFAULT(UserCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(UserCustomTilingData,tilingData,tilingGM);

    GM_ADDR contextGM = AscendC::GetHcclContext<0>(); 
    Hccl hccl;
    hccl.InitV2(contextGM, &tilingData);
    hccl.SetCcTilingV2(offsetof(UserCustomTilingData, tiling));

    // 调用HCCL的Prepare、Commit、Wait、Finalize接口
}
```
