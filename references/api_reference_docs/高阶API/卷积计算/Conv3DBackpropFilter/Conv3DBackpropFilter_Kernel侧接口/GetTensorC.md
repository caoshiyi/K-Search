# GetTensorC

**页面ID:** atlasascendc_api_07_0902  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0902.html

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

在完成Iterate操作后调用本接口，获取结果矩阵块，完成数据从L0C到GM的搬运。此接口与Iterate接口配合使用，用于在Iterate执行迭代计算后，获取结果矩阵。

#### 函数原型

```
template <bool sync = true>
__aicore__ inline void GetTensorC(const AscendC::GlobalTensor<DstT> &output, uint8_t enAtomic = 1, bool enSequentialWrite = false)
```

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 预留参数，用户无需感知。 |

**表2 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| output | 输入 | 将计算结果搬至Global Memory的GM地址。 |
| enAtomic | 输入 | 预留参数，用户无需感知。 |
| enSequentialWrite | 输入 | 预留参数，用户无需感知。 |

#### 约束说明

GetTensorC接口必须在Iterate后进行调用，完成卷积反向实现，调用顺序如下。

```
while (Iterate()) {   
    GetTensorC(); 
}
```

#### 调用示例

```
while (gradWeight_.Iterate()) {
    gradWeight_.GetTensorC(gradWeightGm_[offsetC_]);
}
```
