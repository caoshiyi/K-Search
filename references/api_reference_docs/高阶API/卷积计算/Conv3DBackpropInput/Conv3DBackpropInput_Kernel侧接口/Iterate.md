# Iterate

**页面ID:** atlasascendc_api_07_0925  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0925.html

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

每次调用Iterate时，会计算出一个baseM * baseN的结果矩阵，并将该结果写入L0C Buffer。接口内部会维护迭代进度，每次调用后，矩阵的起始地址将进行偏移。如果传入的数据未对齐存在尾块，则在最后一次迭代中输出尾块的计算结果。本接口需与GetTensorC接口配合使用；在调用本接口后，再调用GetTensorC接口，L0C Buffer中的数据将被写入目标地址。

#### 函数原型

```
template <bool sync = true> 
__aicore__ inline bool Iterate(bool enPartialSum = false)
```

#### 参数说明

**表1 **接口参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enPartialSum | 输入 | 预留参数，用户无需感知。 |

#### 返回值说明

false：设置的SingleShape上的所有数据已经算完。

true：数据仍在迭代计算中。

#### 约束说明

- Iterate接口必须在初始化接口及输入输出配置接口之后进行调用，完成卷积反向计算，调用顺序如下。

```
Init(...);
... // 输入输出配置
while (Iterate()) {   
    GetTensorC(); 
}
End();
```

- 在多轮循环计算的场景中，在单次循环里计算单核SetSingleShape设置的数据大小。每次单核计算完成后，必须将Conv3DBackpropInput对象的ctx.isFirstIter_设置为true，以确保下一轮循环中的单核计算能够正确进行。

#### 调用示例

```
while (gradInput_.Iterate()) {   
    gradInput_.GetTensorC(gradInputGm_[offsetC_]); 
}
// SingleShape计算完成后，需要将ctx.isFirstIter_设置为true，确保下一块SingleShape的正确计算
gradInput_.ctx.isFirstIter_ = true;
```
