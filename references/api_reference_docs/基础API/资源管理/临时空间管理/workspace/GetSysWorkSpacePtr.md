# GetSysWorkSpacePtr

**页面ID:** atlasascendc_api_07_0170  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0170.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | √ |

#### 功能说明

获取系统workspace指针。部分高阶API如Matmul需要使用系统workspace，相关接口需要传入系统workspace指针，此时可以通过该接口获取。使用系统workspace时，host侧开发者需要自行申请系统workspace的空间，其预留空间大小可以通过GetLibApiWorkSpaceSize接口获取。

#### 函数原型

```
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
```

#### 参数说明

无

#### 约束说明

无

#### 返回值说明

系统workspace指针。

#### 调用示例

```
...
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling); // 初始化
// CopyIn阶段：完成从GM到LocalMemory的搬运
mm.SetTensorA(gm_a);    // 设置左矩阵A
mm.SetTensorB(gm_b);    // 设置右矩阵B
mm.SetBias(gm_bias);    // 设置Bias
// Compute阶段：完成矩阵乘计算
while (mm.Iterate()) { 
    // CopyOut阶段：完成从LocalMemory到GM的搬运
    mm.GetTensorC(gm_c); 
}
// 结束矩阵乘操作
mm.End();
```
