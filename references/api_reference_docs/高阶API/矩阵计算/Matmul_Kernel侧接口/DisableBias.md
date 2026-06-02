# DisableBias

**页面ID:** atlasascendc_api_07_0636  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0636.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

清除Bias标志位，表示Matmul计算时没有Bias参与。如果在初始化时配置了TCubeTiling结构中的isBias参数来使能Bias，调用该接口后，会清除Bias标志位，不再使能Bias。

#### 函数原型

```
__aicore__ inline void DisableBias()
```

#### 参数说明

无

#### 约束说明

无

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.DisableBias();    //清除tiling中的Bias标志位
mm.IterateAll(gm_c);
```
