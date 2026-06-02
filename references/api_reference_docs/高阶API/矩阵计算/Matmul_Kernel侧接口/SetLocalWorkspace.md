# SetLocalWorkspace

**页面ID:** atlasascendc_api_07_0653  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0653.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

对于某些场景Matmul内部需要额外占用VECCALC空间，如果用户希望在算子中复用这个额外占用的VECCALC空间，则该空间需要用户预留，并申请好LocalTensor，将其起始物理地址传入给Matmul。具体需要申请的VECCALC临时空间大小由tiling接口MatmulGetTmpBufSize给出，满足以下几个条件之一就需要使用该接口传入UB临时空间：

- C矩阵Position为TPosition::GM；
- C矩阵CubeFormat为CubeFormat::ND；
- A矩阵或者B矩阵CubeFormat为CubeFormat::ND；
- 存在Bias且Bias的Position不是VECCALC。

请在Iterate或者IterateAll之前调用该接口。

获取到的UB临时空间大小以字节为单位。

#### 函数原型

```
__aicore__ inline void SetLocalWorkspace(const LocalTensor<uint8_t>& tmpBuffer)
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tmpBuffer | 输入 | 临时空间，由用户申请并管理，TPosition为VECCALC。 |

#### 约束说明

当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。

#### 调用示例

```
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
mm.SetLocalWorkspace(mmFormatUb);    //设置临时VECCALC空间
mm.SetTensorA(gm_a);
mm.SetTensorB(gm_b);
mm.SetBias(gm_bias);
mm.IterateAll(gm_c);
```
