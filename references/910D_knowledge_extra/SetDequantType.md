# SetDequantType<a name="ZH-CN_TOPIC_0000002523344810"></a>

## 功能说明<a name="section618mcpsimp"></a>

该接口用于设置量化或反量化的模式。

Matmul反量化场景：在Matmul计算时，左、右矩阵的输入为int8\_t或int4b\_t类型，输出为half类型；或者左、右矩阵的输入为int8\_t类型，输出为int8\_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行反量化操作，将最终结果反量化为对应的half或int8\_t类型。

Matmul量化场景：在Matmul计算时，左、右矩阵的输入为half或bfloat16\_t类型，输出为int8\_t类型。该场景下，输出C矩阵的数据从CO1搬出到Global Memory时，会执行量化操作，将最终结果量化为int8\_t类型。

量化或反量化时有两种模式:一种是同一系数的量化/反量化模式，一种是向量的量化/反量化模式。

-   同一系数的量化或反量化模式：对输出矩阵的所有值采用同一系数进行量化或反量化。
-   向量的量化或反量化模式：提供一个参数向量，对输出矩阵的每一列都采用该向量中对应列的系数进行量化或反量化。

## 函数原型<a name="section620mcpsimp"></a>

```
int32_t SetDequantType(DequantType dequantType)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  参数说明

<a name="table9646134355611"></a>
<table><thead align="left"><tr id="row964714433565"><th class="cellrowborder" valign="top" width="14.99%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.02%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.99%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row106481443135617"><td class="cellrowborder" valign="top" width="14.99%" headers="mcps1.2.4.1.1 "><p id="p1050655718146"><a name="p1050655718146"></a><a name="p1050655718146"></a><strong id="b5266950115812"><a name="b5266950115812"></a><a name="b5266950115812"></a>dequantType</strong></p>
</td>
<td class="cellrowborder" valign="top" width="12.02%" headers="mcps1.2.4.1.2 "><p id="p4506145791414"><a name="p4506145791414"></a><a name="p4506145791414"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.99%" headers="mcps1.2.4.1.3 "><p id="p350610573140"><a name="p350610573140"></a><a name="p350610573140"></a>设置量化或反量化时的模式。DequantType类型，该类型的定义如下方代码所示，其中参数的取值及含义如下：</p>
<a name="ul8835546171814"></a><a name="ul8835546171814"></a><ul id="ul8835546171814"><li>SCALAR：表示同一系数的量化或反量化模式。</li><li>TENSOR：表示向量的量化或反量化模式。</li></ul>
</td>
</tr>
</tbody>
</table>

```
enum class DequantType {
    SCALAR = 0,
    TENSOR = 1,
};
```

## 返回值说明<a name="section640mcpsimp"></a>

-1表示设置失败；0表示设置成功。

## 约束说明<a name="section633mcpsimp"></a>

本接口支持的同一系数的量化/反量化模式、向量的量化/反量化模式分别与Kernel侧接口[SetQuantScalar](SetQuantScalar.md)和[SetQuantVector](SetQuantVector.md)对应，本接口设置的量化/反量化模式必须与Kernel侧使用的接口保持一致。

## 调用示例<a name="section1665082013318"></a>

```
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MatmulApiTiling tiling(ascendcPlatform); 
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8);   
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);   
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);   
tiling.SetShape(M, N, K);   
tiling.SetOrgShape(M, N, K);
tiling.EnableBias(true);
tiling.SetDequantType(DequantType::SCALAR);  // 设置同一系数的量化/反量化模式
// tiling.SetDequantType(DequantType::TENSOR);  // 设置向量的量化/反量化模式
tiling.SetBufferSpace(-1, -1, -1);
optiling::TCubeTiling tilingData;   
int ret = tiling.GetTiling(tilingData);
```

