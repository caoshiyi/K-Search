# Iterate

**页面ID:** atlasascendc_api_07_0638  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0638.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | x |
| Atlas 训练系列产品 | x |

#### 功能说明

每调用一次Iterate，会计算出一块baseM * baseN的C矩阵。接口内部会维护迭代进度，调用一次后会对A、B矩阵首地址进行偏移。默认以先M轴再N轴的迭代顺序，也可以通过调整tiling参数iterateOrder，转换为先N轴再M轴的迭代顺序。当传入数据未对齐，存在尾块时，会在最后一次迭代输出尾块的计算结果。

一次Iterate矩阵乘的结果C矩阵存放在逻辑位置CO1的内存中，对于CO1内存中计算结果的获取，当前支持如下两种方式：

- 用户无需自行管理存放矩阵乘结果的CO1内存的申请和释放，由Matmul API内部实现管理。调用接口内部管理CO1的Iterate函数原型后，调用GetTensorC接口完成CO1上计算结果的搬出。
- 用户可以灵活自主地控制矩阵乘计算结果的搬运，例如将多次Iterate计算的矩阵乘结果缓存在CO1内存中，在需要搬出该结果时，一次性搬出多块baseM * baseN的C矩阵。这种灵活搬运场景下，用户需要提前申请CO1的内存，调用用户自主管理CO1的Iterate函数原型后，一次Iterate的计算结果会输出到用户申请的CO1内存上。在需要搬出计算结果时，调用Fixpipe接口搬运CO1上的结果，完成后释放申请的CO1内存。具体示例请参考用户自主管理CO1的矩阵乘场景。

#### 函数原型

- 接口内部管理CO1

```
template <bool sync = true> __aicore__ inline bool Iterate(bool enPartialSum = false)
```

- 用户自主管理CO1

```
template <bool sync = true, typename T> __aicore__ inline bool Iterate(bool enPartialSum, const LocalTensor<T>& localCmatrix)
```

  - 
          Atlas 推理系列产品
         AI Core暂不支持用户自主管理CO1。
  - 
          Atlas 200I/500 A2 推理产品
         暂不支持用户自主管理CO1。

#### 参数说明

**表1 **模板参数说明

| 参数名 | 描述 |
| --- | --- |
| sync | 迭代获取C矩阵分片的过程分为同步和异步两种模式。通过该参数设置同步或者异步模式：同步模式设置为true；异步模式设置为false。默认为同步模式。具体模式的介绍和使用方法请参考GetTensorC。 |
| T | 用户申请的CO1内存上LocalTensor的数据类型，即矩阵乘输出的C矩阵的数据类型。当前支持的数据类型为float、int32_t。 |

**表2 **接口内部管理CO1的函数参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enPartialSum | 输入 | 是否将矩阵乘的结果累加于现有的CO1数据，默认值为false。在L0C累加时，只支持C矩阵规格为singleCoreM==baseM && singleCoreN==baseN。          针对             Atlas 200I/500 A2 推理产品            ，该参数仅支持配置为false。 |

**表3 **用户自主管理CO1的函数参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enPartialSum | 输入 | 是否将矩阵乘的结果累加于现有的CO1数据。在L0C累加时，只支持C矩阵规格为singleCoreM==baseM && singleCoreN==baseN。 |
| localCmatrix | 输出 | 由用户申请的CO1上的LocalTensor内存，用于存放矩阵乘的计算结果。 |

#### 返回值说明

false：单核上的所有数据全部算完。

true：数据仍在迭代计算中。

#### 约束说明

- 当使能MixDualMaster（双主模式）场景时，即模板参数enableMixDualMaster设置为true，不支持使用该接口。
- 对于用户自主管理CO1的Iterate函数，创建Matmul对象时，必须定义C矩阵的内存逻辑位置为TPosition::CO1、数据排布格式为CubeFormat::NZ、数据类型为float或int32_t。

#### 调用示例

同步模式及异步模式的简单调用示例如下，更多完整算子样例请参考[异步场景样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/13_matmulleakyrelu_kernellaunch/MatmulLeakyReluInvocationAsync)、[Iterate异步场景矩阵乘法](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_async_iterate)、[自主管理CO1的算子样例](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples/matrix/matmul_l0c_extend)。

```
// 同步模式样例
while (mm.Iterate()) {   
    mm.GetTensorC(ubCmatrix); 
}

// 异步模式样例
mm.template Iterate<false>();
// …… ……其它计算
for (int i = 0; i < singleM/baseM*singleN/baseN; ++i) {   
    mm.template GetTensorC<false>(ubCmatrix); 
}
```
