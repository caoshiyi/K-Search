# GetTiling

**页面ID:** atlasascendc_api_07_0933  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0933.html

---

#### 功能说明

获取Tiling参数。

#### 函数原型

```
int64_t GetTiling(optiling::Conv3DBackpropInputTilingData &tiling)
```

```
int64_t GetTiling(AscendC::tiling::Conv3DBackpropInputTilingData &tiling)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tiling | 输出 | Conv3DBackpropInput的Tiling结构体，用于存储最终的Tiling结果。TConv3DBackpropInputTiling结构介绍请参考TConv3DApiTiling结构体说明。 |

#### 返回值说明

如果返回值不为-1，则代表Tiling计算成功，用户可以使用该Tiling结构的值。如果返回值为-1，则代表Tiling计算失败，该Tiling结果无法使用。

#### 约束说明

无

#### 调用示例

```
// 构建Conv3dBackpropInput算子tiling的调用示例
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ConvBackpropApi::Conv3DBpInputTiling tiling(ascendcPlatform); 
tiling.SetWeightShape(256, 128, 3, 3, 3);
tiling.SetInputShape(1, 128, 11, 128, 128);
tiling.SetGradOutputShape(1, 256, 9, 128, 128);
tiling.SetPadding(0, 0, 1, 1, 1, 1);   
tiling.SetDilation(1, 1, 1); 
tiling.SetStride(1, 1, 1);
optiling::Conv3DBackpropInputTilingData tilingData;   
int ret = tiling.GetTiling(tilingData);  // 获取Tiling参数
AscendC::tiling::Conv3DBackpropInputTilingData tilingDataNotOp;
ret = tiling.GetTiling(tilingDataNotOp); // 使用AscendC::tiling::Conv3DBackpropInputTilingData获取Tiling参数
```

```
// 构建Conv3dTranspose算子tiling的调用示例
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
ConvBackpropApi::Conv3DBpInputTiling tiling(ascendcPlatform); 
tiling.SetWeightShape(256, 128, 3, 3, 3);
tiling.SetGradOutputShape(256, 9, 128, 128); // 等价于Conv3dTranspose的输入X
tiling.SetPadding(0, 0, 1, 1, 1, 1);   
tiling.SetDilation(1, 1, 1); 
tiling.SetStride(1, 1, 1);
tiling.SetOutputPadding(0, 0, 0); // 对Conv3dTranspose的输出Y进行padding
optiling::Conv3DBackpropInputTilingData tilingData;   
int ret = tiling.GetTiling(tilingData);  // 获取Tiling参数
```
