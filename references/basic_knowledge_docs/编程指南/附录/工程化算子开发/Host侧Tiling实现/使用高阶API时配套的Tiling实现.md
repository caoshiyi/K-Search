# 使用高阶API时配套的Tiling实现

**页面ID:** atlas_ascendc_10_00023  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00023.html

---

1. 首先进行tiling结构定义：

```
namespace optiling {
BEGIN_TILING_DATA_DEF(MyAddTilingData)  // 声明tiling结构名字
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);   // 引用高阶API的tiling结构体
  TILING_DATA_FIELD_DEF(uint32_t, field);   // 结构成员的引用结构体
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MyAdd, MyAddTilingData)  // tiling结构注册给算子
}
```

2. 通过高阶API配套的tiling函数对tiling结构初始化：

```
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    int32_t M = 1024;
    int32_t N = 640;
    int32_t K = 256;
    int32_t baseM = 128;
    int32_t baseN = 128;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(TPosition::LCM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetFixSplit(baseM, baseN, -1);
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    MyAddTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1){
        return ge::GRAPH_FAILED;
    }
    // some code
}
```
