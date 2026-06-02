# COPY_TILING_WITH_STRUCT

**页面ID:** atlasascendc_api_07_00141  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00141.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | √ |
| Atlas 推理系列产品Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

拷贝Tiling结构体，并返回指向拷贝后的Tiling结构体的指针。该宏适用于嵌套结构体场景，可拷贝结构体的子结构体成员变量。该宏将指定结构体拷贝至栈上，适用于频繁访问Tiling数据的场景，能够加快数据访问速度。

#### 函数原型

```
COPY_TILING_WITH_STRUCT(tiling_struct, src_ptr, dst_ptr)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_struct | 输入 | 指定的结构体名称。 |
| src_ptr | 输入 | 指向tiling_struct结构体的指针。 |
| dst_ptr | 输出 | 返回拷贝后的指向tiling_struct结构体的指针。 |

#### 约束说明

- 该宏需在算子Kernel代码处使用，并且传入的dst_ptr参数无需声明类型。
- 该宏需要和GET_TILING_DATA_PTR_WITH_STRUCT配合使用，输入参数src_ptr为GET_TILING_DATA_PTR_WITH_STRUCT获取到的指针。
- 该宏获取到的dst_ptr指针指向的Tiling结构体是局部变量，请确保在合理作用域范围内使用。
- 暂不支持Kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;
    GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);

    if ASCEND_IS_AIV {
        COPY_TILING_WITH_STRUCT(VectorTilingStruct, tilingDataPtr->vectorTiling, vTilingStructPtr);   // Vector侧使用VectorTilingStruct
        op.Init(x, y, z, vTilingStructPtr->totalLength, vTilingStructPtr->tileNum);
        op.Process();
    } else {
        COPY_TILING_WITH_STRUCT(CubeTilingStruct, tilingDataPtr->cubeTiling, cTilingStructPtr); // Cube侧使用CubeTilingStruct
	op.Init(x, y, z, *cTilingStructPtr);
        op.Process();
    }
}
```
