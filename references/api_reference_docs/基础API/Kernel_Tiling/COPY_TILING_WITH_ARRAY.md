# COPY_TILING_WITH_ARRAY

**页面ID:** atlasascendc_api_07_00142  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00142.html

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

拷贝指定大小的数组内容到目标数组中，并返回指向拷贝后数组的指针。适用于拷贝一个结构体的数组成员变量的场景。该宏将指定数组拷贝至栈上，适用于频繁访问Tiling数据的场景，能够加快数据访问速度。

#### 函数原型

```
COPY_TILING_WITH_ARRAY(arr_type, arr_count, src_ptr, dst_ptr)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| arr_type | 输入 | 指定要拷贝的数组类型。 |
| arr_count | 输入 | 指定要拷贝的数组大小。 |
| src_ptr | 输入 | 指向tiling_struct结构体的指针。 |
| dst_ptr | 输出 | 返回拷贝后的指向类型为arr_type、大小为arr_count的数组指针。 |

#### 约束说明

- 该宏需在算子Kernel代码处使用，并且传入的dst_ptr参数无需声明类型。
- 该宏需要和GET_TILING_DATA_PTR_WITH_STRUCT配合使用，输入参数src_ptr为GET_TILING_DATA_PTR_WITH_STRUCT获取到的指针。
- 该宏获取到的dst_ptr指针指向的数组是局部变量，请确保在合理作用域范围内使用。
- 暂不支持Kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;

    GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);

    if ASCEND_IS_AIV {
        COPY_TILING_WITH_ARRAY(uint64_t, 2, tilingDataPtr->vectorTilingArray, vTilingArrayPtr);        
        op.Init(x, y, z, (*vTilingArrayPtr)[0], (*vTilingArrayPtr)[1]);
        op.Process();
    } else {
        COPY_TILING_WITH_ARRAY(uint64_t, 2, tilingDataPtr->cubeTilingArray, cTilingArrayPtr);
	op.Init(x, y, z, (*cTilingArrayPtr)[0], (*cTilingArrayPtr)[1]);
	op.Process();
    }
}
```
