# GET_TILING_DATA_PTR_WITH_STRUCT

**页面ID:** atlasascendc_api_07_00140  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00140.html

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

在使用该宏时，开发者可以通过指定结构体名称来获取相应的Tiling信息，并将其填入对应的Tiling结构体中。完成填充后，该宏将返回一个指向该Tiling结构体的指针，并使用__tiling_data_ptr__修饰符对该指针进行修饰。这种修饰方式能够确保在动静态Shape场景下代码的统一性和兼容性。

#### 函数原型

```
GET_TILING_DATA_PTR_WITH_STRUCT(tiling_struct, dst_ptr, tiling_ptr)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_struct | 输入 | 指定的结构体名称。 |
| dst_ptr | 输出 | 返回指定的Tiling结构体指针。 |
| tiling_ptr | 输入 | 算子入口函数处传入的Tiling参数。 |

#### 约束说明

- 该宏需在算子Kernel代码处使用，并且传入的dst_ptr参数无需声明类型。
- 动态Shape场景下，获取到的dst_ptr是指向Global Memory变量的指针；静态Shape场景下，获取到的dst_ptr是指向局部变量的指针，需确保在合理的作用域范围内使用。
- 暂不支持Kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;

    GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);

    op.Init(x, y, z, tilingDataPtr->totalLength, tilingDataPtr->tileNum);
    op.Process();

}
```

以下是错误调用的示例：

```
__aicore__ __tiling_data_ptr__ AddCustomTilingData* foo(__gm__ uint8_t *tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(AddCustomTilingData, tilingDataPtr, tiling);
    return tilingDataPtr;
}

extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;

    auto tilingDataPtr = foo(tiling);  // 错误，foo函数已经执行完成，非法访问生命周期已经结束的局部变量

    op.Init(x, y, z, tilingDataPtr->totalLength, tilingDataPtr->tileNum);
    op.Process();
}
```
