# GET_TILING_DATA_WITH_STRUCT

**页面ID:** atlasascendc_api_07_0215  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0215.html

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

使用该接口指定结构体名称，可获取指定的tiling信息，并填入对应的Tiling结构体中，此函数会以宏展开的方式进行编译。与GET_TILING_DATA的区别是：GET_TILING_DATA只能获取默认注册的结构体，该接口可以根据指定的结构体名称获取对应的结构体，常用于针对不同的TilingKey注册了不同结构体的情况下。

#### 函数原型

```
GET_TILING_DATA_WITH_STRUCT(struct_name, tiling_data, tiling_arg)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| struct_name | 输入 | 指定的结构体名称。 |
| tiling_data | 输出 | 返回指定Tiling结构体变量。 |
| tiling_arg | 输入 | 此参数为算子入口函数处传入的tiling参数。 |

#### 约束说明

- 本函数需在算子Kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
- 暂不支持Kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    KernelAdd op;
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(Add_Struct_Special, tilingData, tiling); // 使用算子指定注册的结构体
	op.Init(x, y, z, tilingData.totalLengthSpecial, tilingData.tileNumSpecial);
    } else {
        GET_TILING_DATA(tilingData, tiling);   // 使用算子默认注册的结构体
	op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    }
    if (TILING_KEY_IS(1)) {
        op.Process();
    }  else  if (TILING_KEY_IS(2)) {
        op.Process();
    } else  if (TILING_KEY_IS(3)) {
        op.Process();
    }
}
```
