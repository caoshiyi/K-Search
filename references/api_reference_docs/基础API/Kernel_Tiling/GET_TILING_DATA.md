# GET_TILING_DATA

**页面ID:** atlasascendc_api_07_0214  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0214.html

---

#### 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品            /             Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品            /             Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品            AI Core | √ |
| Atlas 推理系列产品            Vector Core | √ |
| Atlas 训练系列产品 | x |

#### 功能说明

用于获取算子kernel入口函数传入的Tiling信息，并填入注册的TilingData结构体中，此函数会以宏展开的方式进行编译。对应的算子host实现中需要定义TilingData结构体，实现并注册计算TilingData的Tiling函数，具体请参考Host侧Tiling实现。如果用户通过TilingData结构注册注册了多个TilingData结构体，使用该接口返回默认注册的结构体。

#### 函数原型

```
GET_TILING_DATA(tiling_data, tiling_arg)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| tiling_data | 输出 | 返回默认TilingData结构体变量。 |
| tiling_arg | 输入 | 此参数为算子入口函数处传入的Tiling参数。 |

#### 约束说明

- 本函数需在算子kernel代码处使用，并且传入的tiling_data参数不需要声明类型。
- 暂不支持kernel直调工程。

#### 调用示例

```
extern "C" __global__ __aicore__ void add_custom(__gm__ uint8_t *x, __gm__ uint8_t *y, __gm__ uint8_t *z, __gm__ uint8_t *tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.blkDim, tilingData.totalSize, tilingData.splitTile);
    op.Process();
}
```

     配套的host侧Tiling函数示例：

```
ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 其他代码逻辑
    ...
    TilingData tiling;  // 与算子host实现中定义TilingData结构体的对应
    tiling.set_blkDim(blockDim);  // 与算子host实现中定义TilingData结构体中的成员的对应
    tiling.set_totalSize(totalSize);
    tiling.set_splitTile(splitTile);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    ...
    // 其他代码逻辑
}
```
