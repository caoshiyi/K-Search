# REGIST_MATMUL_OBJ

**页面ID:** atlasascendc_api_07_0628  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0628.html

---

#### 功能说明

初始化Matmul对象。

#### 函数原型

```
REGIST_MATMUL_OBJ(tpipe, workspace, ...)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tpipe | 输入 | Tpipe对象。 |
| workspace | 输入 | 系统workspace指针。 |
| ... | 输入 | 可变参数，传入Matmul对象和与之对应的Tiling结构，要求Tiling结构的数据类型为TCubeTiling结构。          Tiling参数可以通过Host侧GetTiling接口获取，并传递到kernel侧使用。 |

#### 约束说明

- 在分离模式中，本接口必须在InitBuffer接口前调用。
- 在程序中，最多支持定义4个Matmul对象。
- 当代码中只有一个Matmul对象时，本接口可以不传入tiling参数，通过Init接口单独传入tiling参数。
- 当代码中有多个Matmul对象时，必须满足Matmul对象与其tiling参数一一对应，依次传入，具体方式请参考调用示例。

#### 调用示例

```
Tpipe pipe;
// 推荐：初始化单个matmul对象，传入tiling参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
// 推荐：初始化多个matmul对象，传入对应的tiling参数
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, mm1tiling, mm2, mm2tiling, mm3, mm3tiling, mm4, mm4tiling);
// 初始化单个matmul对象，未传入tiling参数。注意，该场景下需要使用Init接口单独传入tiling参数。这种方式将matmul对象的初始化和tiling的设置分离，比如，Tiling可变的场景，可通过这种方式多次对Tiling进行重新设置
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm);
mm.Init(&tiling);
```
