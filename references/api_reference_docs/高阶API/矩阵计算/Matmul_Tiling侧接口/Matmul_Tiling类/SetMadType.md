# SetMadType

**页面ID:** atlasascendc_api_07_0686  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0686.html

---

#### 功能说明

设置是否使能HF32模式。**当前版本暂不支持。**

#### 函数原型

```
int32_t SetMadType(MatrixMadType madType)
```

#### 参数说明

**表1 **参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| 设置Matmul模式。MatrixMadType类型，定义如下。 ``` enum class MatrixMadType : int32_t { NORMAL = 0, HF32 = 1,  }; ```  - MatrixMadType::NORMAL：普通模式，即非HF32模式。- MatrixMadType::HF32：使能HF32模式。 |  |  |

#### 返回值说明

-1表示设置失败； 0表示设置成功。

#### 约束说明

无
