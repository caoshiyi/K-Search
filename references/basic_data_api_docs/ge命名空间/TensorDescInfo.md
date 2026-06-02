# TensorDescInfo

**页面ID:** atlasopapi_07_00428  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00428.html

---

```
struct TensorDescInfo {
        Format format_ = FORMAT_RESERVED;        /* tbe op register support format */
        DataType dataType_ = DT_UNDEFINED;       /* tbe op register support datatype */
    };
```

Format为枚举类型，定义请参考Format。

DataType为枚举类型，定义请参考DataType。
