# GetOffsetC

**页面ID:** atlasascendc_api_07_0660  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0660.html

---

#### 功能说明

预留接口，为后续功能做预留。

获取本次计算时当前分片在整个C矩阵中的位置。

#### 函数原型

```
__aicore__ inline MatrixOffset GetOffsetC()
```

#### 参数说明

无

#### 返回值说明

MatrixOffset结构体如下：

```
struct MatrixOffset {   
    int32_t offset;   
    int32_t row, col;   
    int32_t height, width; 
};
```

#### 约束说明

无
