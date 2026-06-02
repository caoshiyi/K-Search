# GetOffsetC<a name="ZH-CN_TOPIC_0000002523343764"></a>

## 功能说明<a name="section618mcpsimp"></a>

预留接口，为后续功能做预留。

获取本次计算时当前分片在整个C矩阵中的位置。

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline MatrixOffset GetOffsetC()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

MatrixOffset结构体如下：

```
struct MatrixOffset {   
    int32_t offset;   
    int32_t row, col;   
    int32_t height, width; 
};
```

## 约束说明<a name="section633mcpsimp"></a>

无

