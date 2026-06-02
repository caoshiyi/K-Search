# GetOutputShape

**页面ID:** atlasopapi_07_00711  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00711.html

---

#### 函数功能

获取算子输出张量的实际存储形状。

#### 函数原型

```
const gert::StorageShape* GetOutputShape(int64_t index) const
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 算子输出索引。 这里的输出索引是指算子实例化后实际的索引，不是原型定义中的索引。 |

#### 返回值说明

返回StorageShape结构体，表示该输出张量的存储形状。

#### 约束说明

使用时必须确保：不要对返回的StorageShape做修改，也不要试图释放它。

#### 调用示例

```
gert::StorageShape* GetOutputShape(OpCheckContext* context) {
  int64_t index = 9;
  gert::StorageShape* shape = context->GetOutputShape(index);
  ...
}
```
