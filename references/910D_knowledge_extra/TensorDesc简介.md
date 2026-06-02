# TensorDesc简介<a name="ZH-CN_TOPIC_0000002554344649"></a>

TensorDesc用于储存[ListTensorDesc](ListTensorDesc.md).GetDesc\(\)中根据index获取对应的Tensor描述信息。

## 原型定义<a name="section10580930144614"></a>

```
template<class T> class TensorDesc {
    TensorDesc();
    ~TensorDesc();
    void SetShapeAddr(uint64_t* shapePtr);
    uint64_t GetDim();
    uint64_t GetIndex();
    uint64_t GetShape(uint32_t offset);
    T* GetDataPtr();
    GlobalTensor<T> GetDataObj();
}
```

## 模板参数<a name="section116801320102618"></a>

**表 1**  模板参数说明

<a name="table13588175515344"></a>
<table><thead align="left"><tr id="row1160915519346"><th class="cellrowborder" valign="top" width="21.8%" id="mcps1.2.3.1.1"><p id="p9609105553412"><a name="p9609105553412"></a><a name="p9609105553412"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="78.2%" id="mcps1.2.3.1.2"><p id="p156091955143419"><a name="p156091955143419"></a><a name="p156091955143419"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row260915573419"><td class="cellrowborder" valign="top" width="21.8%" headers="mcps1.2.3.1.1 "><p id="p45051858102513"><a name="p45051858102513"></a><a name="p45051858102513"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="78.2%" headers="mcps1.2.3.1.2 "><p id="p823866165711"><a name="p823866165711"></a><a name="p823866165711"></a>Tensor数据类型。</p>
</td>
</tr>
</tbody>
</table>

## 成员函数<a name="zh-cn_topic_0000002213064918_section1173524710"></a>

```
[TensorDesc](构造和析构函数.md)()
[~TensorDesc](构造和析构函数.md)()
void [SetShapeAddr](SetShapeAddr.md)(uint64_t* shapePtr)
uint64_t [GetDim](GetDim.md)()
uint64_t [GetIndex](GetIndex.md)()
uint64_t [GetShape](GetShape-159.md)(uint32_t offset)
T* [GetDataPtr](GetDataPtr.md)()
GlobalTensor<T> [GetDataObj](GetDataObj.md)()
```

