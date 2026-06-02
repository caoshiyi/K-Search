# GetFormat

**页面ID:** atlasopapi_07_00013  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/basicdataapi/atlasopapi_07_00013.html

---

#### 函数功能

获取编译时Tensor的数据排布格式。

#### 函数原型

```
const StorageFormat &GetFormat() const
```

#### 参数说明

无。

#### 返回值说明

返回数据排布格式。StorageFormat类型，包括原始format和存储format。

#### 约束说明

无。

#### 调用示例

```
StorageFormat storageFormat(ge::Format::FORMAT_NC, ge::FORMAT_NCHW, {});
gert::CompileTimeTensorDesc compileTimeTensorDesc;
compileTimeTensorDesc.SetStorageFormat(storageFormat.GetStorageFormat());
auto storage_fmt = compileTimeTensorDesc.GetFormat(); // ge::FORMAT_NCHW
compileTimeTensorDesc.SetOriginFormat(storageFormat.GetOriginFormat());
auto origin_fmt = compileTimeTensorDesc.GetOriginFormat(); // ge::Format::FORMAT_NC
```
