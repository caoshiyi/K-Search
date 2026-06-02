# LoadTilingLibrary

**页面ID:** atlasascendc_api_07_00075  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00075.html

---

#### 功能说明

根据输入的路径，加载对应的Tiling动态库。开发者基于工程化算子开发开发方式完成算子实现后，可通过**算子包编译**或**算子动态库编译**获取对应的Tiling动态库文件。

- 算子包编译：Tiling实现对应的动态库为算子包部署目录下的liboptiling.so。具体路径可参考算子包部署。

- 动态库编译：Tiling实现集成在算子动态库libcust_opapi.so中。具体路径可参考算子动态库和静态库编译。

#### 函数原型

```
bool LoadTilingLibrary(const char *tilingSoPath) const
```

#### 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tilingSoPath | 输入 | Tiling动态库的路径，支持相对路径与绝对路径。 |

#### 返回值说明

true：Tiling动态库加载成功；false：Tiling动态库加载失败。具体错误可参考Log信息。

关于日志配置和查看，请参考日志。

#### 约束说明

无

#### 调用示例

```
context_ascendc::OpTilingRegistry tmpIns;
bool flag = tmpIns.LoadTilingLibrary("/your/path/to/so_path/liboptiling.so");
if (flag == false) {
    std::cout << "Load tiling so failed" << std::endl;
    ...        
}
// ...
```
