# 调用算子时出现无法打开config.ini的报错

**页面ID:** atlas_ascendc_10_00003  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00003.html

---

#### 现象描述

自定义算子包安装部署后，在调用已部署的算子时，出现如下json文件获取失败的报错信息：

```
[INFO] Start get path and read binary_info_config.json.
[WARNING] Get jsonfile path for */binary_info_config.json failed, errmsg:No such file or directory.
[ERROR] Get path and read binary_info_config.json failed, please check if the opp_kernel package is installed!
```

通过查询前文的报错信息，上述json文件获取失败的原因是前置流程中无法打开config.ini，提示信息如下：

```
[INFO] Start to get opp kernel base path, default custom opp kernel is in ASCEND_OPP_PATH.
[INFO] The real path of config.ini is */opp/vendors/config.ini.
[WARNING]  Can not open file: */opp/vendors/config.ini.
```

#### 问题根因

根因在于当前调用算子的用户缺少对算子包部署目录下的config.ini（ */opp/vendors/config.ini）文件的读权限。config.ini文件权限默认为640，仅允许部署用户和同属组用户访问，当前执行用户与安装用户非同一属组，缺少读权限，导致算子调用失败。

比如下述场景即会导致调用算子时出现执行报错：root用户安装部署自定义算子包，HwHiAiUser属组用户调用已部署的自定义算子。

#### 处理步骤

联系自定义算子包安装用户修改config.ini权限为644：

```
chmod 644 config.ini
```
