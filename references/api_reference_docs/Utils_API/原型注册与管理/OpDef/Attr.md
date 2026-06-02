# Attr

**页面ID:** atlasascendc_api_07_0949  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0949.html

---

#### 功能说明

注册算子属性参数。

当需要设置的参数不参与kernel侧计算时，可以将该参数注册为算子属性参数。

#### 函数原型

```
OpAttrDef &Attr(const char *name)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| name | 输入 | 算子属性名称。 |

#### 返回值说明

算子属性定义，OpAttrDef请参考OpAttrDef。

#### 约束说明

Attr属性名不能与以下python关键字及内置变量名相同，否则会导致未定义错误。

- 常见python关键字参考

and、 as、 assert、 break、 class、 continue、 def、 del、 elif、 else、 except、 finally、 for、 from、 global、 if、 import、 in、 is、 lambda、 not、 or、 pass、 raise、 return、 try、 while、 with、 yield、 False、 None、 True、 nonlocal、 arg。

- 内置变量名

__inputs__、 __outputs__、 __attrs__、 options、 bisheng、 bisheng_path、 tikcpp_path、 impl_mode、 custom_compile_options、 custom_all_compile_options、 soc_version、 soc_short、 custom_compile_options_soc、 custom_all_compile_options_soc、 origin_func_name、 ascendc_src_dir_ex、 ascendc_src_dir、 ascendc_src_file、 src、 op_type、 code_channel、 op_info、 compile_op、 get_code_channel、 result、 isinstance、 attr、 get_current_build_config、 _build_args、 get_dtype_fmt_options、 shutil、 os、 get_kernel_source、ascendc_api_version_header_path、ascendc_api_version_file、ascendc_api_version、re。
