# 项目整理结果

旧build的56个可重建Python副本已隔离，353,527 bytes。
来源 `C:\Users\18165\Documents\ChatGPT\MetaEMGFM\build`；隔离 `C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\archive\quarantine_build_20261012\build`。
逐文件原路径/目标/SHA256见 `outputs/archive/quarantine_build_20261012/path_mapping.json`。
永久删除：0。释放空间：0 bytes。实际分类候选不能当作删除授权。
原始/Canonical HDF5、manifest/corpus、venv、历史源码、全部原有checkpoint保留。
初始66个checkpoint已按原路径与SHA256再核验：全部不变。

初始可读盘点：32324文件 / 13,802,453,218 bytes。
这是下限；初始15扫描错误含14旧目录ACL拒绝和当时尚未建立的主工程。
整理后根统计（含本轮新增实验、文档、Git，不可当作清理净减少量）：
```json
[
  {
    "project": "C:\\Users\\18165\\Documents\\ChatGPT\\MetaEMGFM",
    "path": ".",
    "file_count": "32748",
    "total_bytes": "13833901973",
    "last_modified": "2026-10-12T04:07:56.125115+00:00",
    "file_types": "{\".1\": 2, \".a99\": 2, \".al\": 16, \".apache\": 2, \".b99\": 2, \".bat\": 2, \".bazel\": 1, \".bib\": 2, \".bin\": 4, \".bsd\": 2, \".build\": 3, \".bz2\": 4, \".bzl\": 1, \".c\": 8, \".cfg\": 2, \".ckpt\": 10, \".cmake\": 42, \".cpp\": 40, \".csv\": 429, \".cu\": 1, \".cuh\": 57, \".dat\": 1, \".dll\": 62, \".exe\": 43, \".f\": 25, \".f90\": 62, \".f95\": 1, \".fits\": 1, \".g4\": 4, \".gz\": 5, \".h\": 9200, \".h5\": 979, \".hdf5\": 281, \".hpp\": 77, \".html\": 1, \".in\": 2, \".inc\": 1, \".info\": 3, \".ini\": 4, \".interp\": 2, \".js\": 2, \".json\": 207, \".lark\": 2, \".ld\": 1, \".lib\": 91, \".lock\": 49, \".markdown-it\": 1, \".mat\": 3, \".md\": 99, \".mjs\": 2, \".npy\": 5, \".npz\": 2, \".pc\": 2, \".pem\": 2, \".pkl\": 1, \".png\": 10, \".ps1\": 2, \".pt\": 61, \".pth\": 6, \".pxd\": 16, \".pxi\": 1, \".py\": 9464, \".pyc\": 9426, \".pyd\": 119, \".pyf\": 8, \".pyi\": 374, \".pyx\": 22, \".rst\": 1, \".sample\": 14, \".sh\": 1, \".tab\": 4, \".tar\": 12, \".template\": 1, \".thrift\": 1, \".tmpl\": 2, \".tokens\": 2, \".toml\": 2, \".tpl\": 8, \".txt\": 117, \".typed\": 64, \".whl\": 1, \".xbm\": 1, \".xml\": 1, \".xsl\": 3, \".yaml\": 107, \".yml\": 1, \".zi\": 1, \".zip\": 2, \"<none>\": 1038}",
    "contains_hdf5": "True",
    "contains_checkpoint": "True",
    "contains_unique_checkpoint": "True",
    "optimizer_state": "True",
    "ema_state": "True",
    "metric_files": "True",
    "referenced_by": "[]",
    "regenerable": "not_established",
    "classification": "UNKNOWN",
    "incomplete_scan": "True"
  },
  {
    "project": "C:\\Users\\18165\\Documents\\ChatGPT\\metafmodel",
    "path": ".",
    "file_count": "2243",
    "total_bytes": "108938911",
    "last_modified": "2026-10-12T04:11:14.711648+00:00",
    "file_types": "{\".bib\": 1, \".bin\": 2, \".ckpt\": 33, \".csv\": 327, \".h5\": 933, \".hdf5\": 22, \".ini\": 1, \".json\": 155, \".lock\": 21, \".log\": 53, \".md\": 48, \".ps1\": 1, \".py\": 213, \".pyc\": 96, \".sample\": 14, \".tag\": 1, \".toml\": 1, \".txt\": 12, \".whl\": 1, \".yaml\": 123, \"<none>\": 185}",
    "contains_hdf5": "True",
    "contains_checkpoint": "True",
    "contains_unique_checkpoint": "True",
    "optimizer_state": "True",
    "ema_state": "True",
    "metric_files": "True",
    "referenced_by": "[]",
    "regenerable": "not_established",
    "classification": "UNKNOWN",
    "incomplete_scan": "False"
  }
]
```

初始见file_inventory.csv；清理后见post_cleanup/file_inventory.csv。
metric_files仅表示存在JSON/CSV，不是实验指标有效性证明。
历史test/review/reuse输出及ACL拒绝目录UNKNOWN保留，没有整目录按名字删除。

恢复（原build必须不存在，任何父目录junction/reparse会拒绝）：
```powershell
.\scripts\run_local.ps1 test
$pythonExecutable='../MetaEMGFM/.venv/Scripts/python.exe'
& $pythonExecutable scripts/workspace_quarantine.py --restore outputs/archive/quarantine_build_20261012/path_mapping.json
```
恢复先核验所有hash与完整文件列表，不覆盖原路径。
待另行确认永久删除的候选仅此已验证隔离build；本轮没有请求删除。

清理后主工程：312 passed, 36 warnings in 20.53s；历史工程：43 passed, 1 warning in 9.48s。
清理后统一验收：`C:\Users\18165\Documents\ChatGPT\metafmodel\outputs\verification\local_20261012_121912_773000`，13 PASS / 1 BLOCKED / 0 FAIL / 0 NOT_RUN。
