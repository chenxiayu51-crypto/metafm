# 安全清理计划（清理前）

file_inventory.csv是只读扫描；目录分类为候选判断，非删除授权。SHA256、checkpoint的optimizer/teacher/scheduler键及扫描错误保存在inventory_details.json。

KEEP：全部原始/canonical真实HDF5、官方corpus、清单、源码、配置、文档、Git、现有venv。ACTIVE：有代码/配置/文档引用的路径以及新训练恢复依赖。ARCHIVE：有checkpoint或独特结果但无当前引用的历史产物。CLEANABLE：无checkpoint、无引用且可重生成的已识别构建/缓存。UNKNOWN：ACL不可读、HDF5来源或用途无法确定；保留。

禁止永久删除。metafmodel_verification与canonical_wrist_train、全部唯一checkpoint保持原位。历史test/review/reuse输出不根据名称直接清除，合成HDF5在确认来源前也不移动。

新主工程通过统一验收后，仅考虑移动本次确认可重建的旧build/和无依赖Python缓存到outputs/archive/quarantine_<run_id>。逐文件记录原路径、隔离路径、SHA256，提供恢复命令。当前ACTlVE数据/环境不能移动。隔离不释放空间，实际释放量为0。若核对发现引用则取消移动。

最终执行/未执行清单及回归结果写入新主工程reports/project_cleanup_summary.md；永久删除候选仅列出，等待用户另行授权。
