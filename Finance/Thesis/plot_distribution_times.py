import re
import matplotlib.pyplot as plt
import numpy as np

from Finance.Thesis.palette import palette

# Raw log text (paste your log as a string)
log_text = """
[GIN] 2025/08/22 - 04:17:53 | 200 |   34.8108849s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:17:56.652+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=107284 keep=5 new=2048
[GIN] 2025/08/22 - 04:19:08 | 200 |         1m12s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:19:11.797+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=55208 keep=5 new=2048
[GIN] 2025/08/22 - 04:19:53 | 200 |   41.7887208s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:19:55.418+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=19741 keep=5 new=2048
[GIN] 2025/08/22 - 04:21:13 | 200 |         1m17s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:21:16.206+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=73197 keep=5 new=2048
[GIN] 2025/08/22 - 04:22:27 | 200 |         1m11s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:22:30.576+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=41930 keep=5 new=2048
[GIN] 2025/08/22 - 04:24:41 | 200 |         2m10s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:24:49.178+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=244045 keep=5 new=2048
[GIN] 2025/08/22 - 04:25:53 | 200 |          1m6s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:26:00.930+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=229120 keep=5 new=2048
[GIN] 2025/08/22 - 04:28:59 | 200 |          3m0s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:29:03.086+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=17741 keep=5 new=2048
[GIN] 2025/08/22 - 04:31:01 | 200 |         1m58s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:31:05.680+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=257416 keep=5 new=2048
[GIN] 2025/08/22 - 04:34:25 | 200 |         3m21s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:34:30.727+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=266895 keep=5 new=2048
[GIN] 2025/08/22 - 04:36:43 | 200 |         2m14s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:36:51.954+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=385666 keep=5 new=2048
[GIN] 2025/08/22 - 04:39:19 | 200 |         2m30s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:39:22.190+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=3102 keep=5 new=2048
[GIN] 2025/08/22 - 04:42:00 | 200 |         2m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:42:07.194+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=230725 keep=5 new=2048
[GIN] 2025/08/22 - 04:44:47 | 200 |         2m42s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:44:51.264+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=79453 keep=5 new=2048
[GIN] 2025/08/22 - 04:48:32 | 200 |         3m41s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:48:36.214+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=83829 keep=5 new=2048
[GIN] 2025/08/22 - 04:50:54 | 200 |         2m18s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:50:56.846+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=26156 keep=5 new=2048
[GIN] 2025/08/22 - 04:53:07 | 200 |         2m11s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:53:12.129+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=278655 keep=5 new=2048
[GIN] 2025/08/22 - 04:55:31 | 200 |         2m20s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:55:34.682+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=55925 keep=5 new=2048
[GIN] 2025/08/22 - 04:59:29 | 200 |         3m54s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T04:59:32.287+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=55396 keep=5 new=2048
[GIN] 2025/08/22 - 05:02:35 | 200 |          3m3s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:02:41.635+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=581950 keep=5 new=2048
[GIN] 2025/08/22 - 05:06:10 | 200 |         3m30s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:06:13.332+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=59494 keep=5 new=2048
[GIN] 2025/08/22 - 05:07:56 | 200 |         1m43s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:07:58.941+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=5774 keep=5 new=2048
[GIN] 2025/08/22 - 05:10:07 | 200 |          2m9s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:10:12.885+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=202134 keep=5 new=2048
[GIN] 2025/08/22 - 05:11:45 | 200 |         1m34s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:11:51.673+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=266507 keep=5 new=2048
[GIN] 2025/08/22 - 05:15:05 | 200 |         3m15s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:15:10.647+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=554737 keep=5 new=2048
[GIN] 2025/08/22 - 05:16:56 | 200 |         1m47s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:16:59.870+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=71145 keep=5 new=2048
[GIN] 2025/08/22 - 05:19:24 | 200 |         2m25s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:20:27.081+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=4324032 keep=5 new=2048
[GIN] 2025/08/22 - 05:21:49 | 200 |         1m46s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:21:53.300+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=88131 keep=5 new=2048
[GIN] 2025/08/22 - 05:23:16 | 200 |         1m24s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:23:22.713+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=677021 keep=5 new=2048
[GIN] 2025/08/22 - 05:26:20 | 200 |          3m0s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:26:23.821+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=49820 keep=5 new=2048
[GIN] 2025/08/22 - 05:50:27 | 200 |         24m3s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:50:30.756+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=82862 keep=5 new=2048
[GIN] 2025/08/22 - 05:53:17 | 200 |         2m47s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:53:20.457+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=83128 keep=5 new=2048
[GIN] 2025/08/22 - 05:56:22 | 200 |          3m2s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T05:56:24.901+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=2097 keep=5 new=2048
[GIN] 2025/08/22 - 06:00:46 | 200 |         4m22s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:00:49.723+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=33743 keep=5 new=2048
[GIN] 2025/08/22 - 06:06:08 | 200 |         5m18s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:06:13.549+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=212405 keep=5 new=2048
[GIN] 2025/08/22 - 06:10:55 | 200 |         4m43s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:11:03.632+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=408730 keep=5 new=2048
[GIN] 2025/08/22 - 06:17:30 | 200 |         6m29s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:17:33.273+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=38014 keep=5 new=2048
[GIN] 2025/08/22 - 06:20:08 | 200 |         2m35s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:20:10.637+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=6661 keep=5 new=2048
[GIN] 2025/08/22 - 06:23:46 | 200 |         3m36s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:23:57.986+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=292771 keep=5 new=2048
[GIN] 2025/08/22 - 06:28:06 | 200 |         4m10s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:28:10.911+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=128926 keep=5 new=2048
[GIN] 2025/08/22 - 06:30:57 | 200 |         2m47s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:31:00.423+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=30145 keep=5 new=2048
[GIN] 2025/08/22 - 06:34:33 | 200 |         3m33s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:34:38.499+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=273450 keep=5 new=2048
[GIN] 2025/08/22 - 06:38:14 | 200 |         3m37s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:38:17.199+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=17242 keep=5 new=2048
[GIN] 2025/08/22 - 06:41:21 | 200 |          3m4s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:41:24.294+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=30087 keep=5 new=2048
[GIN] 2025/08/22 - 06:45:44 | 200 |         4m20s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:45:47.198+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=35197 keep=5 new=2048
[GIN] 2025/08/22 - 06:47:44 | 200 |         1m57s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:47:46.775+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=34436 keep=5 new=2048
[GIN] 2025/08/22 - 06:52:12 | 200 |         4m26s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:52:16.854+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=135492 keep=5 new=2048
[GIN] 2025/08/22 - 06:56:44 | 200 |         4m28s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:56:48.016+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=71236 keep=5 new=2048
[GIN] 2025/08/22 - 06:59:05 | 200 |         2m17s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T06:59:08.695+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=143948 keep=5 new=2048
[GIN] 2025/08/22 - 07:03:03 | 200 |         3m55s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:03:06.829+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=81774 keep=5 new=2048
[GIN] 2025/08/22 - 07:06:44 | 200 |         3m37s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:06:49.991+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=332340 keep=5 new=2048
[GIN] 2025/08/22 - 07:12:05 | 200 |         5m17s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:12:11.255+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=328516 keep=5 new=2048
[GIN] 2025/08/22 - 07:16:25 | 200 |         4m16s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:16:28.521+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=27388 keep=5 new=2048
[GIN] 2025/08/22 - 07:21:07 | 200 |         4m39s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:21:12.303+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=147993 keep=5 new=2048
[GIN] 2025/08/22 - 07:25:04 | 200 |         3m53s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:25:07.818+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=59043 keep=5 new=2048
[GIN] 2025/08/22 - 07:30:03 | 200 |         4m56s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:30:07.718+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=105940 keep=5 new=2048
[GIN] 2025/08/22 - 07:34:08 | 200 |          4m0s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:34:13.807+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=115984 keep=5 new=2048
[GIN] 2025/08/22 - 07:38:15 | 200 |          4m2s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:38:20.147+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=103178 keep=5 new=2048
[GIN] 2025/08/22 - 07:41:34 | 200 |         3m15s |       127.0.0.1 | POST     "/api/generate"
[GIN] 2025/08/22 - 07:45:47 | 200 |         4m10s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:45:54.335+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=178628 keep=5 new=2048
[GIN] 2025/08/22 - 07:48:14 | 200 |         2m20s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:48:19.330+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=175561 keep=5 new=2048
[GIN] 2025/08/22 - 07:50:08 | 200 |         1m50s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:50:11.877+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=52810 keep=5 new=2048
[GIN] 2025/08/22 - 07:53:20 | 200 |          3m8s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:53:23.974+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=101544 keep=5 new=2048
[GIN] 2025/08/22 - 07:57:38 | 200 |         4m15s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T07:57:40.681+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=10350 keep=5 new=2048
[GIN] 2025/08/22 - 08:00:30 | 200 |         2m50s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:00:33.315+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=18978 keep=5 new=2048
[GIN] 2025/08/22 - 08:04:25 | 200 |         3m52s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:04:29.704+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=103277 keep=5 new=2048
[GIN] 2025/08/22 - 08:08:06 | 200 |         3m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:08:11.988+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=147837 keep=5 new=2048
[GIN] 2025/08/22 - 08:12:06 | 200 |         3m55s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:12:09.231+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=31256 keep=5 new=2048
[GIN] 2025/08/22 - 08:16:15 | 200 |          4m6s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:16:19.227+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=106213 keep=5 new=2048
[GIN] 2025/08/22 - 08:19:20 | 200 |          3m1s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:19:23.280+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=52262 keep=5 new=2048
[GIN] 2025/08/22 - 08:21:52 | 200 |         2m29s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:22:00.778+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=234207 keep=5 new=2048
[GIN] 2025/08/22 - 08:26:04 | 200 |          4m4s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:26:08.405+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=131970 keep=5 new=2048
[GIN] 2025/08/22 - 08:31:06 | 200 |         4m58s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:31:10.389+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=187730 keep=5 new=2048
[GIN] 2025/08/22 - 08:34:52 | 200 |         3m43s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:34:56.033+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=81012 keep=5 new=2048
[GIN] 2025/08/22 - 08:38:21 | 200 |         3m25s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:38:24.340+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=31077 keep=5 new=2048
[GIN] 2025/08/22 - 08:44:32 | 200 |          6m7s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:44:35.049+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=56962 keep=5 new=2048
[GIN] 2025/08/22 - 08:48:17 | 200 |         3m42s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:48:25.627+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=227661 keep=5 new=2048
[GIN] 2025/08/22 - 08:52:41 | 200 |         4m18s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:52:44.985+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=87775 keep=5 new=2048
[GIN] 2025/08/22 - 08:55:58 | 200 |         3m13s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T08:56:02.128+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=78987 keep=5 new=2048
[GIN] 2025/08/22 - 09:00:04 | 200 |          4m2s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:00:09.679+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=172768 keep=5 new=2048
[GIN] 2025/08/22 - 09:03:19 | 200 |         3m10s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:03:21.664+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=27879 keep=5 new=2048
[GIN] 2025/08/22 - 09:06:48 | 200 |         3m26s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:06:50.599+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=36361 keep=5 new=2048
[GIN] 2025/08/22 - 09:10:49 | 200 |         3m59s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:10:57.472+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=257161 keep=5 new=2048
[GIN] 2025/08/22 - 09:15:04 | 200 |          4m9s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:15:09.807+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=278455 keep=5 new=2048
[GIN] 2025/08/22 - 09:18:24 | 200 |         3m15s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:18:28.045+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=62024 keep=5 new=2048
[GIN] 2025/08/22 - 09:21:32 | 200 |          3m4s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:21:35.264+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=37457 keep=5 new=2048
[GIN] 2025/08/22 - 09:24:03 | 200 |         2m28s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:24:10.406+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=301053 keep=5 new=2048
[GIN] 2025/08/22 - 09:28:22 | 200 |         4m13s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:28:24.758+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=19601 keep=5 new=2048
[GIN] 2025/08/22 - 09:31:37 | 200 |         3m12s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:31:39.766+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=31689 keep=5 new=2048
[GIN] 2025/08/22 - 09:41:24 | 200 |         9m44s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:41:28.743+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=141917 keep=5 new=2048
[GIN] 2025/08/22 - 09:45:19 | 200 |         3m51s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:45:23.902+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=196499 keep=5 new=2048
[GIN] 2025/08/22 - 09:48:56 | 200 |         3m33s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:49:04.160+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=238396 keep=5 new=2048
[GIN] 2025/08/22 - 09:52:32 | 200 |         3m29s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:52:36.578+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=184738 keep=5 new=2048
[GIN] 2025/08/22 - 09:56:05 | 200 |         3m29s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:56:08.078+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=60899 keep=5 new=2048
[GIN] 2025/08/22 - 09:59:08 | 200 |          3m0s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T09:59:20.193+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=126822 keep=5 new=2048
[GIN] 2025/08/22 - 10:02:04 | 200 |         2m46s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:02:08.703+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=104675 keep=5 new=2048
[GIN] 2025/08/22 - 10:14:37 | 200 |        12m30s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:14:42.163+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=119885 keep=5 new=2048
[GIN] 2025/08/22 - 10:20:20 | 200 |         5m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:20:24.620+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=157347 keep=5 new=2048
[GIN] 2025/08/22 - 10:26:22 | 200 |         5m58s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:26:26.784+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=103576 keep=5 new=2048
[GIN] 2025/08/22 - 10:29:56 | 200 |         3m31s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:30:00.044+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=80368 keep=5 new=2048
[GIN] 2025/08/22 - 10:37:05 | 200 |          7m5s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:37:11.298+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=266340 keep=5 new=2048
[GIN] 2025/08/22 - 10:40:30 | 200 |         3m21s |       127.0.0.1 | POST     "/api/generate"
[GIN] 2025/08/22 - 10:43:00 | 200 |         2m28s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:43:06.567+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=251811 keep=5 new=2048
[GIN] 2025/08/22 - 10:46:55 | 200 |         3m50s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:46:59.929+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=118901 keep=5 new=2048
[GIN] 2025/08/22 - 10:51:55 | 200 |         4m55s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:51:58.238+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=18721 keep=5 new=2048
[GIN] 2025/08/22 - 10:54:58 | 200 |          3m0s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:55:02.727+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=171221 keep=5 new=2048
[GIN] 2025/08/22 - 10:59:40 | 200 |         4m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T10:59:45.461+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=180618 keep=5 new=2048
[GIN] 2025/08/22 - 11:04:22 | 200 |         4m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:04:25.879+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=35441 keep=5 new=2048
[GIN] 2025/08/22 - 11:09:35 | 200 |         5m10s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:09:37.911+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=5486 keep=5 new=2048
[GIN] 2025/08/22 - 11:13:11 | 200 |         3m33s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:13:15.857+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=149953 keep=5 new=2048
[GIN] 2025/08/22 - 11:19:40 | 200 |         6m25s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:19:43.082+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=12257 keep=5 new=2048
[GIN] 2025/08/22 - 11:22:47 | 200 |          3m4s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:22:49.844+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=10833 keep=5 new=2048
[GIN] 2025/08/22 - 11:26:20 | 200 |         3m30s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:26:23.364+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=63990 keep=5 new=2048
[GIN] 2025/08/22 - 11:33:46 | 200 |         7m23s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:33:51.672+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=158152 keep=5 new=2048
[GIN] 2025/08/22 - 11:39:44 | 200 |         5m54s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:39:47.391+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=7228 keep=5 new=2048
[GIN] 2025/08/22 - 11:42:57 | 200 |          3m9s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:43:03.257+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=204459 keep=5 new=2048
[GIN] 2025/08/22 - 11:45:48 | 200 |         2m46s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:45:50.367+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=2442 keep=5 new=2048
[GIN] 2025/08/22 - 11:48:54 | 200 |          3m3s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:48:56.612+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=30698 keep=5 new=2048
[GIN] 2025/08/22 - 11:52:49 | 200 |         3m52s |       127.0.0.1 | POST     "/api/generate"
[GIN] 2025/08/22 - 11:54:55 | 200 |          2m4s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:54:58.367+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=28019 keep=5 new=2048
[GIN] 2025/08/22 - 11:57:56 | 200 |         2m58s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T11:57:59.892+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=71192 keep=5 new=2048
[GIN] 2025/08/22 - 12:02:31 | 200 |         4m31s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:02:34.689+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=50179 keep=5 new=2048
[GIN] 2025/08/22 - 12:06:03 | 200 |         3m29s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:06:05.764+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=30361 keep=5 new=2048
[GIN] 2025/08/22 - 12:10:56 | 200 |         4m50s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:10:59.314+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=38540 keep=5 new=2048
[GIN] 2025/08/22 - 12:16:35 | 200 |         5m36s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:16:43.196+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=469000 keep=5 new=2048
[GIN] 2025/08/22 - 12:20:57 | 200 |         4m19s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:21:00.803+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=39017 keep=5 new=2048
[GIN] 2025/08/22 - 12:26:06 | 200 |          5m5s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:26:10.429+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=95593 keep=5 new=2048
[GIN] 2025/08/22 - 12:31:47 | 200 |         5m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:31:49.985+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=2474 keep=5 new=2048
[GIN] 2025/08/22 - 12:35:35 | 200 |         3m45s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:35:38.235+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=10089 keep=5 new=2048
[GIN] 2025/08/22 - 12:40:38 | 200 |         4m59s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:40:43.090+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=118480 keep=5 new=2048
[GIN] 2025/08/22 - 12:57:22 | 200 |        16m40s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T12:57:31.670+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=352704 keep=5 new=2048
[GIN] 2025/08/22 - 13:00:16 | 200 |         2m47s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:00:27.769+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=426772 keep=5 new=2048
[GIN] 2025/08/22 - 13:02:10 | 200 |         1m46s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:02:13.349+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=7102 keep=5 new=2048
[GIN] 2025/08/22 - 13:04:35 | 200 |         2m21s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:04:46.432+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=565863 keep=5 new=2048
[GIN] 2025/08/22 - 13:05:53 | 200 |         1m13s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:06:00.709+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=222777 keep=5 new=2048
[GIN] 2025/08/22 - 13:07:54 | 200 |         1m56s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:07:59.576+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=149720 keep=5 new=2048
[GIN] 2025/08/22 - 13:09:28 | 200 |         1m30s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:09:35.551+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=237603 keep=5 new=2048
[GIN] 2025/08/22 - 13:11:59 | 200 |         2m25s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:12:05.558+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=47389 keep=5 new=2048
[GIN] 2025/08/22 - 13:13:22 | 200 |         1m18s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:13:25.548+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=72790 keep=5 new=2048
[GIN] 2025/08/22 - 13:16:00 | 200 |         2m35s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:16:04.224+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=175569 keep=5 new=2048
[GIN] 2025/08/22 - 13:17:34 | 200 |         1m31s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:17:41.652+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=212749 keep=5 new=2048
[GIN] 2025/08/22 - 13:19:21 | 200 |         1m42s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:19:24.183+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=59852 keep=5 new=2048
[GIN] 2025/08/22 - 13:20:39 | 200 |         1m15s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:20:42.266+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=21330 keep=5 new=2048
[GIN] 2025/08/22 - 13:21:45 | 200 |          1m3s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:21:52.388+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=229154 keep=5 new=2048
[GIN] 2025/08/22 - 13:23:45 | 200 |         1m54s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:23:49.820+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=257627 keep=5 new=2048
[GIN] 2025/08/22 - 13:25:29 | 200 |         1m40s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:25:37.277+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=465408 keep=5 new=2048
[GIN] 2025/08/22 - 13:27:40 | 200 |          2m5s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:27:44.080+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=76639 keep=5 new=2048
[GIN] 2025/08/22 - 13:29:38 | 200 |         1m54s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:29:43.518+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=210385 keep=5 new=2048
[GIN] 2025/08/22 - 13:31:50 | 200 |          2m7s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:31:52.214+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=3626 keep=5 new=2048
[GIN] 2025/08/22 - 13:34:26 | 200 |         2m34s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:34:37.531+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=776574 keep=5 new=2048
[GIN] 2025/08/22 - 13:36:33 | 200 |          2m1s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:36:38.756+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=151364 keep=5 new=2048
[GIN] 2025/08/22 - 13:38:34 | 200 |         1m56s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:38:36.919+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=27445 keep=5 new=2048
[GIN] 2025/08/22 - 13:41:10 | 200 |         2m33s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:41:12.887+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=6225 keep=5 new=2048
[GIN] 2025/08/22 - 13:42:41 | 200 |         1m28s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:42:43.739+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=5715 keep=5 new=2048
[GIN] 2025/08/22 - 13:44:05 | 200 |         1m21s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:44:08.716+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=50375 keep=5 new=2048
[GIN] 2025/08/22 - 13:47:11 | 200 |          3m2s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:47:13.379+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=4569 keep=5 new=2048
[GIN] 2025/08/22 - 13:51:43 | 200 |         4m30s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:51:48.360+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=162379 keep=5 new=2048
[GIN] 2025/08/22 - 13:53:48 | 200 |          2m1s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:53:51.178+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=35442 keep=5 new=2048
[GIN] 2025/08/22 - 13:55:33 | 200 |         1m42s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:55:37.377+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=132576 keep=5 new=2048
[GIN] 2025/08/22 - 13:56:29 | 200 |   52.7747411s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:56:33.027+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=108172 keep=5 new=2048
[GIN] 2025/08/22 - 13:58:54 | 200 |         2m22s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T13:58:57.226+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=29834 keep=5 new=2048
[GIN] 2025/08/22 - 14:00:31 | 200 |         1m34s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:00:33.519+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=10208 keep=5 new=2048
[GIN] 2025/08/22 - 14:02:01 | 200 |         1m27s |       127.0.0.1 | POST     "/api/generate"
[GIN] 2025/08/22 - 14:02:47 | 200 |   44.0822955s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:02:51.068+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=118261 keep=5 new=2048
[GIN] 2025/08/22 - 14:04:41 | 200 |         1m50s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:04:44.977+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=70038 keep=5 new=2048
[GIN] 2025/08/22 - 14:06:21 | 200 |         1m37s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:06:24.447+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=52118 keep=5 new=2048
[GIN] 2025/08/22 - 14:07:32 | 200 |          1m8s |       127.0.0.1 | POST     "/api/generate"
[GIN] 2025/08/22 - 14:08:17 | 200 |   42.3618831s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:08:20.794+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=79310 keep=5 new=2048
[GIN] 2025/08/22 - 14:09:28 | 200 |          1m8s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:09:34.536+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=378000 keep=5 new=2048
[GIN] 2025/08/22 - 14:11:03 | 200 |         1m31s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:11:05.635+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=10113 keep=5 new=2048
[GIN] 2025/08/22 - 14:12:42 | 200 |         1m36s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:12:47.721+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=322649 keep=5 new=2048
[GIN] 2025/08/22 - 14:15:47 | 200 |          3m1s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:15:50.556+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=67213 keep=5 new=2048
[GIN] 2025/08/22 - 14:17:48 | 200 |         1m58s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:17:52.700+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=45603 keep=5 new=2048
[GIN] 2025/08/22 - 14:20:09 | 200 |         2m17s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:20:12.765+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=28980 keep=5 new=2048
[GIN] 2025/08/22 - 14:21:47 | 200 |         1m34s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:21:52.868+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=495987 keep=5 new=2048
[GIN] 2025/08/22 - 14:23:39 | 200 |         1m48s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:23:43.526+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=57420 keep=5 new=2048
[GIN] 2025/08/22 - 14:26:28 | 200 |         2m45s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:26:30.780+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=29635 keep=5 new=2048
[GIN] 2025/08/22 - 14:29:08 | 200 |         2m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:29:12.063+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=72425 keep=5 new=2048
[GIN] 2025/08/22 - 14:31:31 | 200 |         2m20s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:31:38.546+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=225982 keep=5 new=2048
[GIN] 2025/08/22 - 14:33:58 | 200 |         2m21s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:34:02.589+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=51405 keep=5 new=2048
[GIN] 2025/08/22 - 14:35:38 | 200 |         1m36s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:35:56.296+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=979750 keep=5 new=2048
[GIN] 2025/08/22 - 14:37:47 | 200 |          2m2s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:37:51.072+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=18227 keep=5 new=2048
[GIN] 2025/08/22 - 14:39:17 | 200 |         1m26s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:39:20.903+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=127714 keep=5 new=2048
[GIN] 2025/08/22 - 14:40:43 | 200 |         1m23s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:40:45.797+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=12216 keep=5 new=2048
[GIN] 2025/08/22 - 14:41:36 | 200 |   51.1818033s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:41:43.305+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=260849 keep=5 new=2048
[GIN] 2025/08/22 - 14:42:41 | 200 |   59.5459199s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:42:43.466+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=18936 keep=5 new=2048
[GIN] 2025/08/22 - 14:45:11 | 200 |         2m27s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:45:15.126+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=193553 keep=5 new=2048
[GIN] 2025/08/22 - 14:46:49 | 200 |         1m35s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:46:51.861+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=17473 keep=5 new=2048
[GIN] 2025/08/22 - 14:48:36 | 200 |         1m45s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:48:41.543+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=147274 keep=5 new=2048
[GIN] 2025/08/22 - 14:50:12 | 200 |         1m32s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:50:16.926+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=139589 keep=5 new=2048
[GIN] 2025/08/22 - 14:51:56 | 200 |         1m40s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:51:58.980+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=3073 keep=5 new=2048
[GIN] 2025/08/22 - 14:53:58 | 200 |         1m59s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:54:01.120+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=50653 keep=5 new=2048
[GIN] 2025/08/22 - 14:55:25 | 200 |         1m24s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:55:34.762+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=583482 keep=5 new=2048
[GIN] 2025/08/22 - 14:57:48 | 200 |         2m17s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T14:57:50.394+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=2692 keep=5 new=2048
[GIN] 2025/08/22 - 15:00:03 | 200 |         2m12s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:00:16.193+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=817327 keep=5 new=2048
[GIN] 2025/08/22 - 15:03:07 | 200 |         2m56s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:03:10.884+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=82570 keep=5 new=2048
[GIN] 2025/08/22 - 15:04:46 | 200 |         1m36s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:04:50.881+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=116781 keep=5 new=2048
[GIN] 2025/08/22 - 15:06:34 | 200 |         1m44s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:06:37.105+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=61143 keep=5 new=2048
[GIN] 2025/08/22 - 15:08:51 | 200 |         2m15s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:08:54.678+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=28785 keep=5 new=2048
[GIN] 2025/08/22 - 15:11:23 | 200 |         2m29s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:11:28.686+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=490740 keep=5 new=2048
[GIN] 2025/08/22 - 15:14:05 | 200 |         2m39s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:14:14.553+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=254747 keep=5 new=2048
[GIN] 2025/08/22 - 15:18:17 | 200 |          4m5s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:18:20.212+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=34898 keep=5 new=2048
[GIN] 2025/08/22 - 15:21:43 | 200 |         3m24s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:21:48.338+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=340001 keep=5 new=2048
[GIN] 2025/08/22 - 15:25:29 | 200 |         3m43s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:25:33.219+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=88938 keep=5 new=2048
[GIN] 2025/08/22 - 15:29:55 | 200 |         4m23s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:29:59.822+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=140888 keep=5 new=2048
[GIN] 2025/08/22 - 15:34:44 | 200 |         4m45s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:34:48.946+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=146522 keep=5 new=2048
[GIN] 2025/08/22 - 15:40:13 | 200 |         5m25s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:40:21.478+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=254747 keep=5 new=2048
[GIN] 2025/08/22 - 15:44:26 | 200 |          4m7s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:44:40.693+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=736960 keep=5 new=2048
[GIN] 2025/08/22 - 15:55:32 | 200 |         11m1s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:55:41.534+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=280090 keep=5 new=2048
[GIN] 2025/08/22 - 15:59:43 | 200 |          4m3s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T15:59:46.798+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=79153 keep=5 new=2048
[GIN] 2025/08/22 - 16:05:27 | 200 |         5m41s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:05:32.243+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=175159 keep=5 new=2048
[GIN] 2025/08/22 - 16:10:09 | 200 |         4m38s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:10:14.714+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=72272 keep=5 new=2048
[GIN] 2025/08/22 - 16:13:05 | 200 |         2m51s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:13:21.076+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=492368 keep=5 new=2048
[GIN] 2025/08/22 - 16:24:25 | 200 |        11m10s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:24:32.725+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=131732 keep=5 new=2048
[GIN] 2025/08/22 - 16:28:42 | 200 |         4m11s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:28:46.751+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=86344 keep=5 new=2048
[GIN] 2025/08/22 - 16:35:48 | 200 |          7m2s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:36:00.320+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=231889 keep=5 new=2048
[GIN] 2025/08/22 - 16:43:11 | 200 |         7m15s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:43:15.326+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=73316 keep=5 new=2048
[GIN] 2025/08/22 - 16:50:08 | 200 |         6m53s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:50:14.472+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=595542 keep=5 new=2048
[GIN] 2025/08/22 - 16:53:01 | 200 |         2m49s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:53:04.380+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=60899 keep=5 new=2048
[GIN] 2025/08/22 - 16:57:17 | 200 |         4m13s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T16:57:19.936+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=33489 keep=5 new=2048
[GIN] 2025/08/22 - 17:01:07 | 200 |         3m48s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T17:01:11.157+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=53361 keep=5 new=2048
[GIN] 2025/08/22 - 17:02:59 | 200 |         1m48s |       127.0.0.1 | POST     "/api/generate"
time=2025-08-22T17:03:06.864+02:00 level=WARN source=runner.go:131 msg="truncating input prompt" limit=2048 prompt=439362 keep=5 new=2048
[GIN] 2025/08/22 - 17:06:34 | 200 |         3m30s |       127.0.0.1 | POST     "/api/generate" 
"""

# --- Extract all time strings after status code ---
time_strings = re.findall(r'\|\s*(\d+m\d+s|\d+\.\d+s|\d+s)\s*\|', log_text)

# --- Convert times to seconds ---
times_in_seconds = []
for t in time_strings:
    if 'm' in t:  # format like '3m42s'
        m, s = re.match(r'(\d+)m(\d+)s', t).groups()
        sec = int(m) * 60 + int(s)
    else:  # format like '34.81s' or '45s'
        sec = float(t.replace('s', ''))
    times_in_seconds.append(sec)

# --- Compute average time ---
average_time = np.mean(times_in_seconds)
print(f"Average time: {average_time:.2f} seconds")

# --- Plot histogram ---
plt.figure(figsize=(8, 5))
plt.hist(times_in_seconds, bins=20, edgecolor=palette['dark_gray'], color=palette['primary_red'], alpha=0.7)
plt.title("Distribution of API Response Times")
plt.xlabel("Response Time (seconds)")
plt.ylabel("Frequency")
plt.axvline(average_time, color=palette['soft_gray'], linestyle='dashed', linewidth=2, label=f'Avg: {average_time:.2f}s')
plt.legend()
plt.show()
