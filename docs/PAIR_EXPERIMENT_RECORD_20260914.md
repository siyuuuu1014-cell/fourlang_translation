# 六语言对专用模型实验归档

归档时间（UTC）：20260914T053818Z。服务器：/root/autodl-tmp/fourlang_translation。

本归档保留实验原始证据、实际代码快照、数据版本与存放位置。未删除或改写原文件；四语集合模型不在清理范围。

## 阅读与复现

evidence/ 保存代码、配置、训练报告、指标和小型数据文件的原样副本；file_inventory.json 记录所有扫描文件的位置、大小和数据 SHA256。权重及大型文件仍在原位置，清单不等于权重备份。

omissions.json 列出疑似密钥等未复制内容；changed_during_read=true 的文件表示采集期间仍在写入，不能作为冻结数据。DeepSeek 全量任务属于后续实验。

验证集用于训练选模；FLORES dev 用于消融比较；FLORES devtest 属最终评测。不得把这些指标混在同一排名。不同 tokenizer 的 BLEU、chrF2 不直接互相换算。

## 实验经过与选型依据

1. 六个语言对分别建立 SMaLL-100 双向专用模型。Exp1 使用人工平行数据，Exp2 在 Exp1 上引入教师蒸馏数据及人工数据回放。EN-UZ 复用既有冻结模型，EN-RU 复用原 Exp1 后训练 Exp2。

2. ZH-UZ 进一步比较双向全量、两个单向模型、60/40 权重、较低学习率与 FLORES-like 增量数据。以下原始报告保留各变体实际指标。

3. FLORES-like 数据经历源文本审核、Wikipedia 补充、教师翻译与复审，再组成每方向 8000 条增量数据和旧数据回放。2 epoch 与 3 epoch 分别训练；ep3 经 dev 选择后进行 devtest 比较。

4. DeepSeek 先完成 400 条 pilot，再规划每方向 9000 条全量生成。它尚不替代当前已训练模型。规则校验不代表完整语义正确率。

历史操作中出现源语西里尔字母归一化失败、源配额不足、MINOR 复审容量不足、模型路径错误、进程 Killed、CUDA 版本不兼容与 API 连接中断。经过来自会话记录；不能仅凭 Killed 推断已证实的 OOM 根因。以留存报告/日志为证据，缺失历史日志无法补造。

## 训练与评估报告索引

| 报告 | 角色 |
|---|---|
| [reports/experiments/six_pair_baseline_v1/summary.json](evidence/reports/experiments/six_pair_baseline_v1/summary.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v1/flores_profile.json](evidence/reports/experiments/zh_uz_flores_like_v1/flores_profile.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v1/source_calibration_capacity.json](evidence/reports/experiments/zh_uz_flores_like_v1/source_calibration_capacity.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v1/source_selection.json](evidence/reports/experiments/zh_uz_flores_like_v1/source_selection.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v1/source_staging.json](evidence/reports/experiments/zh_uz_flores_like_v1/source_staging.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v1/validation.json](evidence/reports/experiments/zh_uz_flores_like_v1/validation.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v1/wikipedia_extension.json](evidence/reports/experiments/zh_uz_flores_like_v1/wikipedia_extension.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v2/flores_profile.json](evidence/reports/experiments/zh_uz_flores_like_v2/flores_profile.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v2/source_selection.json](evidence/reports/experiments/zh_uz_flores_like_v2/source_selection.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v2/source_staging.json](evidence/reports/experiments/zh_uz_flores_like_v2/source_staging.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v2/teacher_calibration_capacity.json](evidence/reports/experiments/zh_uz_flores_like_v2/teacher_calibration_capacity.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v2/teacher_calibration_diagnostics.json](evidence/reports/experiments/zh_uz_flores_like_v2/teacher_calibration_diagnostics.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_like_v2/validation.json](evidence/reports/experiments/zh_uz_flores_like_v2/validation.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_relaxed_8k/assembly.json](evidence/reports/experiments/zh_uz_flores_relaxed_8k/assembly.json) | 原始报告 |
| [reports/experiments/zh_uz_flores_relaxed_8k/validation.json](evidence/reports/experiments/zh_uz_flores_relaxed_8k/validation.json) | 原始报告 |
| [results/evaluation/en_ru/exp1/metrics.json](evidence/results/evaluation/en_ru/exp1/metrics.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_ru/exp1.json](evidence/results/evaluation/pair_specialists/en_ru/exp1.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_ru/exp2.json](evidence/results/evaluation/pair_specialists/en_ru/exp2.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_ru/promotion_gate.json](evidence/results/evaluation/pair_specialists/en_ru/promotion_gate.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_uz/exp1.json](evidence/results/evaluation/pair_specialists/en_uz/exp1.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_uz/exp2.json](evidence/results/evaluation/pair_specialists/en_uz/exp2.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_uz/promotion_gate.json](evidence/results/evaluation/pair_specialists/en_uz/promotion_gate.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_zh/exp1.json](evidence/results/evaluation/pair_specialists/en_zh/exp1.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_zh/exp2.json](evidence/results/evaluation/pair_specialists/en_zh/exp2.json) | 原始报告 |
| [results/evaluation/pair_specialists/en_zh/promotion_gate.json](evidence/results/evaluation/pair_specialists/en_zh/promotion_gate.json) | 原始报告 |
| [results/evaluation/pair_specialists/uz_ru/exp1.json](evidence/results/evaluation/pair_specialists/uz_ru/exp1.json) | 原始报告 |
| [results/evaluation/pair_specialists/uz_ru/exp2.json](evidence/results/evaluation/pair_specialists/uz_ru/exp2.json) | 原始报告 |
| [results/evaluation/pair_specialists/uz_ru/promotion_gate.json](evidence/results/evaluation/pair_specialists/uz_ru/promotion_gate.json) | 原始报告 |
| [results/evaluation/pair_specialists/zh_ru/exp1.json](evidence/results/evaluation/pair_specialists/zh_ru/exp1.json) | 原始报告 |
| [results/evaluation/pair_specialists/zh_ru/exp2.json](evidence/results/evaluation/pair_specialists/zh_ru/exp2.json) | 原始报告 |
| [results/evaluation/pair_specialists/zh_ru/promotion_gate.json](evidence/results/evaluation/pair_specialists/zh_ru/promotion_gate.json) | 原始报告 |
| [results/evaluation/pair_specialists/zh_uz/exp1.json](evidence/results/evaluation/pair_specialists/zh_uz/exp1.json) | 原始报告 |
| [results/evaluation/pair_specialists/zh_uz/exp2.json](evidence/results/evaluation/pair_specialists/zh_uz/exp2.json) | 原始报告 |
| [results/evaluation/pair_specialists/zh_uz/promotion_gate.json](evidence/results/evaluation/pair_specialists/zh_uz/promotion_gate.json) | 原始报告 |
| [results/evaluation/weak_pair_ablation/zh_uz/baseline_exp1.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/baseline_exp1.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/baseline_exp2.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/baseline_exp2.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/bidir_full.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/bidir_full.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/comparison.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/comparison.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/directional_full__uz_zh.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/directional_full__uz_zh.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/directional_full__zh_uz.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/directional_full__zh_uz.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/final_devtest.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/final_devtest.json) | flores_devtest |
| [results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/full_native_lr2e6.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/full_native_lr2e6.json) | flores_dev |
| [results/evaluation/weak_pair_ablation/zh_uz/full_weighted_60_40.json](evidence/results/evaluation/weak_pair_ablation/zh_uz/full_weighted_60_40.json) | flores_dev |
| [results/experiments/weak_pair_ablation/zh_uz/bidir_full/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/bidir_full/train_report.json) | 训练报告 |
| [results/experiments/weak_pair_ablation/zh_uz/directional_full__uz_zh/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/directional_full__uz_zh/train_report.json) | 训练报告 |
| [results/experiments/weak_pair_ablation/zh_uz/directional_full__zh_uz/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/directional_full__zh_uz/train_report.json) | 训练报告 |
| [results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/train_report.json) | 训练报告 |
| [results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/train_report.json) | 训练报告 |
| [results/experiments/weak_pair_ablation/zh_uz/full_native_lr2e6/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/full_native_lr2e6/train_report.json) | 训练报告 |
| [results/experiments/weak_pair_ablation/zh_uz/full_weighted_60_40/train_report.json](evidence/results/experiments/weak_pair_ablation/zh_uz/full_weighted_60_40/train_report.json) | 训练报告 |
| [results/student/en_ru/exp1/train_report.json](evidence/results/student/en_ru/exp1/train_report.json) | 训练报告 |
| [results/student/pair_specialists/en_ru/exp2/train_report.json](evidence/results/student/pair_specialists/en_ru/exp2/train_report.json) | 训练报告 |
| [results/student/pair_specialists/en_zh/exp1/train_report.json](evidence/results/student/pair_specialists/en_zh/exp1/train_report.json) | 训练报告 |
| [results/student/pair_specialists/en_zh/exp2/train_report.json](evidence/results/student/pair_specialists/en_zh/exp2/train_report.json) | 训练报告 |
| [results/student/pair_specialists/uz_ru/exp1/train_report.json](evidence/results/student/pair_specialists/uz_ru/exp1/train_report.json) | 训练报告 |
| [results/student/pair_specialists/uz_ru/exp2/train_report.json](evidence/results/student/pair_specialists/uz_ru/exp2/train_report.json) | 训练报告 |
| [results/student/pair_specialists/zh_ru/exp1/train_report.json](evidence/results/student/pair_specialists/zh_ru/exp1/train_report.json) | 训练报告 |
| [results/student/pair_specialists/zh_ru/exp2/train_report.json](evidence/results/student/pair_specialists/zh_ru/exp2/train_report.json) | 训练报告 |
| [results/student/pair_specialists/zh_uz/exp1/train_report.json](evidence/results/student/pair_specialists/zh_uz/exp1/train_report.json) | 训练报告 |
| [results/student/pair_specialists/zh_uz/exp2/train_report.json](evidence/results/student/pair_specialists/zh_uz/exp2/train_report.json) | 训练报告 |

## 原始数值与参数（完整摘录）

### reports/experiments/six_pair_baseline_v1/summary.json

```json
{
  "schema_version": 1,
  "experiment": "six_pair_specialists",
  "benchmark_role": "final_devtest_gate",
  "samples_per_direction": 1012,
  "pass_pairs": 5,
  "fail_pairs": 1,
  "pairs": {
    "en_zh": {
      "gate": "PASS",
      "selected_baseline_stage": "exp2",
      "directions": {
        "en-zh": {
          "exp1": {
            "bleu": 32.58890424635219,
            "chrf2": 21.443099503614782,
            "samples": 1012
          },
          "exp2": {
            "bleu": 33.01686019508286,
            "chrf2": 21.945770718616156,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.4279559487306699,
            "chrf2": 0.5026712150013743
          },
          "passed": true
        },
        "zh-en": {
          "exp1": {
            "bleu": 19.21926776166106,
            "chrf2": 47.37036311762749,
            "samples": 1012
          },
          "exp2": {
            "bleu": 20.190539071084686,
            "chrf2": 48.140038358634264,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.9712713094236243,
            "chrf2": 0.7696752410067731
          },
          "passed": true
        }
      }
    },
    "en_uz": {
      "gate": "PASS",
      "selected_baseline_stage": "exp2",
      "directions": {
        "en-uz": {
          "exp1": {
            "bleu": 15.009204124235044,
            "chrf2": 47.57774178959463,
            "samples": 1012
          },
          "exp2": {
            "bleu": 15.323007108177192,
            "chrf2": 47.87027032901657,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.3138029839421481,
            "chrf2": 0.29252853942193724
          },
          "passed": true
        },
        "uz-en": {
          "exp1": {
            "bleu": 23.93138846424699,
            "chrf2": 50.3374970460535,
            "samples": 1012
          },
          "exp2": {
            "bleu": 24.47775531337278,
            "chrf2": 50.57860459714267,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.5463668491257891,
            "chrf2": 0.24110755108917203
          },
          "passed": true
        }
      }
    },
    "en_ru": {
      "gate": "PASS",
      "selected_baseline_stage": "exp2",
      "directions": {
        "en-ru": {
          "exp1": {
            "bleu": 23.63215965465385,
            "chrf2": 49.2916981012424,
            "samples": 1012
          },
          "exp2": {
            "bleu": 23.902043762374994,
            "chrf2": 49.629503918334535,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.26988410772114335,
            "chrf2": 0.3378058170921321
          },
          "passed": true
        },
        "ru-en": {
          "exp1": {
            "bleu": 25.811736293324255,
            "chrf2": 51.911983829455,
            "samples": 1012
          },
          "exp2": {
            "bleu": 26.71584930105291,
            "chrf2": 52.73497271966179,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.9041130077286539,
            "chrf2": 0.8229888902067941
          },
          "passed": true
        }
      }
    },
    "zh_uz": {
      "gate": "FAIL",
      "selected_baseline_stage": "exp1",
      "directions": {
        "zh-uz": {
          "exp1": {
            "bleu": 4.631407955500526,
            "chrf2": 31.78868138705453,
            "samples": 1012
          },
          "exp2": {
            "bleu": 4.7752320879812835,
            "chrf2": 31.925600672703375,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.14382413248075743,
            "chrf2": 0.13691928564884392
          },
          "passed": true
        },
        "uz-zh": {
          "exp1": {
            "bleu": 19.07196419983334,
            "chrf2": 13.981110740141464,
            "samples": 1012
          },
          "exp2": {
            "bleu": 19.094028563551934,
            "chrf2": 13.847286301852415,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.022064363718595104,
            "chrf2": -0.13382443828904833
          },
          "passed": false
        }
      }
    },
    "zh_ru": {
      "gate": "PASS",
      "selected_baseline_stage": "exp2",
      "directions": {
        "zh-ru": {
          "exp1": {
            "bleu": 12.521395850484703,
            "chrf2": 38.4399467927828,
            "samples": 1012
          },
          "exp2": {
            "bleu": 12.97337863773094,
            "chrf2": 39.245919014809324,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.4519827872462372,
            "chrf2": 0.8059722220265257
          },
          "passed": true
        },
        "ru-zh": {
          "exp1": {
            "bleu": 26.508608749363663,
            "chrf2": 18.21796868503126,
            "samples": 1012
          },
          "exp2": {
            "bleu": 27.181061636712702,
            "chrf2": 18.46761207226311,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.6724528873490385,
            "chrf2": 0.2496433872318491
          },
          "passed": true
        }
      }
    },
    "uz_ru": {
      "gate": "PASS",
      "selected_baseline_stage": "exp2",
      "directions": {
        "uz-ru": {
          "exp1": {
            "bleu": 6.577107628549916,
            "chrf2": 27.736407369135012,
            "samples": 1012
          },
          "exp2": {
            "bleu": 7.277864576497285,
            "chrf2": 29.606147379691585,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.7007569479473688,
            "chrf2": 1.8697400105565727
          },
          "passed": true
        },
        "ru-uz": {
          "exp1": {
            "bleu": 2.4065308637156217,
            "chrf2": 22.74740659882642,
            "samples": 1012
          },
          "exp2": {
            "bleu": 2.744255131071184,
            "chrf2": 24.77226151486562,
            "samples": 1012
          },
          "delta": {
            "bleu": 0.33772426735556227,
            "chrf2": 2.0248549160391995
          },
          "passed": true
        }
      }
    }
  },
  "decision": "This closes the controlled SMaLL-100 baseline only; PASS does not select a final architecture. Architecture bake-off uses FLORES dev."
}
```

### reports/experiments/zh_uz_flores_like_v1/flores_profile.json

```json
{
  "schema_version": 1,
  "source": "/root/autodl-tmp/fourlang_translation/data/benchmark/fourlang/flores_dev.parquet",
  "rows": 997,
  "languages": {
    "zh": {
      "length_boundaries_characters": {
        "short_max": 33,
        "medium_max": 46
      },
      "feature_distribution": {
        "long|question|no_digit|multi": 1,
        "long|statement|digit|multi": 82,
        "long|statement|digit|simple": 41,
        "long|statement|no_digit|multi": 137,
        "long|statement|no_digit|simple": 68,
        "medium|question|no_digit|simple": 2,
        "medium|statement|digit|multi": 24,
        "medium|statement|digit|simple": 39,
        "medium|statement|no_digit|multi": 120,
        "medium|statement|no_digit|simple": 140,
        "short|statement|digit|multi": 3,
        "short|statement|digit|simple": 36,
        "short|statement|no_digit|multi": 36,
        "short|statement|no_digit|simple": 268
      }
    },
    "uz": {
      "length_boundaries_characters": {
        "short_max": 117,
        "medium_max": 157
      },
      "feature_distribution": {
        "long|question|no_digit|simple": 1,
        "long|statement|digit|multi": 25,
        "long|statement|digit|simple": 53,
        "long|statement|no_digit|multi": 94,
        "long|statement|no_digit|simple": 150,
        "medium|question|no_digit|simple": 2,
        "medium|statement|digit|multi": 10,
        "medium|statement|digit|simple": 66,
        "medium|statement|no_digit|multi": 52,
        "medium|statement|no_digit|simple": 209,
        "short|question|no_digit|simple": 1,
        "short|statement|digit|multi": 4,
        "short|statement|digit|simple": 62,
        "short|statement|no_digit|multi": 25,
        "short|statement|no_digit|simple": 243
      }
    }
  },
  "flores_used_as_training_data": false,
  "sha256": "e5d9565440682c8ad1fab045df0fd7f752fc8912642bf834ddfadad39cf937a4"
}
```

### reports/experiments/zh_uz_flores_like_v1/source_calibration_capacity.json

```json
{
  "schema_version": 1,
  "calibration": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/source_judge_calibration.parquet",
  "calibration_rows": 4000,
  "directions": {
    "zh-uz": {
      "target": 12000,
      "expected_usable_rows": 13376,
      "conservative_usable_rows": 12582,
      "decision": "READY"
    },
    "uz-zh": {
      "target": 12000,
      "expected_usable_rows": 14693,
      "conservative_usable_rows": 13972,
      "decision": "READY"
    }
  },
  "groups": {
    "zh-uz|fineweb": {
      "available": 6146,
      "calibration_rows": 458,
      "pass_rows": 183,
      "pass_rate": 0.39956331877729256,
      "conservative_pass_rate_95": 0.35470430244693296,
      "configured_cap_rows": 5400,
      "expected_usable_rows": 2455,
      "conservative_usable_rows": 2180
    },
    "zh-uz|hplt": {
      "available": 9603,
      "calibration_rows": 478,
      "pass_rows": 210,
      "pass_rate": 0.4393305439330544,
      "conservative_pass_rate_95": 0.39483758683880793,
      "configured_cap_rows": 7800,
      "expected_usable_rows": 4218,
      "conservative_usable_rows": 3791
    },
    "zh-uz|v2_bronze": {
      "available": 1978,
      "calibration_rows": 420,
      "pass_rows": 149,
      "pass_rate": 0.3547619047619048,
      "conservative_pass_rate_95": 0.309004654779788,
      "configured_cap_rows": 1800,
      "expected_usable_rows": 701,
      "conservative_usable_rows": 611
    },
    "zh-uz|wikipedia": {
      "available": 15273,
      "calibration_rows": 810,
      "pass_rows": 585,
      "pass_rate": 0.7222222222222222,
      "conservative_pass_rate_95": 0.6913763017078955,
      "configured_cap_rows": 6000,
      "expected_usable_rows": 6000,
      "conservative_usable_rows": 6000
    },
    "uz-zh|fineweb": {
      "available": 5587,
      "calibration_rows": 444,
      "pass_rows": 226,
      "pass_rate": 0.509009009009009,
      "conservative_pass_rate_95": 0.4625078173376071,
      "configured_cap_rows": 5400,
      "expected_usable_rows": 2843,
      "conservative_usable_rows": 2584
    },
    "uz-zh|hplt": {
      "available": 8760,
      "calibration_rows": 490,
      "pass_rows": 284,
      "pass_rate": 0.5795918367346938,
      "conservative_pass_rate_95": 0.5358844609969816,
      "configured_cap_rows": 7800,
      "expected_usable_rows": 5077,
      "conservative_usable_rows": 4694
    },
    "uz-zh|v2_bronze": {
      "available": 1627,
      "calibration_rows": 413,
      "pass_rows": 196,
      "pass_rate": 0.4745762711864407,
      "conservative_pass_rate_95": 0.42641600302272625,
      "configured_cap_rows": 1800,
      "expected_usable_rows": 772,
      "conservative_usable_rows": 693
    },
    "uz-zh|wikipedia": {
      "available": 8026,
      "calibration_rows": 487,
      "pass_rows": 403,
      "pass_rate": 0.8275154004106776,
      "conservative_pass_rate_95": 0.7939605895131902,
      "configured_cap_rows": 6000,
      "expected_usable_rows": 6000,
      "conservative_usable_rows": 6000
    }
  },
  "full_source_audit_recommended": true
}
```

### reports/experiments/zh_uz_flores_like_v1/source_selection.json

```json
{
  "schema_version": 2,
  "stage": "final_source_selection",
  "candidate_pool": [
    "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/monolingual_candidates.jsonl"
  ],
  "by_direction": {
    "zh-uz": {
      "eligible": 19726,
      "selected": 12000,
      "target": 12000,
      "by_source_group": {
        "fineweb": 2077,
        "hplt": 3325,
        "v2_bronze": 598,
        "wikipedia": 6000
      },
      "by_feature": {
        "long|question|digit|multi": 13,
        "long|question|digit|simple": 3,
        "long|question|no_digit|multi": 34,
        "long|question|no_digit|simple": 1,
        "long|statement|digit|multi": 1393,
        "long|statement|digit|simple": 310,
        "long|statement|no_digit|multi": 2092,
        "long|statement|no_digit|simple": 606,
        "medium|question|digit|multi": 2,
        "medium|question|digit|simple": 2,
        "medium|question|no_digit|multi": 14,
        "medium|question|no_digit|simple": 11,
        "medium|statement|digit|multi": 424,
        "medium|statement|digit|simple": 384,
        "medium|statement|no_digit|multi": 1026,
        "medium|statement|no_digit|simple": 972,
        "short|question|digit|multi": 1,
        "short|question|digit|simple": 9,
        "short|question|no_digit|multi": 17,
        "short|question|no_digit|simple": 91,
        "short|statement|digit|multi": 147,
        "short|statement|digit|simple": 648,
        "short|statement|no_digit|multi": 574,
        "short|statement|no_digit|simple": 3226
      },
      "rejections": {}
    },
    "uz-zh": {
      "eligible": 15313,
      "selected": 12000,
      "target": 12000,
      "by_source_group": {
        "fineweb": 1961,
        "hplt": 3911,
        "v2_bronze": 575,
        "wikipedia": 5553
      },
      "by_feature": {
        "long|question|digit|multi": 1,
        "long|question|digit|simple": 2,
        "long|question|no_digit|multi": 9,
        "long|question|no_digit|simple": 1,
        "long|statement|digit|multi": 548,
        "long|statement|digit|simple": 524,
        "long|statement|no_digit|multi": 1230,
        "long|statement|no_digit|simple": 1151,
        "medium|question|digit|multi": 1,
        "medium|question|no_digit|multi": 2,
        "medium|question|no_digit|simple": 5,
        "medium|statement|digit|multi": 228,
        "medium|statement|digit|simple": 635,
        "medium|statement|no_digit|multi": 645,
        "medium|statement|no_digit|simple": 1881,
        "short|question|digit|simple": 8,
        "short|question|no_digit|multi": 6,
        "short|question|no_digit|simple": 78,
        "short|statement|digit|multi": 151,
        "short|statement|digit|simple": 1022,
        "short|statement|no_digit|multi": 549,
        "short|statement|no_digit|simple": 3323
      },
      "rejections": {}
    }
  },
  "flores_rows_selected": 0,
  "output": "/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/flores_like_v1/selected_sources.jsonl",
  "teacher_input": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/kd_candidates.jsonl",
  "inputs_sha256": {
    "candidate_pool": "ef0322f71f2629758dd89727c88159326de115da3d864b5fc367c17389aff6f4",
    "source_review": "cf00c948d5a10b359ebd61ce387c494e8d8b9178e3c32933e056b93960bddccf",
    "previous_selected": "7b9c6aa4fe9f75b8b7048fedb105c6d5c6f4a88c8b17efe973fb2b99fa336591",
    "existing_train": "fb98038b0e462f4dec03254e4fa1431a4d8a87b945d738cb5d428a1b55e7afbc",
    "validation": "20906dbe010fabefa92cd54d365dfde557b7858c227502e9ceb83b39fdde5228",
    "flores_dev": "e5d9565440682c8ad1fab045df0fd7f752fc8912642bf834ddfadad39cf937a4",
    "flores_devtest": "933a9c29300d65ca6c1b85345dce5922dfe78c4797c5b306c0481f33d25db41e"
  },
  "shortages": {}
}
```

### reports/experiments/zh_uz_flores_like_v1/source_staging.json

```json
{
  "schema_version": 2,
  "stage": "source_review_staging",
  "candidate_pool": [
    "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_v3/monolingual_collected.parquet",
    "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/wikipedia_extension.parquet"
  ],
  "by_direction": {
    "zh-uz": {
      "eligible": 33000,
      "selected": 33000,
      "target": 33000,
      "by_source_group": {
        "fineweb": 6146,
        "hplt": 9603,
        "v2_bronze": 1978,
        "wikipedia": 15273
      },
      "by_feature": {
        "long|question|digit|multi": 48,
        "long|question|digit|simple": 7,
        "long|question|no_digit|multi": 101,
        "long|question|no_digit|simple": 11,
        "long|statement|digit|multi": 3561,
        "long|statement|digit|simple": 687,
        "long|statement|no_digit|multi": 4859,
        "long|statement|no_digit|simple": 1298,
        "medium|question|digit|multi": 5,
        "medium|question|digit|simple": 16,
        "medium|question|no_digit|multi": 39,
        "medium|question|no_digit|simple": 34,
        "medium|statement|digit|multi": 1218,
        "medium|statement|digit|simple": 843,
        "medium|statement|no_digit|multi": 2181,
        "medium|statement|no_digit|simple": 2119,
        "short|question|digit|multi": 12,
        "short|question|digit|simple": 44,
        "short|question|no_digit|multi": 61,
        "short|question|no_digit|simple": 562,
        "short|statement|digit|multi": 870,
        "short|statement|digit|simple": 2342,
        "short|statement|no_digit|multi": 2010,
        "short|statement|no_digit|simple": 10072
      },
      "rejections": {
        "previous_or_validation_exact": 20000
      }
    },
    "uz-zh": {
      "eligible": 24000,
      "selected": 24000,
      "target": 24000,
      "by_source_group": {
        "fineweb": 5587,
        "hplt": 8760,
        "v2_bronze": 1627,
        "wikipedia": 8026
      },
      "by_feature": {
        "long|question|digit|multi": 3,
        "long|question|digit|simple": 3,
        "long|question|no_digit|multi": 18,
        "long|question|no_digit|simple": 5,
        "long|statement|digit|multi": 1033,
        "long|statement|digit|simple": 721,
        "long|statement|no_digit|multi": 2342,
        "long|statement|no_digit|simple": 1596,
        "medium|question|digit|multi": 1,
        "medium|question|no_digit|multi": 7,
        "medium|question|no_digit|simple": 10,
        "medium|statement|digit|multi": 398,
        "medium|statement|digit|simple": 854,
        "medium|statement|no_digit|multi": 1275,
        "medium|statement|no_digit|simple": 2442,
        "short|question|digit|simple": 21,
        "short|question|no_digit|multi": 30,
        "short|question|no_digit|simple": 222,
        "short|statement|digit|multi": 427,
        "short|statement|digit|simple": 2336,
        "short|statement|no_digit|multi": 1400,
        "short|statement|no_digit|simple": 8856
      },
      "rejections": {
        "previous_or_validation_exact": 17000
      }
    }
  },
  "flores_rows_selected": 0,
  "source_review_input": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/monolingual_candidates.jsonl",
  "inputs_sha256": {
    "candidate_pool": "ef0322f71f2629758dd89727c88159326de115da3d864b5fc367c17389aff6f4",
    "source_review": "cf00c948d5a10b359ebd61ce387c494e8d8b9178e3c32933e056b93960bddccf",
    "previous_selected": "7b9c6aa4fe9f75b8b7048fedb105c6d5c6f4a88c8b17efe973fb2b99fa336591",
    "existing_train": "fb98038b0e462f4dec03254e4fa1431a4d8a87b945d738cb5d428a1b55e7afbc",
    "validation": "20906dbe010fabefa92cd54d365dfde557b7858c227502e9ceb83b39fdde5228",
    "flores_dev": "e5d9565440682c8ad1fab045df0fd7f752fc8912642bf834ddfadad39cf937a4",
    "flores_devtest": "933a9c29300d65ca6c1b85345dce5922dfe78c4797c5b306c0481f33d25db41e"
  },
  "shortages": {}
}
```

### reports/experiments/zh_uz_flores_like_v1/validation.json

```json
{
  "schema_version": 1,
  "status": "PASS",
  "errors": []
}
```

### reports/experiments/zh_uz_flores_like_v1/wikipedia_extension.json

```json
{
  "schema_version": 1,
  "status": "PASS",
  "output": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/wikipedia_extension.parquet",
  "rows": 23000,
  "by_language": {
    "uz": 8000,
    "zh": 15000
  },
  "targets": {
    "zh": 15000,
    "uz": 8000
  },
  "shortages": {},
  "source_state": {
    "wikipedia_flores_like_zh": {
      "status": "completed",
      "documents_seen": 92,
      "accepted": 15000
    },
    "wikipedia_flores_like_uz": {
      "status": "completed",
      "documents_seen": 100,
      "accepted": 8000
    }
  },
  "rejections": {
    "wikipedia_flores_like_uz:CITATION_DEBRIS": 1,
    "wikipedia_flores_like_uz:DUPLICATE_OR_PREVIOUS_SOURCE": 95,
    "wikipedia_flores_like_uz:TOO_LONG": 39,
    "wikipedia_flores_like_uz:TOO_MANY_DIGITS": 24,
    "wikipedia_flores_like_uz:TOO_SHORT": 1168,
    "wikipedia_flores_like_uz:URL_EMAIL_OR_MARKUP": 8,
    "wikipedia_flores_like_uz:UZ_LANGUAGE_SIGNAL": 1445
  },
  "output_sha256": "edff7ac1becb991ee10d7f5ed759d18363234a5def23860c0d30d2c200e95a02",
  "base_candidate_pool_modified": false
}
```

### reports/experiments/zh_uz_flores_like_v2/flores_profile.json

```json
{
  "schema_version": 1,
  "source": "/root/autodl-tmp/fourlang_translation/data/benchmark/fourlang/flores_dev.parquet",
  "rows": 997,
  "languages": {
    "zh": {
      "length_boundaries_characters": {
        "short_max": 33,
        "medium_max": 46
      },
      "feature_distribution": {
        "long|question|no_digit|multi": 1,
        "long|statement|digit|multi": 82,
        "long|statement|digit|simple": 41,
        "long|statement|no_digit|multi": 137,
        "long|statement|no_digit|simple": 68,
        "medium|question|no_digit|simple": 2,
        "medium|statement|digit|multi": 24,
        "medium|statement|digit|simple": 39,
        "medium|statement|no_digit|multi": 120,
        "medium|statement|no_digit|simple": 140,
        "short|statement|digit|multi": 3,
        "short|statement|digit|simple": 36,
        "short|statement|no_digit|multi": 36,
        "short|statement|no_digit|simple": 268
      }
    },
    "uz": {
      "length_boundaries_characters": {
        "short_max": 117,
        "medium_max": 157
      },
      "feature_distribution": {
        "long|question|no_digit|simple": 1,
        "long|statement|digit|multi": 25,
        "long|statement|digit|simple": 53,
        "long|statement|no_digit|multi": 94,
        "long|statement|no_digit|simple": 150,
        "medium|question|no_digit|simple": 2,
        "medium|statement|digit|multi": 10,
        "medium|statement|digit|simple": 66,
        "medium|statement|no_digit|multi": 52,
        "medium|statement|no_digit|simple": 209,
        "short|question|no_digit|simple": 1,
        "short|statement|digit|multi": 4,
        "short|statement|digit|simple": 62,
        "short|statement|no_digit|multi": 25,
        "short|statement|no_digit|simple": 243
      }
    }
  },
  "flores_used_as_training_data": false,
  "sha256": "e5d9565440682c8ad1fab045df0fd7f752fc8912642bf834ddfadad39cf937a4"
}
```

### reports/experiments/zh_uz_flores_like_v2/source_selection.json

```json
{
  "schema_version": 2,
  "stage": "final_source_selection",
  "candidate_pool": [
    "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/monolingual_candidates.jsonl"
  ],
  "by_direction": {
    "zh-uz": {
      "eligible": 19726,
      "selected": 14000,
      "target": 14000,
      "by_source_group": {
        "fineweb": 1810,
        "hplt": 2950,
        "v2_bronze": 140,
        "wikipedia": 9100
      },
      "by_feature": {
        "long|question|digit|multi": 13,
        "long|question|digit|simple": 3,
        "long|question|no_digit|multi": 28,
        "long|question|no_digit|simple": 1,
        "long|statement|digit|multi": 1434,
        "long|statement|digit|simple": 407,
        "long|statement|no_digit|multi": 2165,
        "long|statement|no_digit|simple": 820,
        "medium|question|digit|multi": 2,
        "medium|question|digit|simple": 2,
        "medium|question|no_digit|multi": 13,
        "medium|question|no_digit|simple": 13,
        "medium|statement|digit|multi": 437,
        "medium|statement|digit|simple": 528,
        "medium|statement|no_digit|multi": 1400,
        "medium|statement|no_digit|simple": 1433,
        "short|question|digit|multi": 1,
        "short|question|digit|simple": 8,
        "short|question|no_digit|multi": 17,
        "short|question|no_digit|simple": 83,
        "short|statement|digit|multi": 145,
        "short|statement|digit|simple": 660,
        "short|statement|no_digit|multi": 605,
        "short|statement|no_digit|simple": 3782
      },
      "rejections": {}
    },
    "uz-zh": {
      "eligible": 15313,
      "selected": 14000,
      "target": 14000,
      "by_source_group": {
        "fineweb": 2606,
        "hplt": 4887,
        "v2_bronze": 140,
        "wikipedia": 6367
      },
      "by_feature": {
        "long|question|digit|multi": 1,
        "long|question|digit|simple": 1,
        "long|question|no_digit|multi": 6,
        "long|question|no_digit|simple": 1,
        "long|statement|digit|multi": 631,
        "long|statement|digit|simple": 475,
        "long|statement|no_digit|multi": 1448,
        "long|statement|no_digit|simple": 1078,
        "medium|question|digit|multi": 1,
        "medium|question|no_digit|multi": 1,
        "medium|question|no_digit|simple": 5,
        "medium|statement|digit|multi": 252,
        "medium|statement|digit|simple": 608,
        "medium|statement|no_digit|multi": 829,
        "medium|statement|no_digit|simple": 1834,
        "short|question|digit|simple": 10,
        "short|question|no_digit|multi": 11,
        "short|question|no_digit|simple": 98,
        "short|statement|digit|multi": 188,
        "short|statement|digit|simple": 1294,
        "short|statement|no_digit|multi": 695,
        "short|statement|no_digit|simple": 4533
      },
      "rejections": {}
    }
  },
  "flores_rows_selected": 0,
  "output": "/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/flores_like_v2/selected_sources.jsonl",
  "teacher_input": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/kd_candidates.jsonl",
  "inputs_sha256": {
    "candidate_pool": "ef0322f71f2629758dd89727c88159326de115da3d864b5fc367c17389aff6f4",
    "source_review": "cf00c948d5a10b359ebd61ce387c494e8d8b9178e3c32933e056b93960bddccf",
    "previous_selected": "7b9c6aa4fe9f75b8b7048fedb105c6d5c6f4a88c8b17efe973fb2b99fa336591",
    "existing_train": "fb98038b0e462f4dec03254e4fa1431a4d8a87b945d738cb5d428a1b55e7afbc",
    "validation": "20906dbe010fabefa92cd54d365dfde557b7858c227502e9ceb83b39fdde5228",
    "flores_dev": "e5d9565440682c8ad1fab045df0fd7f752fc8912642bf834ddfadad39cf937a4",
    "flores_devtest": "933a9c29300d65ca6c1b85345dce5922dfe78c4797c5b306c0481f33d25db41e"
  },
  "shortages": {}
}
```

### reports/experiments/zh_uz_flores_like_v2/source_staging.json

```json
{
  "schema_version": 2,
  "stage": "source_review_staging",
  "candidate_pool": [
    "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_v3/monolingual_collected.parquet",
    "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v1/wikipedia_extension.parquet"
  ],
  "by_direction": {
    "zh-uz": {
      "eligible": 33000,
      "selected": 33000,
      "target": 33000,
      "by_source_group": {
        "fineweb": 6146,
        "hplt": 9603,
        "v2_bronze": 1978,
        "wikipedia": 15273
      },
      "by_feature": {
        "long|question|digit|multi": 48,
        "long|question|digit|simple": 7,
        "long|question|no_digit|multi": 101,
        "long|question|no_digit|simple": 11,
        "long|statement|digit|multi": 3561,
        "long|statement|digit|simple": 687,
        "long|statement|no_digit|multi": 4859,
        "long|statement|no_digit|simple": 1298,
        "medium|question|digit|multi": 5,
        "medium|question|digit|simple": 16,
        "medium|question|no_digit|multi": 39,
        "medium|question|no_digit|simple": 34,
        "medium|statement|digit|multi": 1218,
        "medium|statement|digit|simple": 843,
        "medium|statement|no_digit|multi": 2181,
        "medium|statement|no_digit|simple": 2119,
        "short|question|digit|multi": 12,
        "short|question|digit|simple": 44,
        "short|question|no_digit|multi": 61,
        "short|question|no_digit|simple": 562,
        "short|statement|digit|multi": 870,
        "short|statement|digit|simple": 2342,
        "short|statement|no_digit|multi": 2010,
        "short|statement|no_digit|simple": 10072
      },
      "rejections": {
        "previous_or_validation_exact": 20000
      }
    },
    "uz-zh": {
      "eligible": 24000,
      "selected": 24000,
      "target": 24000,
      "by_source_group": {
        "fineweb": 5587,
        "hplt": 8760,
        "v2_bronze": 1627,
        "wikipedia": 8026
      },
      "by_feature": {
        "long|question|digit|multi": 3,
        "long|question|digit|simple": 3,
        "long|question|no_digit|multi": 18,
        "long|question|no_digit|simple": 5,
        "long|statement|digit|multi": 1033,
        "long|statement|digit|simple": 721,
        "long|statement|no_digit|multi": 2342,
        "long|statement|no_digit|simple": 1596,
        "medium|question|digit|multi": 1,
        "medium|question|no_digit|multi": 7,
        "medium|question|no_digit|simple": 10,
        "medium|statement|digit|multi": 398,
        "medium|statement|digit|simple": 854,
        "medium|statement|no_digit|multi": 1275,
        "medium|statement|no_digit|simple": 2442,
        "short|question|digit|simple": 21,
        "short|question|no_digit|multi": 30,
        "short|question|no_digit|simple": 222,
        "short|statement|digit|multi": 427,
        "short|statement|digit|simple": 2336,
        "short|statement|no_digit|multi": 1400,
        "short|statement|no_digit|simple": 8856
      },
      "rejections": {
        "previous_or_validation_exact": 17000
      }
    }
  },
  "flores_rows_selected": 0,
  "source_review_input": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/monolingual_candidates.jsonl",
  "inputs_sha256": {
    "candidate_pool": "ef0322f71f2629758dd89727c88159326de115da3d864b5fc367c17389aff6f4",
    "source_review": "cf00c948d5a10b359ebd61ce387c494e8d8b9178e3c32933e056b93960bddccf",
    "previous_selected": "7b9c6aa4fe9f75b8b7048fedb105c6d5c6f4a88c8b17efe973fb2b99fa336591",
    "existing_train": "fb98038b0e462f4dec03254e4fa1431a4d8a87b945d738cb5d428a1b55e7afbc",
    "validation": "20906dbe010fabefa92cd54d365dfde557b7858c227502e9ceb83b39fdde5228",
    "flores_dev": "e5d9565440682c8ad1fab045df0fd7f752fc8912642bf834ddfadad39cf937a4",
    "flores_devtest": "933a9c29300d65ca6c1b85345dce5922dfe78c4797c5b306c0481f33d25db41e"
  },
  "shortages": {}
}
```

### reports/experiments/zh_uz_flores_like_v2/teacher_calibration_capacity.json

```json
{
  "schema_version": 1,
  "first_review": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/teacher_judge_calibration.parquet",
  "minor_second_review": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/teacher_minor_second_review_calibration.parquet",
  "directions": {
    "zh-uz": {
      "samples": 519,
      "first_pass": 135,
      "minor_second_pass": 30,
      "accepted_total": 165,
      "accepted_rate": 0.3179190751445087,
      "projected_accepted_rows": 4450,
      "conservative_projected_rows_95": 3910,
      "minimum_required_rows": 8000,
      "decision": "NOT_READY"
    },
    "uz-zh": {
      "samples": 481,
      "first_pass": 208,
      "minor_second_pass": 16,
      "accepted_total": 224,
      "accepted_rate": 0.4656964656964657,
      "projected_accepted_rows": 6519,
      "conservative_projected_rows_95": 5901,
      "minimum_required_rows": 8000,
      "decision": "NOT_READY"
    }
  },
  "full_teacher_audit_recommended": false
}
```

### reports/experiments/zh_uz_flores_like_v2/teacher_calibration_diagnostics.json

```json
{
  "schema_version": 1,
  "stage": "teacher_calibration_diagnostics",
  "first_review": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/teacher_judge_calibration.parquet",
  "minor_second_review": "/root/autodl-tmp/fourlang_translation/data/pipeline_v2/zh_uz_flores_like_v2/teacher_minor_second_review_calibration.parquet",
  "directions": {
    "zh-uz": {
      "samples": 519,
      "labels": {
        "FAIL": 140,
        "MINOR": 244,
        "PASS": 135
      },
      "usefulness": {
        "HIGH": 305,
        "MEDIUM": 74,
        "REJECT": 140
      },
      "error_flags": {
        "omission": 123,
        "addition": 68,
        "mistranslation": 146,
        "number_error": 10,
        "entity_error": 166,
        "negation_error": 10
      },
      "semantic_inconsistent": 140,
      "parse_failures": 0,
      "minor_rows": 244,
      "clean_minor_sent_to_second_review": 199,
      "second_review_pass": 30,
      "second_review_salvage_rate": 0.1507537688442211,
      "accepted_rate": 0.3179190751445087,
      "conservative_accepted_rate_95": 0.2793184868483656,
      "selected_candidates": 14000,
      "minimum_required_rows": 8000,
      "estimated_candidates_for_minimum": 28642,
      "additional_candidates_for_minimum": 14642,
      "by_source_corpus": {
        "fineweb2_cmn_hani": {
          "samples": 69,
          "pass": 15,
          "minor": 36,
          "fail": 18,
          "uncertain": 0,
          "strict_pass_rate": 0.21739130434782608
        },
        "hplt3_cmn_hans": {
          "samples": 96,
          "pass": 14,
          "minor": 57,
          "fail": 25,
          "uncertain": 0,
          "strict_pass_rate": 0.14583333333333334
        },
        "v2_bronze_zh": {
          "samples": 4,
          "pass": 0,
          "minor": 4,
          "fail": 0,
          "uncertain": 0,
          "strict_pass_rate": 0.0
        },
        "wikipedia_flores_like_zh": {
          "samples": 348,
          "pass": 106,
          "minor": 145,
          "fail": 97,
          "uncertain": 0,
          "strict_pass_rate": 0.3045977011494253
        },
        "wikipedia_zh": {
          "samples": 2,
          "pass": 0,
          "minor": 2,
          "fail": 0,
          "uncertain": 0,
          "strict_pass_rate": 0.0
        }
      },
      "top_non_pass_reasons": {
        "Minor stylistic issues, but overall accurate and fluent translation.": 9,
        "Minor stylistic improvement possible for fluency": 8,
        "Minor stylistic difference in phrasing, but meaning is fully preserved.": 8,
        "Minor stylistic improvement possible, but meaning is fully preserved.": 5,
        "Minor issue with word order and phrasing, but overall meaning is clear and accurate.": 5,
        "Minor stylistic issues, but meaning is fully preserved.": 4,
        "Minor stylistic or phrasing issue, but meaning is fully preserved.": 3,
        "Minor stylistic improvement in phrasing, but meaning is fully preserved.": 3,
        "Minor stylistic or phrasing issue, but meaning is clear and accurate.": 2,
        "Minor issue with phrasing, but overall meaning is clear and accurate.": 2,
        "Minor issue with date format, but overall translation is accurate and fluent.": 2,
        "Minor stylistic issue in the translation, but meaning is fully preserved.": 2,
        "Minor stylistic issues, but meaning is clear and accurate.": 2,
        "Minor stylistic issue; the translation is semantically accurate but could be slightly more natural in Uzbek.": 2,
        "Minor stylistic issue with word order, but meaning is fully preserved.": 2,
        "Minor fluency issues, but overall meaning is clear and accurate.": 2,
        "Minor stylistic variation, but meaning is fully preserved.": 2,
        "Minor lexical inaccuracies and awkward phrasing, but overall meaning is clear and usable.": 1,
        "Incorrect translation of '胡先骕' as 'Hu Zhengyu', and incorrect translation of '水杉' as 'suv daraxtini' (water tree) instead of the proper scientific name 'Metasequoia'": 1,
        "Minor stylistic improvements could be made for better fluency, but the translation is semantically accurate and fully usable.": 1
      }
    },
    "uz-zh": {
      "samples": 481,
      "labels": {
        "FAIL": 73,
        "MINOR": 200,
        "PASS": 208
      },
      "usefulness": {
        "HIGH": 346,
        "MEDIUM": 62,
        "REJECT": 73
      },
      "error_flags": {
        "omission": 64,
        "addition": 61,
        "mistranslation": 79,
        "number_error": 2,
        "entity_error": 89,
        "negation_error": 1
      },
      "semantic_inconsistent": 73,
      "parse_failures": 0,
      "minor_rows": 200,
      "clean_minor_sent_to_second_review": 160,
      "second_review_pass": 16,
      "second_review_salvage_rate": 0.1,
      "accepted_rate": 0.4656964656964657,
      "conservative_accepted_rate_95": 0.42156552733857966,
      "selected_candidates": 14000,
      "minimum_required_rows": 8000,
      "estimated_candidates_for_minimum": 18977,
      "additional_candidates_for_minimum": 4977,
      "by_source_corpus": {
        "fineweb2_uzn_latn": {
          "samples": 90,
          "pass": 39,
          "minor": 27,
          "fail": 24,
          "uncertain": 0,
          "strict_pass_rate": 0.43333333333333335
        },
        "hplt3_uzn_latn": {
          "samples": 176,
          "pass": 57,
          "minor": 90,
          "fail": 29,
          "uncertain": 0,
          "strict_pass_rate": 0.32386363636363635
        },
        "v2_bronze_uz": {
          "samples": 6,
          "pass": 3,
          "minor": 3,
          "fail": 0,
          "uncertain": 0,
          "strict_pass_rate": 0.5
        },
        "wikipedia_flores_like_uz": {
          "samples": 208,
          "pass": 109,
          "minor": 79,
          "fail": 20,
          "uncertain": 0,
          "strict_pass_rate": 0.5240384615384616
        },
        "wikipedia_uz": {
          "samples": 1,
          "pass": 0,
          "minor": 1,
          "fail": 0,
          "uncertain": 0,
          "strict_pass_rate": 0.0
        }
      },
      "top_non_pass_reasons": {
        "Minor stylistic difference in phrasing, but meaning is fully preserved.": 3,
        "Minor issue with naturalness and fluency, but meaning is fully preserved.": 2,
        "Minor wording difference, but meaning is fully preserved.": 2,
        "Minor stylistic issue with word order, but meaning is fully preserved.": 2,
        "Minor stylistic improvement, but meaning is fully preserved.": 2,
        "Minor stylistic improvement possible, but meaning is fully preserved.": 2,
        "错误地将 'Xiva' 翻译为 '哈维'，'Xiva' 是乌兹别克斯坦的一个地区，应音译为 '希瓦'。": 1,
        "Significant semantic inconsistencies, omissions, and additions; mistranslation of key terms like 'markazlashtirilgan' and 'qimor o'ynash'; entity error with '1win' being a brand name that should be properly translated or retained as is.": 1,
        "严重语义不一致，遗漏关键信息，添加无意义字符，实体翻译错误": 1,
        "Semantic inconsistency, omissions, additions, and mistranslations affect meaning and fluency.": 1,
        "Minor stylistic issue; 'Tao Fortune' is translated as a proper noun, which is acceptable, but the structure could be slightly more natural in Chinese.": 1,
        "Semantic inconsistency and mistranslation; omission of key terms like 'block' and 'gambling problem'; addition of irrelevant terms like 'field website'; entity error with 'field website' instead of 'casino websites'.": 1,
        "The translation is semantically accurate but slightly informal and uses '投注' (betting) which may not be the most natural term for the context. The original text is about 'pul tikish' (money repair), which is more likely to mean 'repairing money' or 'reimbursement' rather than 'betting'.": 1,
        "Incomplete and mistranslated scientific terminology; missing key concepts like 'magnit dipol momenti' and 'atom orbitalini' which are essential for understanding the original meaning.": 1,
        "Minor fluency issue with '技巧和技巧' which is redundant. The translation is otherwise accurate and fully usable.": 1,
        "Minor stylistic issue with the phrase '城市有2个地区报纸' which could be more natural as '有2份地区报纸' or similar.": 1,
        "Minor stylistic choice in parentheses usage, but meaning is fully preserved.": 1,
        "Minor issue with '在线场' which is less natural than '在线赌场' but still understandable.": 1,
        "严重语义不一致，遗漏关键信息，添加不相关词汇，实体错误（如'博'未明确指代），整体翻译不可用": 1,
        "严重语义错误，遗漏关键信息，添加不相关内容，实体错误": 1
      }
    }
  },
  "recommendation": "DO_NOT_RUN_FULL_AUDIT"
}
```

### reports/experiments/zh_uz_flores_like_v2/validation.json

```json
{
  "schema_version": 1,
  "status": "PASS",
  "errors": []
}
```

### reports/experiments/zh_uz_flores_relaxed_8k/assembly.json

```json
{
  "schema_version": 1,
  "status": "FLORES_LIKE_RELAXED_8K_BUILT_NOT_TRAINED",
  "rows": 45521,
  "new_teacher_rows": {
    "uz-zh": 8000,
    "zh-uz": 8000
  },
  "raw_mass": {
    "zh-uz|human": 4275.0,
    "zh-uz|existing_kd": 8208.5,
    "zh-uz|flores_like_kd": 4499.2,
    "uz-zh|human": 4268.0,
    "uz-zh|existing_kd": 10911.0,
    "uz-zh|flores_like_kd": 6536.0
  },
  "weight_multipliers": {
    "zh-uz|human": 2.129637426900585,
    "zh-uz|existing_kd": 0.8318389474325394,
    "zh-uz|flores_like_kd": 1.5176364687055477,
    "uz-zh|human": 2.133130271790066,
    "uz-zh|existing_kd": 0.6258042342590047,
    "uz-zh|flores_like_kd": 1.0446985924112606
  },
  "effective_mass": {
    "zh-uz|human": 9104.200000000004,
    "zh-uz|existing_kd": 6828.149999999999,
    "zh-uz|flores_like_kd": 6828.149999999998,
    "uz-zh|human": 9104.200000000003,
    "uz-zh|existing_kd": 6828.15,
    "uz-zh|flores_like_kd": 6828.149999999999
  },
  "target_global_shares": {
    "zh-uz|human": 0.2,
    "zh-uz|existing_kd": 0.15,
    "zh-uz|flores_like_kd": 0.15,
    "uz-zh|human": 0.2,
    "uz-zh|existing_kd": 0.15,
    "uz-zh|flores_like_kd": 0.15
  },
  "train": {
    "path": "/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/flores_relaxed_8k/train.jsonl",
    "sha256": "83a0ccbc8233c6629ca6e5fab020efadcae051fd804b80a2b43f3a7ebab29874"
  },
  "validation": {
    "path": "/root/autodl-tmp/fourlang_translation/data/distillation/zh_uz/flores_relaxed_8k/validation.jsonl",
    "sha256": "1d537dab6b47ae7f5c77ed5dc82b7f2c52a9e3a4c860b16edd947dfad9ea70ac"
  },
  "flores_rows_in_training": 0,
  "existing_v4_modified": false
}
```

### reports/experiments/zh_uz_flores_relaxed_8k/validation.json

```json
{
  "schema_version": 1,
  "status": "PASS",
  "errors": []
}
```

### results/evaluation/en_ru/exp1/metrics.json

```json
{
  "en-ru": {
    "bleu": 26.729786114748038,
    "chrf2": 51.088756393580994,
    "samples": 3012,
    "benchmarks": {
      "flores_devtest": {
        "bleu": 23.629200730090908,
        "chrf2": 49.28717231470925,
        "samples": 1012
      },
      "tatoeba": {
        "bleu": 32.01425003228796,
        "chrf2": 54.87280832210429,
        "samples": 2000
      }
    }
  },
  "ru-en": {
    "bleu": 29.725619132795785,
    "chrf2": 53.60046100087883,
    "samples": 3012,
    "benchmarks": {
      "flores_devtest": {
        "bleu": 25.806070136694093,
        "chrf2": 51.90711004103484,
        "samples": 1012
      },
      "tatoeba": {
        "bleu": 37.7926293374569,
        "chrf2": 57.08085693762397,
        "samples": 2000
      }
    }
  }
}
```

### results/evaluation/pair_specialists/en_ru/exp1.json

```json
{
  "en-ru": {
    "bleu": 23.63215965465385,
    "chrf2": 49.2916981012424,
    "samples": 1012
  },
  "ru-en": {
    "bleu": 25.811736293324255,
    "chrf2": 51.911983829455,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/en_ru/exp2.json

```json
{
  "en-ru": {
    "bleu": 23.902043762374994,
    "chrf2": 49.629503918334535,
    "samples": 1012
  },
  "ru-en": {
    "bleu": 26.71584930105291,
    "chrf2": 52.73497271966179,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/en_ru/promotion_gate.json

```json
{
  "status": "PASS",
  "directions": [
    {
      "direction": "en-ru",
      "passed": true,
      "exp1": {
        "bleu": 23.63215965465385,
        "chrf2": 49.2916981012424,
        "samples": 1012
      },
      "exp2": {
        "bleu": 23.902043762374994,
        "chrf2": 49.629503918334535,
        "samples": 1012
      }
    },
    {
      "direction": "ru-en",
      "passed": true,
      "exp1": {
        "bleu": 25.811736293324255,
        "chrf2": 51.911983829455,
        "samples": 1012
      },
      "exp2": {
        "bleu": 26.71584930105291,
        "chrf2": 52.73497271966179,
        "samples": 1012
      }
    }
  ]
}
```

### results/evaluation/pair_specialists/en_uz/exp1.json

```json
{
  "en-uz": {
    "bleu": 15.009204124235044,
    "chrf2": 47.57774178959463,
    "samples": 1012
  },
  "uz-en": {
    "bleu": 23.93138846424699,
    "chrf2": 50.3374970460535,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/en_uz/exp2.json

```json
{
  "en-uz": {
    "bleu": 15.323007108177192,
    "chrf2": 47.87027032901657,
    "samples": 1012
  },
  "uz-en": {
    "bleu": 24.47775531337278,
    "chrf2": 50.57860459714267,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/en_uz/promotion_gate.json

```json
{
  "status": "PASS",
  "directions": [
    {
      "direction": "en-uz",
      "passed": true,
      "exp1": {
        "bleu": 15.009204124235044,
        "chrf2": 47.57774178959463,
        "samples": 1012
      },
      "exp2": {
        "bleu": 15.323007108177192,
        "chrf2": 47.87027032901657,
        "samples": 1012
      }
    },
    {
      "direction": "uz-en",
      "passed": true,
      "exp1": {
        "bleu": 23.93138846424699,
        "chrf2": 50.3374970460535,
        "samples": 1012
      },
      "exp2": {
        "bleu": 24.47775531337278,
        "chrf2": 50.57860459714267,
        "samples": 1012
      }
    }
  ]
}
```

### results/evaluation/pair_specialists/en_zh/exp1.json

```json
{
  "en-zh": {
    "bleu": 32.58890424635219,
    "chrf2": 21.443099503614782,
    "samples": 1012
  },
  "zh-en": {
    "bleu": 19.21926776166106,
    "chrf2": 47.37036311762749,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/en_zh/exp2.json

```json
{
  "en-zh": {
    "bleu": 33.01686019508286,
    "chrf2": 21.945770718616156,
    "samples": 1012
  },
  "zh-en": {
    "bleu": 20.190539071084686,
    "chrf2": 48.140038358634264,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/en_zh/promotion_gate.json

```json
{
  "status": "PASS",
  "directions": [
    {
      "direction": "en-zh",
      "passed": true,
      "exp1": {
        "bleu": 32.58890424635219,
        "chrf2": 21.443099503614782,
        "samples": 1012
      },
      "exp2": {
        "bleu": 33.01686019508286,
        "chrf2": 21.945770718616156,
        "samples": 1012
      }
    },
    {
      "direction": "zh-en",
      "passed": true,
      "exp1": {
        "bleu": 19.21926776166106,
        "chrf2": 47.37036311762749,
        "samples": 1012
      },
      "exp2": {
        "bleu": 20.190539071084686,
        "chrf2": 48.140038358634264,
        "samples": 1012
      }
    }
  ]
}
```

### results/evaluation/pair_specialists/uz_ru/exp1.json

```json
{
  "uz-ru": {
    "bleu": 6.577107628549916,
    "chrf2": 27.736407369135012,
    "samples": 1012
  },
  "ru-uz": {
    "bleu": 2.4065308637156217,
    "chrf2": 22.74740659882642,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/uz_ru/exp2.json

```json
{
  "uz-ru": {
    "bleu": 7.277864576497285,
    "chrf2": 29.606147379691585,
    "samples": 1012
  },
  "ru-uz": {
    "bleu": 2.744255131071184,
    "chrf2": 24.77226151486562,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/uz_ru/promotion_gate.json

```json
{
  "status": "PASS",
  "directions": [
    {
      "direction": "uz-ru",
      "passed": true,
      "exp1": {
        "bleu": 6.577107628549916,
        "chrf2": 27.736407369135012,
        "samples": 1012
      },
      "exp2": {
        "bleu": 7.277864576497285,
        "chrf2": 29.606147379691585,
        "samples": 1012
      }
    },
    {
      "direction": "ru-uz",
      "passed": true,
      "exp1": {
        "bleu": 2.4065308637156217,
        "chrf2": 22.74740659882642,
        "samples": 1012
      },
      "exp2": {
        "bleu": 2.744255131071184,
        "chrf2": 24.77226151486562,
        "samples": 1012
      }
    }
  ]
}
```

### results/evaluation/pair_specialists/zh_ru/exp1.json

```json
{
  "zh-ru": {
    "bleu": 12.521395850484703,
    "chrf2": 38.4399467927828,
    "samples": 1012
  },
  "ru-zh": {
    "bleu": 26.508608749363663,
    "chrf2": 18.21796868503126,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/zh_ru/exp2.json

```json
{
  "zh-ru": {
    "bleu": 12.97337863773094,
    "chrf2": 39.245919014809324,
    "samples": 1012
  },
  "ru-zh": {
    "bleu": 27.181061636712702,
    "chrf2": 18.46761207226311,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/zh_ru/promotion_gate.json

```json
{
  "status": "PASS",
  "directions": [
    {
      "direction": "zh-ru",
      "passed": true,
      "exp1": {
        "bleu": 12.521395850484703,
        "chrf2": 38.4399467927828,
        "samples": 1012
      },
      "exp2": {
        "bleu": 12.97337863773094,
        "chrf2": 39.245919014809324,
        "samples": 1012
      }
    },
    {
      "direction": "ru-zh",
      "passed": true,
      "exp1": {
        "bleu": 26.508608749363663,
        "chrf2": 18.21796868503126,
        "samples": 1012
      },
      "exp2": {
        "bleu": 27.181061636712702,
        "chrf2": 18.46761207226311,
        "samples": 1012
      }
    }
  ]
}
```

### results/evaluation/pair_specialists/zh_uz/exp1.json

```json
{
  "zh-uz": {
    "bleu": 4.631407955500526,
    "chrf2": 31.78868138705453,
    "samples": 1012
  },
  "uz-zh": {
    "bleu": 19.07196419983334,
    "chrf2": 13.981110740141464,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/zh_uz/exp2.json

```json
{
  "zh-uz": {
    "bleu": 4.7752320879812835,
    "chrf2": 31.925600672703375,
    "samples": 1012
  },
  "uz-zh": {
    "bleu": 19.094028563551934,
    "chrf2": 13.847286301852415,
    "samples": 1012
  }
}
```

### results/evaluation/pair_specialists/zh_uz/promotion_gate.json

```json
{
  "status": "FAIL",
  "directions": [
    {
      "direction": "zh-uz",
      "passed": true,
      "exp1": {
        "bleu": 4.631407955500526,
        "chrf2": 31.78868138705453,
        "samples": 1012
      },
      "exp2": {
        "bleu": 4.7752320879812835,
        "chrf2": 31.925600672703375,
        "samples": 1012
      }
    },
    {
      "direction": "uz-zh",
      "passed": false,
      "exp1": {
        "bleu": 19.07196419983334,
        "chrf2": 13.981110740141464,
        "samples": 1012
      },
      "exp2": {
        "bleu": 19.094028563551934,
        "chrf2": 13.847286301852415,
        "samples": 1012
      }
    }
  ]
}
```

### results/evaluation/weak_pair_ablation/zh_uz/baseline_exp1.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "baseline_exp1",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 4.4481282845140315,
      "chrf2": 31.597544618696976,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 19.218277583563253,
      "chrf2": 14.049281524101628,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/baseline_exp2.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "baseline_exp2",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp2/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 4.593644461521452,
      "chrf2": 31.73289228953939,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 19.4816383061151,
      "chrf2": 14.277819794610632,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/bidir_full.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "bidir_full",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/bidir_full/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 4.68256602499473,
      "chrf2": 31.999335996498367,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 19.4282642369544,
      "chrf2": 14.3472563221879,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/comparison.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "benchmark": "flores_dev",
  "baseline_variant": "baseline_exp1",
  "thresholds": {
    "noise_floor_chrf2": 0.3,
    "meaningful_gain_chrf2": 1.0
  },
  "winners": {
    "zh-uz": {
      "variant": "flores_relaxed_8k_ep3",
      "direction": "zh-uz",
      "bleu": 5.201786246140437,
      "chrf2": 33.37534264408274,
      "samples": 997,
      "delta_bleu": 0.7536579616264056,
      "delta_chrf2": 1.777798025385767,
      "signal": "meaningful_gain"
    },
    "uz-zh": {
      "variant": "flores_relaxed_8k_ep3",
      "direction": "uz-zh",
      "bleu": 20.073594862856027,
      "chrf2": 14.600897774127091,
      "samples": 997,
      "delta_bleu": 0.8553172792927732,
      "delta_chrf2": 0.5516162500254627,
      "signal": "small_gain"
    }
  },
  "comparisons": [
    {
      "variant": "flores_relaxed_8k_ep3",
      "direction": "zh-uz",
      "bleu": 5.201786246140437,
      "chrf2": 33.37534264408274,
      "samples": 997,
      "delta_bleu": 0.7536579616264056,
      "delta_chrf2": 1.777798025385767,
      "signal": "meaningful_gain"
    },
    {
      "variant": "flores_relaxed_8k",
      "direction": "zh-uz",
      "bleu": 4.984261214919152,
      "chrf2": 32.78750777036767,
      "samples": 997,
      "delta_bleu": 0.5361329304051203,
      "delta_chrf2": 1.1899631516706926,
      "signal": "meaningful_gain"
    },
    {
      "variant": "full_weighted_60_40",
      "direction": "zh-uz",
      "bleu": 4.7789760041257585,
      "chrf2": 32.10596198183988,
      "samples": 997,
      "delta_bleu": 0.330847719611727,
      "delta_chrf2": 0.5084173631429074,
      "signal": "small_gain"
    },
    {
      "variant": "bidir_full",
      "direction": "zh-uz",
      "bleu": 4.68256602499473,
      "chrf2": 31.999335996498367,
      "samples": 997,
      "delta_bleu": 0.23443774048069876,
      "delta_chrf2": 0.4017913778013913,
      "signal": "small_gain"
    },
    {
      "variant": "directional_full__zh_uz",
      "direction": "zh-uz",
      "bleu": 4.6113666698436555,
      "chrf2": 31.87115323528287,
      "samples": 997,
      "delta_bleu": 0.16323838532962398,
      "delta_chrf2": 0.27360861658589286,
      "signal": "flat"
    },
    {
      "variant": "baseline_exp1",
      "direction": "zh-uz",
      "bleu": 4.4481282845140315,
      "chrf2": 31.597544618696976,
      "samples": 997,
      "delta_bleu": 0.0,
      "delta_chrf2": 0.0,
      "signal": "baseline"
    },
    {
      "variant": "full_native_lr2e6",
      "direction": "zh-uz",
      "bleu": 4.523500544933246,
      "chrf2": 31.253219657614046,
      "samples": 997,
      "delta_bleu": 0.07537226041921485,
      "delta_chrf2": -0.3443249610829291,
      "signal": "regression"
    },
    {
      "variant": "flores_relaxed_8k_ep3",
      "direction": "uz-zh",
      "bleu": 20.073594862856027,
      "chrf2": 14.600897774127091,
      "samples": 997,
      "delta_bleu": 0.8553172792927732,
      "delta_chrf2": 0.5516162500254627,
      "signal": "small_gain"
    },
    {
      "variant": "flores_relaxed_8k",
      "direction": "uz-zh",
      "bleu": 19.72175690271379,
      "chrf2": 14.408420181336218,
      "samples": 997,
      "delta_bleu": 0.5034793191505358,
      "delta_chrf2": 0.35913865723459004,
      "signal": "small_gain"
    },
    {
      "variant": "full_weighted_60_40",
      "direction": "uz-zh",
      "bleu": 19.500504447376084,
      "chrf2": 14.405045385469245,
      "samples": 997,
      "delta_bleu": 0.2822268638128307,
      "delta_chrf2": 0.35576386136761684,
      "signal": "small_gain"
    },
    {
      "variant": "bidir_full",
      "direction": "uz-zh",
      "bleu": 19.4282642369544,
      "chrf2": 14.3472563221879,
      "samples": 997,
      "delta_bleu": 0.20998665339114808,
      "delta_chrf2": 0.2979747980862708,
      "signal": "flat"
    },
    {
      "variant": "baseline_exp1",
      "direction": "uz-zh",
      "bleu": 19.218277583563253,
      "chrf2": 14.049281524101628,
      "samples": 997,
      "delta_bleu": 0.0,
      "delta_chrf2": 0.0,
      "signal": "baseline"
    },
    {
      "variant": "full_native_lr2e6",
      "direction": "uz-zh",
      "bleu": 19.20433273994721,
      "chrf2": 13.9368800543068,
      "samples": 997,
      "delta_bleu": -0.013944843616044977,
      "delta_chrf2": -0.11240146979482901,
      "signal": "flat"
    },
    {
      "variant": "directional_full__uz_zh",
      "direction": "uz-zh",
      "bleu": 19.081531705887297,
      "chrf2": 13.912438760578787,
      "samples": 997,
      "delta_bleu": -0.13674587767595625,
      "delta_chrf2": -0.13684276352284108,
      "signal": "flat"
    }
  ],
  "missing_runs_are_omitted": true
}
```

### results/evaluation/weak_pair_ablation/zh_uz/directional_full__uz_zh.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "directional_full",
  "direction": "uz-zh",
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/directional_full__uz_zh/best_model/uz_zh",
  "scores": {
    "uz-zh": {
      "bleu": 19.081531705887297,
      "chrf2": 13.912438760578787,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/directional_full__zh_uz.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "directional_full",
  "direction": "zh-uz",
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/directional_full__zh_uz/best_model/zh_uz",
  "scores": {
    "zh-uz": {
      "bleu": 4.6113666698436555,
      "chrf2": 31.87115323528287,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/final_devtest.json

```json
{
  "schema_version": 1,
  "status": "PASS",
  "pair": "zh_uz",
  "benchmark": "flores_devtest",
  "final_evaluation_policy": "frozen_dev_winner_evaluated_once",
  "selection_evidence": {
    "benchmark": "flores_dev",
    "comparison": "/root/autodl-tmp/fourlang_translation/results/evaluation/weak_pair_ablation/zh_uz/comparison.json",
    "comparison_sha256": "205177e62f35bfeb42399082954e715944bf3f3d6011f61c0013b237929ef1ad",
    "winner_in_both_directions": "flores_relaxed_8k_ep3"
  },
  "benchmark_evidence": {
    "path": "/root/autodl-tmp/fourlang_translation/data/benchmark/fourlang/flores_devtest.parquet",
    "sha256": "933a9c29300d65ca6c1b85345dce5922dfe78c4797c5b306c0481f33d25db41e"
  },
  "baseline": {
    "variant": "baseline_exp1",
    "model_path": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
    "scores": {
      "zh-uz": {
        "bleu": 4.631407955500526,
        "chrf2": 31.78868138705453,
        "samples": 1012
      },
      "uz-zh": {
        "bleu": 19.07196419983334,
        "chrf2": 13.981110740141464,
        "samples": 1012
      }
    }
  },
  "candidate": {
    "variant": "flores_relaxed_8k_ep3",
    "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared",
    "scores": {
      "zh-uz": {
        "bleu": 5.487347352220015,
        "chrf2": 33.69348543455878,
        "samples": 1012
      },
      "uz-zh": {
        "bleu": 20.2645993723638,
        "chrf2": 14.548632726792379,
        "samples": 1012
      }
    }
  },
  "directions": [
    {
      "direction": "zh-uz",
      "passed": true,
      "metric_checks": {
        "bleu": true,
        "chrf2": true
      },
      "baseline": {
        "bleu": 4.631407955500526,
        "chrf2": 31.78868138705453,
        "samples": 1012
      },
      "candidate": {
        "bleu": 5.487347352220015,
        "chrf2": 33.69348543455878,
        "samples": 1012
      },
      "delta_bleu": 0.8559393967194886,
      "delta_chrf2": 1.9048040475042498
    },
    {
      "direction": "uz-zh",
      "passed": true,
      "metric_checks": {
        "bleu": true,
        "chrf2": true
      },
      "baseline": {
        "bleu": 19.07196419983334,
        "chrf2": 13.981110740141464,
        "samples": 1012
      },
      "candidate": {
        "bleu": 20.2645993723638,
        "chrf2": 14.548632726792379,
        "samples": 1012
      },
      "delta_bleu": 1.192635172530462,
      "delta_chrf2": 0.5675219866509149
    }
  ]
}
```

### results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "flores_relaxed_8k",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 4.984261214919152,
      "chrf2": 32.78750777036767,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 19.72175690271379,
      "chrf2": 14.408420181336218,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "flores_relaxed_8k_ep3",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 5.201786246140437,
      "chrf2": 33.37534264408274,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 20.073594862856027,
      "chrf2": 14.600897774127091,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/full_native_lr2e6.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "full_native_lr2e6",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/full_native_lr2e6/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 4.523500544933246,
      "chrf2": 31.253219657614046,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 19.20433273994721,
      "chrf2": 13.9368800543068,
      "samples": 997
    }
  }
}
```

### results/evaluation/weak_pair_ablation/zh_uz/full_weighted_60_40.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "full_weighted_60_40",
  "direction": null,
  "benchmark": "flores_dev",
  "model_path": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/full_weighted_60_40/best_model/shared",
  "scores": {
    "zh-uz": {
      "bleu": 4.7789760041257585,
      "chrf2": 32.10596198183988,
      "samples": 997
    },
    "uz-zh": {
      "bleu": 19.500504447376084,
      "chrf2": 14.405045385469245,
      "samples": 997
    }
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/bidir_full/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "bidir_full",
  "direction": null,
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 29521,
    "validation_samples": 1000,
    "train_loss": 1.6490034473291069,
    "train_loss_scope": "full_run",
    "seconds": 471.4886140823364,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/bidir_full/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1846,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "a4bff2cb1664ed5eef28dbb2c134b429c40bd7560519167db49965442c467cc6",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/bidir_full/checkpoints/shared/checkpoint-1846",
    "best_metric": 40.711318850092844,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 30.56532852127973,
      "eval_macro_chrf2": 41.93010986882537,
      "eval_macro_loss": 1.8622659611701966,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -1.2187910187325244
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/directional_full__uz_zh/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "directional_full",
  "direction": "uz-zh",
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "uz-zh",
    "train_samples": 15179,
    "validation_samples": 500,
    "train_loss": 1.4305599814967105,
    "train_loss_scope": "full_run",
    "seconds": 223.95391297340393,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/directional_full__uz_zh/best_model/uz_zh",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 950,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "8161e9381c324f1f1af6bb64611e26f925df2e3b20ff912f2bc01284923bd345",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/directional_full__uz_zh/checkpoints/uz_zh/checkpoint-475",
    "best_metric": 37.46712766374454,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_macro_bleu": 35.889531318994656,
      "eval_macro_chrf2": 38.63580045411069,
      "eval_macro_loss": 1.8319190311431885,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -1.168672790366145
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/directional_full__zh_uz/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "directional_full",
  "direction": "zh-uz",
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "zh-uz",
    "train_samples": 14342,
    "validation_samples": 500,
    "train_loss": 1.9519320957379245,
    "train_loss_scope": "full_run",
    "seconds": 260.331547498703,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/directional_full__zh_uz/best_model/zh_uz",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 898,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "a783e8a3f838ae19221b84fce61406052c69521aa62563b8cdcf8689c787cbe4",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/directional_full__zh_uz/checkpoints/zh_uz/checkpoint-898",
    "best_metric": 44.48975664117847,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 25.241125723564807,
      "eval_macro_chrf2": 45.224419283540044,
      "eval_macro_loss": 1.8926128911972047,
      "eval_worst_chrf2": 45.224419283540044
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -0.7346626423615774
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "flores_relaxed_8k",
  "direction": null,
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 45521,
    "validation_samples": 1000,
    "train_loss": 1.630679357747167,
    "train_loss_scope": "full_run",
    "seconds": 667.3823182582855,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 2846,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "c1bb178a3321ed305e723615444b6bf0b3fca1d43080a026fd6a4c59f5fd7835",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints/shared/checkpoint-2846",
    "best_metric": 41.5207308572346,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 30.56532852127973,
      "eval_macro_chrf2": 41.93010986882537,
      "eval_macro_loss": 1.8622659611701966,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -0.4093790115907723
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "flores_relaxed_8k_ep3",
  "direction": null,
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 45521,
    "validation_samples": 1000,
    "train_loss": 1.5635194329417934,
    "train_loss_scope": "full_run",
    "seconds": 1008.9333786964417,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared",
    "seed": 2026,
    "epochs": 3.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 4269,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "2ad177ad57abb5a525773e7236c9d1243e60f686b439858386be7311c10312c5",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints/shared/checkpoint-4269",
    "best_metric": 42.471031896764686,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 30.56532852127973,
      "eval_macro_chrf2": 41.93010986882537,
      "eval_macro_loss": 1.8622659611701966,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": 0.5409220279393168
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/full_native_lr2e6/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "full_native_lr2e6",
  "direction": null,
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 29521,
    "validation_samples": 1000,
    "train_loss": 1.7352679449972068,
    "train_loss_scope": "full_run",
    "seconds": 488.7579300403595,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/full_native_lr2e6/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1846,
    "learning_rate": 2e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "0dbd5e225806768534a2ab0842659720b95a5640638f5725a4b429942b283662",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/full_native_lr2e6/checkpoints/shared/checkpoint-1846",
    "best_metric": 40.49884235017606,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 30.56532852127973,
      "eval_macro_chrf2": 41.93010986882537,
      "eval_macro_loss": 1.8622659611701966,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -1.4312675186493067
  }
}
```

### results/experiments/weak_pair_ablation/zh_uz/full_weighted_60_40/train_report.json

```json
{
  "schema_version": 1,
  "pair": "zh_uz",
  "variant": "full_weighted_60_40",
  "direction": null,
  "source_model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 29521,
    "validation_samples": 1000,
    "train_loss": 1.6413026357136207,
    "train_loss_scope": "full_run",
    "seconds": 502.65176916122437,
    "model": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/full_weighted_60_40/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1846,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "52bb5929344716cf4cc0b397c50b917c2de2b13857e45029d8028083fa556f18",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/experiments/weak_pair_ablation/zh_uz/full_weighted_60_40/checkpoints/shared/checkpoint-1846",
    "best_metric": 40.90809471803379,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 30.56532852127973,
      "eval_macro_chrf2": 41.93010986882537,
      "eval_macro_loss": 1.8622659611701966,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -1.0220151507915816
  }
}
```

### results/student/en_ru/exp1/train_report.json

```json
{
  "experiment": "exp1",
  "full_parameter_finetuning": true,
  "distillation": false,
  "layout": "shared_bidirectional",
  "models": [
    {
      "direction": "shared_bidirectional",
      "train_samples": 122432,
      "validation_samples": 6000,
      "train_loss": 1.3205616285128343,
      "seconds": 2172.112684726715,
      "model": "/root/autodl-tmp/fourlang_translation/results/student/en_ru/exp1/best_model/shared",
      "seed": 2026,
      "epochs": 3.0,
      "physical_batch_size": 16,
      "gradient_accumulation_steps": 2,
      "effective_batch_size": 32,
      "planned_optimizer_steps": 11478,
      "learning_rate": 3e-05,
      "optimizer": "adamw_torch_fused"
    }
  ]
}
```

### results/student/pair_specialists/en_ru/exp2/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 6000,
    "train_loss": 0.7788540405273437,
    "train_loss_scope": "full_run",
    "seconds": 340.9368360042572,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_ru/exp2/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1250,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "ceab506db32adf7466b4191a9fab2301a88ed42052ed8bf1d258058074c6f28d",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_ru/exp2/checkpoints/shared/checkpoint-625",
    "best_metric": 56.439530066697415,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_en-ru_bleu": 26.231047228530933,
      "eval_en-ru_chrf2": 52.14451104350899,
      "eval_en-ru_samples": 200,
      "eval_en-ru_loss": 1.267203254699707,
      "eval_ru-en_bleu": 35.43466554774751,
      "eval_ru-en_chrf2": 59.360890370897266,
      "eval_ru-en_samples": 200,
      "eval_ru-en_loss": 1.2939929866790771,
      "eval_macro_bleu": 30.83285638813922,
      "eval_macro_chrf2": 55.75270070720313,
      "eval_macro_loss": 1.2805981206893922,
      "eval_worst_chrf2": 52.14451104350899
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": 0.6868293594942827
  }
}
```

### results/student/pair_specialists/en_zh/exp1/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 6630,
    "train_loss": 1.251000048828125,
    "train_loss_scope": "full_run",
    "seconds": 357.6971035003662,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_zh/exp1/best_model/shared",
    "seed": 2026,
    "epochs": 3.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1875,
    "learning_rate": 3e-05,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "c2ddcb604391800fa322bbacaba2601ac091c2293bc100434586e32102d8a96f",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_zh/exp1/checkpoints/shared/checkpoint-1250",
    "best_metric": 1.3277719020843506,
    "metric_for_best_model": "eval_loss",
    "initial_validation": null,
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false
  }
}
```

### results/student/pair_specialists/en_zh/exp2/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 6630,
    "train_loss": 1.103502490234375,
    "train_loss_scope": "full_run",
    "seconds": 301.69406938552856,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_zh/exp2/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1250,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "0e151005eb072eade31aa22cdd53c8e1fda4b1e460547a09feeacd643c2003ec",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/en_zh/exp2/checkpoints/shared/checkpoint-1250",
    "best_metric": 41.988735388379176,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_en-zh_bleu": 40.99785334897308,
      "eval_en-zh_chrf2": 30.57022088361171,
      "eval_en-zh_samples": 200,
      "eval_en-zh_loss": 1.4529237508773805,
      "eval_zh-en_bleu": 29.512781490915728,
      "eval_zh-en_chrf2": 52.92836155408474,
      "eval_zh-en_samples": 200,
      "eval_zh-en_loss": 1.2767919969558716,
      "eval_macro_bleu": 35.255317419944404,
      "eval_macro_chrf2": 41.74929121884823,
      "eval_macro_loss": 1.364857873916626,
      "eval_worst_chrf2": 30.57022088361171
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": 0.23944416953094816
  }
}
```

### results/student/pair_specialists/uz_ru/exp1/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 3000,
    "train_loss": 2.1758512044270835,
    "train_loss_scope": "full_run",
    "seconds": 347.05951714515686,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/uz_ru/exp1/best_model/shared",
    "seed": 2026,
    "epochs": 3.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1875,
    "learning_rate": 3e-05,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "0c4c96505bb96ccac9a1cd57fb4441b4302a02a865924d361114d147554555cc",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/uz_ru/exp1/checkpoints/shared/checkpoint-1875",
    "best_metric": 1.9414770603179932,
    "metric_for_best_model": "eval_loss",
    "initial_validation": null,
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false
  }
}
```

### results/student/pair_specialists/uz_ru/exp2/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 3000,
    "train_loss": 1.5601135986328125,
    "train_loss_scope": "full_run",
    "seconds": 284.0122616291046,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/uz_ru/exp2/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1250,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "7ad2340142b82b582addb58327ec453cd93e1823b2608c4e6f88a3df24c05a3e",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/uz_ru/exp2/checkpoints/shared/checkpoint-1250",
    "best_metric": 37.08774956212476,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_ru-uz_bleu": 18.91273890122576,
      "eval_ru-uz_chrf2": 34.83232348704929,
      "eval_ru-uz_samples": 200,
      "eval_ru-uz_loss": 2.3313164710998535,
      "eval_uz-ru_bleu": 17.253462870436177,
      "eval_uz-ru_chrf2": 37.21601706622155,
      "eval_uz-ru_samples": 200,
      "eval_uz-ru_loss": 1.6801200008392334,
      "eval_macro_bleu": 18.08310088583097,
      "eval_macro_chrf2": 36.02417027663542,
      "eval_macro_loss": 2.0057182359695434,
      "eval_worst_chrf2": 34.83232348704929
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": 1.0635792854893467
  }
}
```

### results/student/pair_specialists/zh_ru/exp1/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 4000,
    "train_loss": 2.5419947591145835,
    "train_loss_scope": "full_run",
    "seconds": 502.87024760246277,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_ru/exp1/best_model/shared",
    "seed": 2026,
    "epochs": 3.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1875,
    "learning_rate": 3e-05,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "b335c21c27f43403c8eda3d51897323d6619286cf0b99b35aca03c03b4ed8ba4",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_ru/exp1/checkpoints/shared/checkpoint-1875",
    "best_metric": 2.4314239025115967,
    "metric_for_best_model": "eval_loss",
    "initial_validation": null,
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false
  }
}
```

### results/student/pair_specialists/zh_ru/exp2/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 4000,
    "train_loss": 1.6759184814453125,
    "train_loss_scope": "full_run",
    "seconds": 428.4257161617279,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_ru/exp2/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1250,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "99c20a92c92d9fbeb46d29cd43e2c45cc64f6cc99b79925a0b0482b95166e451",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_ru/exp2/checkpoints/shared/checkpoint-625",
    "best_metric": 27.44682067417409,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_ru-zh_bleu": 20.297556658793827,
      "eval_ru-zh_chrf2": 17.03733683750545,
      "eval_ru-zh_samples": 200,
      "eval_ru-zh_loss": 2.855306930541992,
      "eval_zh-ru_bleu": 11.424815753371766,
      "eval_zh-ru_chrf2": 37.92904364402908,
      "eval_zh-ru_samples": 200,
      "eval_zh-ru_loss": 2.0395460319519043,
      "eval_macro_bleu": 15.861186206082795,
      "eval_macro_chrf2": 27.483190240767264,
      "eval_macro_loss": 2.447426481246948,
      "eval_worst_chrf2": 17.03733683750545
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -0.03636956659317292
  }
}
```

### results/student/pair_specialists/zh_uz/exp1/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 1000,
    "train_loss": 1.8743291341145834,
    "train_loss_scope": "full_run",
    "seconds": 408.1047103404999,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/best_model/shared",
    "seed": 2026,
    "epochs": 3.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1875,
    "learning_rate": 3e-05,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "7ff3d535be0a6117e73cdd26e5fa84d1e5802a523bb4fb81f2b8d535172f79fe",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp1/checkpoints/shared/checkpoint-1875",
    "best_metric": 1.8756541013717651,
    "metric_for_best_model": "eval_loss",
    "initial_validation": null,
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false
  }
}
```

### results/student/pair_specialists/zh_uz/exp2/train_report.json

```json
{
  "single_bidirectional_model": true,
  "model": {
    "direction": "shared_bidirectional",
    "train_samples": 20000,
    "validation_samples": 1000,
    "train_loss": 1.6248020263671874,
    "train_loss_scope": "full_run",
    "seconds": 398.7096176147461,
    "model": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp2/best_model/shared",
    "seed": 2026,
    "epochs": 2.0,
    "physical_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 32,
    "planned_optimizer_steps": 1250,
    "learning_rate": 5e-06,
    "optimizer": "adamw_torch_fused",
    "resumed_from_checkpoint": null,
    "run_fingerprint": "967bba87a1594839c3fcea94f6f0532bce046320f41f72bb77483d4eb2c313da",
    "best_model_checkpoint": "/root/autodl-tmp/fourlang_translation/results/student/pair_specialists/zh_uz/exp2/checkpoints/shared/checkpoint-1250",
    "best_metric": 41.02092962315028,
    "metric_for_best_model": "eval_macro_chrf2",
    "initial_validation": {
      "eval_uz-zh_bleu": 35.889531318994656,
      "eval_uz-zh_chrf2": 38.63580045411069,
      "eval_uz-zh_samples": 200,
      "eval_uz-zh_loss": 1.8319190311431885,
      "eval_zh-uz_bleu": 25.241125723564807,
      "eval_zh-uz_chrf2": 45.224419283540044,
      "eval_zh-uz_samples": 200,
      "eval_zh-uz_loss": 1.8926128911972047,
      "eval_macro_bleu": 30.56532852127973,
      "eval_macro_chrf2": 41.93010986882537,
      "eval_macro_loss": 1.8622659611701966,
      "eval_worst_chrf2": 38.63580045411069
    },
    "checkpoint_interval_steps": 1000,
    "recovered_final_export": false,
    "best_macro_chrf2_delta": -0.9091802456750884
  }
}
```
