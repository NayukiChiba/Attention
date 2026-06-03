from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def get_dir(path: Path) -> Path:
    """确保目录存在,不存在则创建"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

# 数据集位置
DATASETS_DIR = get_dir(ROOT / 'datasets')
RAW_DATASETS_DIR = get_dir(DATASETS_DIR / 'raw')
PROCESSED_DATASETS_DIR = get_dir(DATASETS_DIR / 'processed')

# 产出位置
OUTPUT_DIR = get_dir(ROOT / 'output')

CHECKPOINTS_DIR = get_dir(OUTPUT_DIR / 'checkpoints')
LOGS_DIR = get_dir(OUTPUT_DIR / 'logs')
TENSORBOARD_DIR = get_dir(OUTPUT_DIR / 'tensorboard')
FIGURES_DIR = get_dir(OUTPUT_DIR / 'figures')

# 模型保存位置
BEST_MODEL_PATH = CHECKPOINTS_DIR / 'best_model.pth'
LAST_MODEL_PATH = CHECKPOINTS_DIR / 'last_model.pth'

# 原始数据集文件
RAW_TRAIN_DATASET_PATH = RAW_DATASETS_DIR / 'train.csv'
RAW_TEST_DATASET_PATH = RAW_DATASETS_DIR / 'test.csv'
RAW_VAL_DATASET_PATH = RAW_DATASETS_DIR / 'val.csv'

# 处理后数据集文件
PROCESSED_TRAIN_DATASET_PATH = PROCESSED_DATASETS_DIR / 'train_processed.csv'
PROCESSED_TEST_DATASET_PATH = PROCESSED_DATASETS_DIR / 'test_processed.csv'
PROCESSED_VAL_DATASET_PATH = PROCESSED_DATASETS_DIR / 'val_processed.csv'