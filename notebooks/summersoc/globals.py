from pathlib import Path

DIR_METRICS = Path('../../statics/agent_experience')
PATH_METRICS_DEMO_EXPLORE = DIR_METRICS / 'metrics_DEMO_EXPLORE.csv'
PATH_METRICS_DEMO_EXPLOIT = DIR_METRICS / 'metrics_DEMO_OPERATE.csv'

SPLIT_DATA_INTO_X_PARTS = 60
ITERATE_THROUGH_X_PARTS = 60

PATH_MODEL_LIST = Path('./') / '2_analysis_models.joblib'

