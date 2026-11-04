import pandas as pd
from part_a import CLEAN, validate
validate(*(pd.read_csv(CLEAN/f'{name}.csv') for name in ('customers','products','orders')))
