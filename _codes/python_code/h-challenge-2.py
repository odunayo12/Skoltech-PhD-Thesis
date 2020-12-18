# %%
import math
import os
import random
import re
import sys
# %%
# Complete the solve function below.


def solve(meal_cost, tip_percent, tax_percent):
    if __name__ == '__main__':
        meal_cost = float(input(meal_cost))

        tip_percent = (int(input(tip_percent))/100)*meal_cost

        tax_percent = (int(input(tax_percent))/100)*meal_cost

        total_cost = round(meal_cost+tax_percent+tip_percent)

        return total_cost


solve(15, 20, 8)

# %%
